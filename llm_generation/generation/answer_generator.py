"""
Answer Generator Module
Generates answers from retrieved context using multiple LLM backends
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import os

from llm_generation.prompt.prompt_templates import (
    PromptTemplate,
    PromptTemplateLibrary,
    PromptType,
    format_context_with_sources
)
from retrieval_pipeline.pipeline import RetrievalResult

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """Supported LLM backends"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


@dataclass
class GenerationConfig:
    """
    Configuration for answer generation
    
    Attributes:
        backend: LLM backend to use
        model_name: Model name for the backend
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Nucleus sampling parameter
        presence_penalty: Presence penalty for token repetition
        frequency_penalty: Frequency penalty for token repetition
        timeout_seconds: Generation timeout
        retry_attempts: Number of retry attempts on failure
        stream: Whether to stream responses
    """
    backend: LLMBackend = LLMBackend.OPENAI
    model_name: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    timeout_seconds: int = 30
    retry_attempts: int = 2
    stream: bool = False
    
    def __post_init__(self):
        """Set default model names based on backend"""
        if self.model_name is None:
            default_models = {
                LLMBackend.OPENAI: "gpt-3.5-turbo",
                LLMBackend.OLLAMA: "llama2",
                LLMBackend.ANTHROPIC: "claude-3-sonnet-20240229"
            }
            self.model_name = default_models.get(self.backend, "gpt-3.5-turbo")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "backend": self.backend.value,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "stream": self.stream
        }


@dataclass
class GenerationResult:
    """
    Result of answer generation
    
    Attributes:
        answer: Generated answer text
        query: Original query
        model: Model used for generation
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in completion
        total_tokens: Total tokens used
        generation_time_ms: Time taken for generation
        finish_reason: Reason generation finished
        metadata: Additional metadata
    """
    answer: str
    query: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "answer": self.answer,
            "query": self.query,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "generation_time_ms": self.generation_time_ms,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata
        }


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients
    
    All LLM backends must implement this interface
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer from prompt
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional backend-specific parameters
            
        Returns:
            GenerationResult
        """
        pass
    
    @abstractmethod
    async def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer using chat format
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            config: Generation configuration
            **kwargs: Additional backend-specific parameters
            
        Returns:
            GenerationResult
        """
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client for answer generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (uses env var if None)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Lazy import to avoid dependency issues
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """Generate answer using completion API"""
        # OpenAI's newer models use chat format, so we convert
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_chat(messages, config, **kwargs)
    
    async def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """Generate answer using chat completion API"""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                stream=config.stream,
                **kwargs
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Extract usage information
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            # Extract finish reason
            finish_reason = response.choices[0].finish_reason or "stop"
            
            # Get original query from messages
            query = messages[-1]["content"] if messages else ""
            
            result = GenerationResult(
                answer=answer,
                query=query,
                model=config.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time,
                finish_reason=finish_reason,
                metadata={
                    "backend": "openai",
                    "response_id": response.id,
                    "created": response.created
                }
            )
            
            logger.info(
                f"OpenAI generation completed: {total_tokens} tokens in {generation_time:.0f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class OllamaClient(BaseLLMClient):
    """Ollama API client for local model generation"""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama API base URL (uses env var if None)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        try:
            import ollama
            self.client = ollama.AsyncClient(host=self.base_url)
            logger.info(f"Ollama client initialized at {self.base_url}")
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """Generate answer using completion API"""
        start_time = time.time()
        
        try:
            response = await self.client.generate(
                model=config.model_name,
                prompt=prompt,
                options={
                    "num_predict": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                },
                stream=False,
                **kwargs
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            # Extract answer
            answer = response.get("response", "")
            
            # Estimate tokens (Ollama doesn't always provide exact counts)
            prompt_tokens = response.get("prompt_eval_count", len(prompt.split()) * 1.3)
            completion_tokens = response.get("eval_count", len(answer.split()) * 1.3)
            total_tokens = int(prompt_tokens + completion_tokens)
            
            result = GenerationResult(
                answer=answer,
                query=prompt[:100],  # Use first 100 chars as query
                model=config.model_name,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=total_tokens,
                generation_time_ms=generation_time,
                finish_reason="stop",
                metadata={
                    "backend": "ollama",
                    "model_info": response.get("model", ""),
                    "context": response.get("context", [])
                }
            )
            
            logger.info(
                f"Ollama generation completed: {total_tokens} tokens in {generation_time:.0f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def generate_chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """Generate answer using chat API"""
        start_time = time.time()
        
        try:
            response = await self.client.chat(
                model=config.model_name,
                messages=messages,
                options={
                    "num_predict": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                },
                stream=False,
                **kwargs
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            # Extract answer
            answer = response.get("message", {}).get("content", "")
            
            # Estimate tokens
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Get original query
            query = messages[-1]["content"] if messages else ""
            
            result = GenerationResult(
                answer=answer,
                query=query,
                model=config.model_name,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=total_tokens,
                generation_time_ms=generation_time,
                finish_reason="stop",
                metadata={
                    "backend": "ollama",
                    "model_info": response.get("model", "")
                }
            )
            
            logger.info(
                f"Ollama chat generation completed: {total_tokens} tokens in {generation_time:.0f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ollama chat generation failed: {e}")
            raise


class AnswerGenerator:
    """
    Main answer generator with multi-backend support
    
    Features:
    - Multiple LLM backend support (OpenAI, Ollama, Anthropic)
    - Automatic prompt template selection
    - Context formatting with source attribution
    - Retry logic for transient failures
    - Comprehensive error handling
    - Performance tracking
    """
    
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        custom_client: Optional[BaseLLMClient] = None
    ):
        """
        Initialize the answer generator
        
        Args:
            config: Generation configuration (uses default if None)
            custom_client: Custom LLM client (auto-creates if None)
        """
        self.config = config or GenerationConfig()
        
        # Initialize LLM client
        if custom_client:
            self.client = custom_client
        else:
            self.client = self._create_client(self.config.backend)
        
        # Statistics
        self._stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_tokens": 0,
            "average_time_ms": 0.0
        }
        
        logger.info(f"AnswerGenerator initialized with {self.config.backend.value} backend")
    
    def _create_client(self, backend: LLMBackend) -> BaseLLMClient:
        """
        Create LLM client for specified backend
        
        Args:
            backend: LLM backend type
            
        Returns:
            BaseLLMClient instance
        """
        if backend == LLMBackend.OPENAI:
            return OpenAIClient()
        elif backend == LLMBackend.OLLAMA:
            return OllamaClient()
        elif backend == LLMBackend.ANTHROPIC:
            raise NotImplementedError("Anthropic backend not yet implemented")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    async def generate(
        self,
        retrieval_result: RetrievalResult,
        prompt_template: Optional[PromptTemplate] = None,
        custom_config: Optional[GenerationConfig] = None,
        include_sources: bool = True,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer from retrieval result
        
        Args:
            retrieval_result: RetrievalResult with query and context
            prompt_template: Optional custom prompt template
            custom_config: Optional custom generation config
            include_sources: Whether to include source attribution in context
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with generated answer
        """
        config = custom_config or self.config
        
        # Select appropriate prompt template
        if prompt_template is None:
            prompt_template = PromptTemplateLibrary.get_template_for_query(
                retrieval_result.query
            )
        
        # Override config with template settings if provided
        if prompt_template.max_tokens:
            config.max_tokens = prompt_template.max_tokens
        if prompt_template.temperature:
            config.temperature = prompt_template.temperature
        
        # Format context
        context_text = retrieval_result.final_context.text
        
        if include_sources and retrieval_result.answer_sources:
            context_text = format_context_with_sources(
                context_text,
                retrieval_result.answer_sources,
                include_metadata=True
            )
        
        # Generate answer with retry logic
        result = await self._generate_with_retry(
            query=retrieval_result.query,
            context=context_text,
            prompt_template=prompt_template,
            config=config,
            **kwargs
        )
        
        # Add retrieval metadata to result
        result.metadata.update({
            "retrieval_time_ms": retrieval_result.total_time_ms,
            "num_sources": len(retrieval_result.answer_sources),
            "context_length": len(context_text),
            "total_candidates": retrieval_result.total_candidates,
            "final_results": retrieval_result.final_results
        })
        
        return result
    
    async def _generate_with_retry(
        self,
        query: str,
        context: str,
        prompt_template: PromptTemplate,
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer with retry logic
        
        Args:
            query: User query
            context: Retrieved context
            prompt_template: Prompt template to use
            config: Generation configuration
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult
        """
        last_exception = None
        
        for attempt in range(config.retry_attempts + 1):
            try:
                result = await self._perform_generation(
                    query, context, prompt_template, config, **kwargs
                )
                
                # Update statistics
                self._update_stats(result, success=True)
                
                if attempt > 0:
                    logger.info(f"Generation succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < config.retry_attempts:
                    logger.warning(
                        f"Generation attempt {attempt + 1} failed: {e}. Retrying..."
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Generation failed after {attempt + 1} attempts: {e}")
        
        # All retries failed
        self._update_stats(None, success=False)
        raise last_exception
    
    async def _perform_generation(
        self,
        query: str,
        context: str,
        prompt_template: PromptTemplate,
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """
        Perform actual generation
        
        Args:
            query: User query
            context: Retrieved context
            prompt_template: Prompt template
            config: Generation configuration
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult
        """
        # Format messages for chat-based models
        messages = prompt_template.to_messages(query, context, **kwargs)
        
        # Generate with timeout
        try:
            generation_task = self.client.generate_chat(messages, config, **kwargs)
            result = await asyncio.wait_for(generation_task, timeout=config.timeout_seconds)
            
            logger.info(
                f"Generated answer for query: '{query[:50]}...' "
                f"({result.total_tokens} tokens in {result.generation_time_ms:.0f}ms)"
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Generation timeout after {config.timeout_seconds}s")
            raise TimeoutError("Answer generation timed out")
    
    def _update_stats(self, result: Optional[GenerationResult], success: bool) -> None:
        """
        Update generation statistics
        
        Args:
            result: Generation result (None if failed)
            success: Whether generation succeeded
        """
        self._stats["total_generations"] += 1
        
        if success and result:
            self._stats["successful_generations"] += 1
            self._stats["total_tokens"] += result.total_tokens
            
            # Update running average time
            n = self._stats["successful_generations"]
            old_avg = self._stats["average_time_ms"]
            self._stats["average_time_ms"] = (
                old_avg + (result.generation_time_ms - old_avg) / n
            )
        else:
            self._stats["failed_generations"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get generator statistics
        
        Returns:
            Dictionary of statistics
        """
        success_rate = 0.0
        if self._stats["total_generations"] > 0:
            success_rate = (
                self._stats["successful_generations"] / self._stats["total_generations"]
            )
        
        return {
            **self._stats,
            "success_rate": success_rate,
            "config": self.config.to_dict()
        }
    
    async def generate_from_text(
        self,
        query: str,
        context: str,
        prompt_type: PromptType = PromptType.STANDARD,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer from raw text (utility method)
        
        Args:
            query: User query
            context: Context text
            prompt_type: Type of prompt to use
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult
        """
        prompt_template = PromptTemplateLibrary.get_template(prompt_type)
        
        return await self._generate_with_retry(
            query=query,
            context=context,
            prompt_template=prompt_template,
            config=self.config,
            **kwargs
        )

