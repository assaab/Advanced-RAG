"""
RankLLaMA/RankZephyr Listwise LLM Reranker  
Final stage listwise reranking with LLM reasoning (20 → 5)
"""
import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# For HuggingFace LLM inference
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# For API-based LLM inference
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMRerankConfig:
    """Configuration for LLM reranker"""
    model_name: str = "microsoft/RankZephyr-7B-beta"  # HuggingFace model name
    model_type: str = "huggingface"  # "huggingface", "openai", "simple"
    max_input_length: int = 4096
    max_candidates: int = 20
    final_top_k: int = 5
    temperature: float = 0.1
    use_reasoning: bool = True  # Include reasoning in prompt
    batch_size: int = 1  # LLMs typically process one at a time
    device: Optional[str] = None
    
    # OpenAI specific settings
    openai_model: str = "gpt-3.5-turbo"  # For OpenAI API
    openai_api_key: Optional[str] = None  # Set from environment
    openai_max_tokens: int = 1000
    
    # HuggingFace specific settings
    hf_use_auth_token: bool = False
    hf_cache_dir: Optional[str] = None


@dataclass
class LLMRerankResult:
    """LLM reranking result"""
    chunk_id: str
    document_id: str
    llm_rank: int  # 1-based ranking from LLM
    llm_score: float  # Normalized score (1.0 for rank 1, 0.8 for rank 2, etc.)
    reasoning: str  # LLM's reasoning for ranking
    original_score: float  # MonoT5 score
    confidence: float  # LLM confidence in ranking
    text: str
    metadata: Dict[str, Any]


class BaseLLMReranker(ABC):
    """Base class for LLM rerankers"""
    
    def __init__(self, config: LLMRerankConfig):
        self.config = config
        if HF_AVAILABLE and torch.cuda.is_available():
            self.device = config.device or "cuda"
        else:
            self.device = "cpu"
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[LLMRerankResult]:
        """Rerank candidates using LLM"""
        pass
    
    def _create_ranking_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """Create prompt for LLM ranking"""
        
        prompt_parts = [
            "You are an expert information retrieval system. Your task is to rank the following document passages based on their relevance to the given query.",
            "",
            f"Query: {query}",
            "",
            "Please rank the following passages from most relevant (1) to least relevant. Provide your ranking as a JSON list of passage IDs in order of relevance.",
            ""
        ]
        
        # Add candidates
        for i, candidate in enumerate(candidates, 1):
            passage_text = candidate["text"][:500] + ("..." if len(candidate["text"]) > 500 else "")
            prompt_parts.append(f"Passage {i} (ID: {candidate['chunk_id'][:8]}):")
            prompt_parts.append(passage_text)
            prompt_parts.append("")
        
        if self.config.use_reasoning:
            prompt_parts.extend([
                "Instructions:",
                "1. Consider semantic relevance, topical alignment, and completeness of information",
                "2. Focus on passages that directly answer or relate to the query",
                "3. Consider the quality and specificity of information",
                "4. Provide brief reasoning for your top 3 choices",
                "",
                "Response format:",
                "{",
                '  "ranking": [list of passage IDs in order of relevance],',
                '  "reasoning": {',
                '    "1": "Why this passage is most relevant...",',
                '    "2": "Why this passage is second most relevant...",',
                '    "3": "Why this passage is third most relevant..."',
                "  }",
                "}"
            ])
        else:
            prompt_parts.extend([
                "Instructions:",
                "Provide only the ranking as a JSON list of passage IDs in order of relevance.",
                "",
                "Response format:",
                '{"ranking": [list of passage IDs in order]}'
            ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response: str, candidates: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, str]]:
        """Parse LLM response to extract ranking and reasoning"""
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in LLM response, using fallback ranking")
            return [c["chunk_id"] for c in candidates], {}
        
        try:
            result = json.loads(json_match.group())
            ranking = result.get("ranking", [])
            reasoning = result.get("reasoning", {})
            
            # Validate ranking
            candidate_ids = {c["chunk_id"][:8]: c["chunk_id"] for c in candidates}
            
            # Map short IDs back to full IDs
            full_ranking = []
            for short_id in ranking:
                if short_id in candidate_ids:
                    full_ranking.append(candidate_ids[short_id])
                else:
                    # Try exact match
                    for candidate in candidates:
                        if candidate["chunk_id"] == short_id:
                            full_ranking.append(short_id)
                            break
            
            # Add missing candidates
            ranked_set = set(full_ranking)
            for candidate in candidates:
                if candidate["chunk_id"] not in ranked_set:
                    full_ranking.append(candidate["chunk_id"])
            
            return full_ranking, reasoning
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return [c["chunk_id"] for c in candidates], {}
    
    def _convert_to_results(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        ranking: List[str],
        reasoning: Dict[str, str]
    ) -> List[LLMRerankResult]:
        """Convert ranking to LLMRerankResult objects"""
        
        # Create lookup for candidates
        candidate_lookup = {c["chunk_id"]: c for c in candidates}
        
        results = []
        
        for i, chunk_id in enumerate(ranking):
            if chunk_id not in candidate_lookup:
                continue
                
            candidate = candidate_lookup[chunk_id]
            rank = i + 1
            
            # Convert rank to score (rank 1 = 1.0, rank 2 = 0.9, etc.)
            score = max(0.0, 1.0 - (rank - 1) * 0.1)
            
            # Get reasoning
            rank_reasoning = reasoning.get(str(rank), f"Ranked at position {rank}")
            
            # Estimate confidence based on position and reasoning length
            confidence = max(0.5, 1.0 - (rank - 1) * 0.05)
            if len(rank_reasoning) > 50:  # More detailed reasoning = higher confidence
                confidence += 0.1
            confidence = min(1.0, confidence)
            
            result = LLMRerankResult(
                chunk_id=chunk_id,
                document_id=candidate["document_id"],
                llm_rank=rank,
                llm_score=score,
                reasoning=rank_reasoning,
                original_score=candidate.get("monot5_score", 0.0),
                confidence=confidence,
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            
            results.append(result)
        
        return results


class HuggingFaceLLMReranker(BaseLLMReranker):
    """HuggingFace LLM reranker using transformers library"""
    
    def __init__(self, config: LLMRerankConfig):
        super().__init__(config)
        
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch required for HuggingFace LLM reranking")
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
        
        logger.info(f"HuggingFace LLM reranker initialized: {config.model_name}")
    
    def _load_model(self):
        """Load HuggingFace LLM model"""
        try:
            # Set up authentication if needed
            use_auth_token = self.config.hf_use_auth_token
            if use_auth_token and os.getenv("HF_TOKEN"):
                use_auth_token = os.getenv("HF_TOKEN")
            
            # Try loading as causal LM first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_auth_token=use_auth_token,
                cache_dir=self.config.hf_cache_dir
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() and self.device == "cuda" else None,
                trust_remote_code=True,
                use_auth_token=use_auth_token,
                cache_dir=self.config.hf_cache_dir
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info(f"Loaded HuggingFace model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            # Fallback to a smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load smaller fallback model"""
        try:
            fallback_model = "microsoft/DialoGPT-small"  # Small conversational model
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Loaded fallback model: {fallback_model}")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[LLMRerankResult]:
        """Perform HuggingFace LLM reranking"""
        
        if not candidates:
            return []
        
        top_k = top_k or self.config.final_top_k
        candidates = candidates[:self.config.max_candidates]  # Limit input size
        
        logger.info(f"HuggingFace LLM reranking {len(candidates)} candidates → top {top_k}")
        
        # Create ranking prompt
        prompt = self._create_ranking_prompt(query, candidates)
        
        # Truncate prompt if too long
        if len(prompt) > self.config.max_input_length:
            prompt = prompt[:self.config.max_input_length - 100] + "...\n\nPlease rank the passages above."
        
        try:
            # Generate response
            response = await self._generate_response(prompt)
            
            # Parse response
            ranking, reasoning = self._parse_llm_response(response, candidates)
            
            # Convert to results
            results = self._convert_to_results(query, candidates, ranking, reasoning)
            
            logger.info(f"HuggingFace LLM reranking completed: {len(candidates)} → {min(top_k, len(results))}")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"HuggingFace LLM reranking failed: {e}")
            # Fallback: return original order with default scores
            return self._create_fallback_results(candidates, top_k)
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using HuggingFace LLM"""
        
        try:
            # Generate with pipeline
            responses = self.pipeline(
                prompt,
                max_new_tokens=200,  # Limit generation length
                temperature=self.config.temperature,
                do_sample=True if self.config.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1,
                return_full_text=False  # Only return generated part
            )
            
            if responses and len(responses) > 0:
                response = responses[0]["generated_text"].strip()
                return response
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def _create_fallback_results(self, candidates: List[Dict[str, Any]], top_k: int) -> List[LLMRerankResult]:
        """Create fallback results when LLM fails"""
        
        results = []
        
        for i, candidate in enumerate(candidates[:top_k]):
            rank = i + 1
            score = max(0.0, 1.0 - (rank - 1) * 0.1)
            
            result = LLMRerankResult(
                chunk_id=candidate["chunk_id"],
                document_id=candidate["document_id"],
                llm_rank=rank,
                llm_score=score,
                reasoning=f"HuggingFace fallback ranking at position {rank}",
                original_score=candidate.get("monot5_score", 0.0),
                confidence=0.5,  # Low confidence for fallback
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            
            results.append(result)
        
        return results


class OpenAILLMReranker(BaseLLMReranker):
    """OpenAI API-based LLM reranker"""
    
    def __init__(self, config: LLMRerankConfig):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required for OpenAI LLM reranking")
        
        self.client = None
        self._setup_client()
        
        logger.info(f"OpenAI LLM reranker initialized: {config.openai_model}")
    
    def _setup_client(self):
        """Setup OpenAI client"""
        try:
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in config or OPENAI_API_KEY environment variable")
            
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenAI client: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[LLMRerankResult]:
        """Perform OpenAI LLM reranking"""
        
        if not candidates:
            return []
        
        top_k = top_k or self.config.final_top_k
        candidates = candidates[:self.config.max_candidates]
        
        logger.info(f"OpenAI LLM reranking {len(candidates)} candidates → top {top_k}")
        
        # Create ranking prompt
        prompt = self._create_ranking_prompt(query, candidates)
        
        # Truncate prompt if too long
        if len(prompt) > self.config.max_input_length:
            prompt = prompt[:self.config.max_input_length - 100] + "...\n\nPlease rank the passages above."
        
        try:
            # Generate response using OpenAI API
            response = await self._generate_openai_response(prompt)
            
            # Parse response
            ranking, reasoning = self._parse_llm_response(response, candidates)
            
            # Convert to results
            results = self._convert_to_results(query, candidates, ranking, reasoning)
            
            logger.info(f"OpenAI LLM reranking completed: {len(candidates)} → {min(top_k, len(results))}")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"OpenAI LLM reranking failed: {e}")
            # Fallback: return original order with default scores
            return self._create_fallback_results(candidates, top_k)
    
    async def _generate_openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert information retrieval system that ranks documents by relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.openai_max_tokens
            )
            
            if response.choices:
                return response.choices[0].message.content
            else:
                return ""
                
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""
    
    def _create_fallback_results(self, candidates: List[Dict[str, Any]], top_k: int) -> List[LLMRerankResult]:
        """Create fallback results when OpenAI API fails"""
        
        results = []
        
        for i, candidate in enumerate(candidates[:top_k]):
            rank = i + 1
            score = max(0.0, 1.0 - (rank - 1) * 0.1)
            
            result = LLMRerankResult(
                chunk_id=candidate["chunk_id"],
                document_id=candidate["document_id"],
                llm_rank=rank,
                llm_score=score,
                reasoning=f"OpenAI API fallback ranking at position {rank}",
                original_score=candidate.get("monot5_score", 0.0),
                confidence=0.6,
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            
            results.append(result)
        
        return results


class LLMReranker:
    """Main LLM reranker factory"""
    
    @staticmethod
    def create(config: LLMRerankConfig) -> BaseLLMReranker:
        """Create appropriate LLM reranker based on config"""
        
        if config.model_type == "huggingface":
            return HuggingFaceLLMReranker(config)
        elif config.model_type == "openai":
            return OpenAILLMReranker(config)
        elif config.model_type == "simple":
            return SimpleLLMReranker(config)
        else:
            logger.warning(f"Unknown model type {config.model_type}, using simple fallback")
            return SimpleLLMReranker(config)


class SimpleLLMReranker:
    """
    Simplified LLM reranker for cases where full LLMs are not available
    Uses rule-based reasoning to simulate LLM-like reranking
    """
    
    def __init__(self, config: Optional[LLMRerankConfig] = None):
        self.config = config or LLMRerankConfig()
        logger.info("Simple LLM reranker initialized (rule-based)")
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[LLMRerankResult]:
        """Rule-based reranking that simulates LLM reasoning"""
        
        if not candidates:
            return []
        
        top_k = top_k or self.config.final_top_k
        
        # Analyze query for key terms
        query_terms = set(query.lower().split())
        
        # Score candidates based on multiple factors
        scored_candidates = []
        
        for candidate in candidates:
            text = candidate["text"].lower()
            
            # Factor 1: Term overlap
            text_terms = set(text.split())
            overlap_ratio = len(query_terms.intersection(text_terms)) / len(query_terms) if query_terms else 0
            
            # Factor 2: Query term density
            query_term_count = sum(text.count(term) for term in query_terms)
            density = query_term_count / len(text.split()) if text.split() else 0
            
            # Factor 3: Position bias (prefer earlier candidates from MonoT5)
            position_bonus = 1.0 / (candidates.index(candidate) + 1)
            
            # Factor 4: Text quality (length, completeness)
            length_score = min(1.0, len(text.split()) / 100) if text.split() else 0  # Prefer moderate length
            
            # Combine factors
            combined_score = (
                overlap_ratio * 0.4 +
                density * 0.3 +
                position_bonus * 0.2 +
                length_score * 0.1
            )
            
            scored_candidates.append((combined_score, candidate))
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Create results
        results = []
        
        for i, (score, candidate) in enumerate(scored_candidates[:top_k]):
            rank = i + 1
            normalized_score = max(0.0, 1.0 - (rank - 1) * 0.1)
            
            # Generate simple reasoning
            reasoning = self._generate_reasoning(query, candidate["text"], rank, score)
            
            result = LLMRerankResult(
                chunk_id=candidate["chunk_id"],
                document_id=candidate["document_id"],
                llm_rank=rank,
                llm_score=normalized_score,
                reasoning=reasoning,
                original_score=candidate.get("monot5_score", 0.0),
                confidence=min(0.9, score + 0.3),  # Reasonable confidence
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            
            results.append(result)
        
        logger.info(f"Simple LLM reranking completed: {len(candidates)} → {len(results)}")
        return results
    
    def _generate_reasoning(self, query: str, text: str, rank: int, score: float) -> str:
        """Generate simple reasoning for ranking"""
        
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        overlap = query_terms.intersection(text_terms)
        
        if rank == 1:
            return f"Top ranked due to strong relevance (score: {score:.2f}). Contains key terms: {', '.join(list(overlap)[:3])}"
        elif rank <= 3:
            return f"High relevance (score: {score:.2f}). Good match with query terms and context."
        else:
            return f"Moderate relevance (score: {score:.2f}). Some alignment with query but less comprehensive."