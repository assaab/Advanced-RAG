"""
Integrated Validator Module
Combines answer generation with HallBayes hallucination detection
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from llm_generation.generation.answer_generator import (
    AnswerGenerator,
    GenerationConfig,
    GenerationResult,
    LLMBackend
)
from observability.evaluation.answer_validator import (
    AnswerValidator,
    ValidationResult
)
from observability.evaluation.hallucination_detector import HallucinationDetector
from observability.evaluation.config import HallBayesConfig, RiskLevel
from llm_generation.prompt.prompt_templates import PromptTemplate, PromptType
from retrieval_pipeline.pipeline import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ValidatedAnswerConfig:
    """
    Configuration for validated answer generation
    
    Attributes:
        generation_config: Configuration for answer generation
        validation_config: Configuration for hallucination detection
        auto_regenerate_on_high_risk: Regenerate if high hallucination risk
        max_regeneration_attempts: Maximum regeneration attempts
        risk_threshold: Risk score threshold for regeneration
        require_validation: Whether validation is required
        async_validation: Perform validation asynchronously (non-blocking)
    """
    generation_config: Optional[GenerationConfig] = None
    validation_config: Optional[HallBayesConfig] = None
    auto_regenerate_on_high_risk: bool = True
    max_regeneration_attempts: int = 2
    risk_threshold: float = 0.7
    require_validation: bool = True
    async_validation: bool = False
    
    def __post_init__(self):
        """Initialize default configs if not provided"""
        if self.generation_config is None:
            self.generation_config = GenerationConfig()
        if self.validation_config is None:
            self.validation_config = HallBayesConfig()


@dataclass
class ValidatedAnswer:
    """
    Complete validated answer with generation and validation results
    
    Attributes:
        answer: Final answer text
        query: Original query
        generation_result: Result from answer generation
        validation_result: Result from hallucination detection
        is_valid: Whether answer passed validation
        should_serve: Whether answer should be served to user
        regeneration_attempts: Number of regeneration attempts
        total_time_ms: Total time including generation and validation
        metadata: Additional metadata
    """
    answer: str
    query: str
    generation_result: GenerationResult
    validation_result: ValidationResult
    is_valid: bool
    should_serve: bool
    regeneration_attempts: int = 0
    total_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "answer": self.answer,
            "query": self.query,
            "is_valid": self.is_valid,
            "should_serve": self.should_serve,
            "generation_result": self.generation_result.to_dict(),
            "validation_result": self.validation_result.to_dict(),
            "regeneration_attempts": self.regeneration_attempts,
            "total_time_ms": self.total_time_ms,
            "metadata": self.metadata
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get concise summary of results"""
        return {
            "answer": self.answer,
            "query": self.query,
            "is_valid": self.is_valid,
            "should_serve": self.should_serve,
            "risk_level": self.validation_result.hallucination_result.risk_level.value,
            "risk_score": self.validation_result.hallucination_result.risk_score,
            "confidence": self.validation_result.confidence,
            "quality_score": self.validation_result.quality_score,
            "total_time_ms": self.total_time_ms,
            "regeneration_attempts": self.regeneration_attempts
        }
    
    def get_user_response(self) -> Dict[str, Any]:
        """Get user-facing response (excludes internal details)"""
        response = {
            "answer": self.answer,
            "query": self.query,
            "confidence": self.validation_result.confidence,
            "sources": self.validation_result.sources,
            "metadata": {
                "generation_time_ms": self.generation_result.generation_time_ms,
                "total_time_ms": self.total_time_ms
            }
        }
        
        # Add warnings if present
        if self.validation_result.warnings:
            response["warnings"] = self.validation_result.warnings
        
        # Add risk information for transparency
        if not self.should_serve or self.validation_result.hallucination_result.risk_level in [
            RiskLevel.HIGH, RiskLevel.CRITICAL
        ]:
            response["quality_note"] = {
                "risk_level": self.validation_result.hallucination_result.risk_level.value,
                "message": self._get_risk_message()
            }
        
        return response
    
    def _get_risk_message(self) -> str:
        """Get user-friendly risk message"""
        risk_level = self.validation_result.hallucination_result.risk_level
        
        messages = {
            RiskLevel.LOW: "This answer has high confidence and low hallucination risk.",
            RiskLevel.MEDIUM: "This answer has moderate confidence. Please verify important details.",
            RiskLevel.HIGH: "This answer has elevated uncertainty. Manual verification recommended.",
            RiskLevel.CRITICAL: "This answer has critical quality issues and should not be trusted."
        }
        
        return messages.get(risk_level, "Quality assessment unavailable.")


class IntegratedValidator:
    """
    Integrated validator combining generation and validation
    
    Features:
    - Seamless integration of answer generation and hallucination detection
    - Automatic regeneration for high-risk answers
    - Configurable risk thresholds
    - Async and sync validation modes
    - Comprehensive performance tracking
    - Graceful error handling
    """
    
    def __init__(
        self,
        generator: Optional[AnswerGenerator] = None,
        validator: Optional[AnswerValidator] = None,
        detector: Optional[HallucinationDetector] = None,
        config: Optional[ValidatedAnswerConfig] = None
    ):
        """
        Initialize the integrated validator
        
        Args:
            generator: AnswerGenerator instance (creates new if None)
            validator: AnswerValidator instance (creates new if None)
            detector: HallucinationDetector instance (creates new if None)
            config: Configuration (uses default if None)
        """
        self.config = config or ValidatedAnswerConfig()
        
        # Initialize components
        self.generator = generator or AnswerGenerator(self.config.generation_config)
        self.detector = detector or HallucinationDetector(self.config.validation_config)
        self.validator = validator or AnswerValidator(self.detector, self.config.validation_config)
        
        # Statistics
        self._stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "regenerations": 0,
            "average_total_time_ms": 0.0
        }
        
        logger.info("IntegratedValidator initialized")
    
    async def generate_and_validate(
        self,
        retrieval_result: RetrievalResult,
        prompt_template: Optional[PromptTemplate] = None,
        custom_config: Optional[ValidatedAnswerConfig] = None,
        **kwargs
    ) -> ValidatedAnswer:
        """
        Generate and validate answer in one operation
        
        Args:
            retrieval_result: RetrievalResult with query and context
            prompt_template: Optional custom prompt template
            custom_config: Optional custom configuration
            **kwargs: Additional parameters
            
        Returns:
            ValidatedAnswer with complete results
        """
        import time
        start_time = time.time()
        
        config = custom_config or self.config
        regeneration_attempts = 0
        
        # Generate initial answer
        generation_result = await self.generator.generate(
            retrieval_result=retrieval_result,
            prompt_template=prompt_template,
            custom_config=config.generation_config,
            **kwargs
        )
        
        # Validate answer
        if config.require_validation:
            validation_result = await self.validator.validate(
                answer=generation_result.answer,
                retrieval_result=retrieval_result
            )
            
            # Check if regeneration needed
            if config.auto_regenerate_on_high_risk:
                while (
                    validation_result.hallucination_result.risk_score > config.risk_threshold
                    and regeneration_attempts < config.max_regeneration_attempts
                ):
                    regeneration_attempts += 1
                    logger.warning(
                        f"High hallucination risk detected (score: "
                        f"{validation_result.hallucination_result.risk_score:.2f}). "
                        f"Regenerating (attempt {regeneration_attempts})..."
                    )
                    
                    # Regenerate with adjusted parameters
                    adjusted_config = self._adjust_generation_config(
                        config.generation_config,
                        regeneration_attempts
                    )
                    
                    generation_result = await self.generator.generate(
                        retrieval_result=retrieval_result,
                        prompt_template=prompt_template,
                        custom_config=adjusted_config,
                        **kwargs
                    )
                    
                    # Re-validate
                    validation_result = await self.validator.validate(
                        answer=generation_result.answer,
                        retrieval_result=retrieval_result
                    )
        else:
            # Skip validation (create minimal validation result)
            validation_result = self._create_minimal_validation(
                generation_result.answer,
                retrieval_result
            )
        
        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000
        
        # Build validated answer
        validated_answer = ValidatedAnswer(
            answer=generation_result.answer,
            query=retrieval_result.query,
            generation_result=generation_result,
            validation_result=validation_result,
            is_valid=validation_result.is_valid,
            should_serve=validation_result.should_serve(),
            regeneration_attempts=regeneration_attempts,
            total_time_ms=total_time_ms,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "backend": config.generation_config.backend.value,
                "model": config.generation_config.model_name,
                "validation_enabled": config.require_validation
            }
        )
        
        # Update statistics
        self._update_stats(validated_answer)
        
        # Log result
        self._log_validation(validated_answer)
        
        return validated_answer
    
    def _adjust_generation_config(
        self,
        config: GenerationConfig,
        attempt: int
    ) -> GenerationConfig:
        """
        Adjust generation config for regeneration attempts
        
        Args:
            config: Original configuration
            attempt: Regeneration attempt number
            
        Returns:
            Adjusted GenerationConfig
        """
        # Create copy of config
        adjusted = GenerationConfig(
            backend=config.backend,
            model_name=config.model_name,
            max_tokens=config.max_tokens,
            temperature=max(0.1, config.temperature - 0.1 * attempt),  # Lower temperature
            top_p=max(0.7, config.top_p - 0.05 * attempt),  # Lower top_p
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            timeout_seconds=config.timeout_seconds,
            retry_attempts=config.retry_attempts,
            stream=config.stream
        )
        
        logger.debug(
            f"Adjusted generation config for attempt {attempt}: "
            f"temperature={adjusted.temperature}, top_p={adjusted.top_p}"
        )
        
        return adjusted
    
    def _create_minimal_validation(
        self,
        answer: str,
        retrieval_result: RetrievalResult
    ) -> ValidationResult:
        """
        Create minimal validation result when validation is skipped
        
        Args:
            answer: Generated answer
            retrieval_result: Retrieval result
            
        Returns:
            Minimal ValidationResult
        """
        from observability.evaluation.hallucination_detector import HallucinationResult
        
        # Create basic hallucination result
        hallucination_result = HallucinationResult(
            is_hallucination=False,
            risk_score=0.5,  # Neutral score
            risk_level=RiskLevel.MEDIUM,
            confidence=0.5,
            metadata={"validation_skipped": True}
        )
        
        return ValidationResult(
            answer=answer,
            query=retrieval_result.query,
            is_valid=True,
            hallucination_result=hallucination_result,
            confidence=0.5,
            quality_score=0.5,
            sources=retrieval_result.answer_sources,
            warnings=["Validation was skipped"],
            metadata={"validation_skipped": True}
        )
    
    async def batch_generate_and_validate(
        self,
        retrieval_results: List[RetrievalResult],
        max_concurrent: int = 5,
        **kwargs
    ) -> List[ValidatedAnswer]:
        """
        Batch generate and validate multiple queries
        
        Args:
            retrieval_results: List of RetrievalResult
            max_concurrent: Maximum concurrent operations
            **kwargs: Additional parameters
            
        Returns:
            List of ValidatedAnswer
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(result: RetrievalResult) -> ValidatedAnswer:
            async with semaphore:
                return await self.generate_and_validate(result, **kwargs)
        
        tasks = [process_with_semaphore(result) for result in retrieval_results]
        validated_answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(validated_answers):
            if isinstance(result, Exception):
                logger.error(f"Batch validation {i} failed: {result}")
                # Create error result
                error_result = self._create_error_result(
                    retrieval_results[i],
                    str(result)
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    def _create_error_result(
        self,
        retrieval_result: RetrievalResult,
        error: str
    ) -> ValidatedAnswer:
        """
        Create error result when generation/validation fails
        
        Args:
            retrieval_result: Original retrieval result
            error: Error message
            
        Returns:
            ValidatedAnswer indicating error
        """
        from observability.evaluation.hallucination_detector import HallucinationResult
        
        # Create error generation result
        generation_result = GenerationResult(
            answer=f"Error generating answer: {error}",
            query=retrieval_result.query,
            model="error",
            metadata={"error": error}
        )
        
        # Create error hallucination result
        hallucination_result = HallucinationResult(
            is_hallucination=True,
            risk_score=1.0,
            risk_level=RiskLevel.CRITICAL,
            confidence=0.0,
            metadata={"error": error}
        )
        
        # Create error validation result
        validation_result = ValidationResult(
            answer=generation_result.answer,
            query=retrieval_result.query,
            is_valid=False,
            hallucination_result=hallucination_result,
            confidence=0.0,
            quality_score=0.0,
            warnings=[f"Generation failed: {error}"],
            metadata={"error": error}
        )
        
        return ValidatedAnswer(
            answer=generation_result.answer,
            query=retrieval_result.query,
            generation_result=generation_result,
            validation_result=validation_result,
            is_valid=False,
            should_serve=False,
            metadata={"error": error}
        )
    
    def _update_stats(self, validated_answer: ValidatedAnswer) -> None:
        """
        Update statistics
        
        Args:
            validated_answer: Validated answer result
        """
        self._stats["total_validations"] += 1
        
        if validated_answer.is_valid:
            self._stats["passed_validations"] += 1
        else:
            self._stats["failed_validations"] += 1
        
        self._stats["regenerations"] += validated_answer.regeneration_attempts
        
        # Update running average time
        n = self._stats["total_validations"]
        old_avg = self._stats["average_total_time_ms"]
        self._stats["average_total_time_ms"] = (
            old_avg + (validated_answer.total_time_ms - old_avg) / n
        )
    
    def _log_validation(self, validated_answer: ValidatedAnswer) -> None:
        """
        Log validation result
        
        Args:
            validated_answer: Validated answer to log
        """
        summary = validated_answer.get_summary()
        
        log_level = logging.INFO if validated_answer.should_serve else logging.WARNING
        
        logger.log(
            log_level,
            f"Validated answer: query='{summary['query'][:50]}...' "
            f"valid={summary['is_valid']} risk={summary['risk_level']} "
            f"score={summary['quality_score']:.2f} time={summary['total_time_ms']:.0f}ms "
            f"regenerations={summary['regeneration_attempts']}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get validator statistics
        
        Returns:
            Dictionary of statistics
        """
        pass_rate = 0.0
        if self._stats["total_validations"] > 0:
            pass_rate = self._stats["passed_validations"] / self._stats["total_validations"]
        
        return {
            **self._stats,
            "pass_rate": pass_rate,
            "generator_stats": self.generator.get_stats(),
            "validator_stats": self.validator.get_stats(),
            "config": {
                "auto_regenerate": self.config.auto_regenerate_on_high_risk,
                "max_regeneration_attempts": self.config.max_regeneration_attempts,
                "risk_threshold": self.config.risk_threshold
            }
        }
    
    async def generate_only(
        self,
        retrieval_result: RetrievalResult,
        **kwargs
    ) -> GenerationResult:
        """
        Generate answer without validation (utility method)
        
        Args:
            retrieval_result: RetrievalResult with query and context
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult
        """
        return await self.generator.generate(retrieval_result, **kwargs)
    
    async def validate_existing(
        self,
        answer: str,
        retrieval_result: RetrievalResult
    ) -> ValidationResult:
        """
        Validate an existing answer (utility method)
        
        Args:
            answer: Answer to validate
            retrieval_result: Original retrieval result
            
        Returns:
            ValidationResult
        """
        return await self.validator.validate(answer, retrieval_result)

