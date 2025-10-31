"""
Answer Validator Module
Validates generated answers against retrieved context using HallBayes
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from observability.evaluation.hallucination_detector import (
    HallucinationDetector,
    HallucinationResult
)
from observability.evaluation.config import HallBayesConfig, RiskLevel
from retrieval_pipeline.context.parent_retriever import RepackedContext
from retrieval_pipeline.pipeline import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Complete validation result for an answer
    
    Attributes:
        answer: The validated answer
        query: Original query
        is_valid: Whether answer passes validation
        hallucination_result: Detailed hallucination detection result
        confidence: Overall confidence in the answer
        quality_score: Quality score (0.0 to 1.0)
        sources: List of source attributions
        warnings: List of validation warnings
        recommendations: List of recommendations for improvement
        metadata: Additional metadata
    """
    answer: str
    query: str
    is_valid: bool
    hallucination_result: HallucinationResult
    confidence: float
    quality_score: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "answer": self.answer,
            "query": self.query,
            "is_valid": self.is_valid,
            "hallucination_result": self.hallucination_result.to_dict(),
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "sources": self.sources,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }
    
    def should_serve(self) -> bool:
        """
        Determine if answer should be served to user
        
        Returns:
            True if answer is safe to serve
        """
        # Don't serve if marked invalid
        if not self.is_valid:
            return False
        
        # Don't serve critical risk answers
        if self.hallucination_result.risk_level == RiskLevel.CRITICAL:
            return False
        
        # Don't serve if confidence is too low
        if self.confidence < 0.3:
            return False
        
        return True
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of risk factors"""
        return {
            "risk_level": self.hallucination_result.risk_level.value,
            "risk_score": self.hallucination_result.risk_score,
            "is_hallucination": self.hallucination_result.is_hallucination,
            "should_serve": self.should_serve(),
            "confidence": self.confidence,
            "quality_score": self.quality_score
        }


class AnswerValidator:
    """
    Validates generated answers against retrieved context
    
    Features:
    - Integrates with RetrievalResult for context extraction
    - Performs hallucination detection using HallBayes
    - Provides quality scoring and recommendations
    - Supports batch validation
    - Configurable validation thresholds
    """
    
    def __init__(
        self,
        detector: Optional[HallucinationDetector] = None,
        config: Optional[HallBayesConfig] = None,
        min_quality_score: float = 0.5,
        min_confidence: float = 0.5
    ):
        """
        Initialize the answer validator
        
        Args:
            detector: HallucinationDetector instance (creates new if None)
            config: Configuration (uses default if None)
            min_quality_score: Minimum quality score for valid answers
            min_confidence: Minimum confidence for valid answers
        """
        self.config = config or HallBayesConfig()
        self.detector = detector or HallucinationDetector(self.config)
        self.min_quality_score = min_quality_score
        self.min_confidence = min_confidence
        
        logger.info("AnswerValidator initialized")
    
    async def validate(
        self,
        answer: str,
        retrieval_result: RetrievalResult,
        additional_context: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate an answer against retrieval result
        
        Args:
            answer: The generated answer to validate
            retrieval_result: RetrievalResult containing context
            additional_context: Optional additional context
            
        Returns:
            ValidationResult with validation details
        """
        try:
            # Extract context from retrieval result
            context = self._extract_context(retrieval_result, additional_context)
            
            # Perform hallucination detection
            hallucination_result = await self.detector.detect(
                answer=answer,
                context=context,
                query=retrieval_result.query
            )
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(
                answer,
                retrieval_result,
                hallucination_result
            )
            
            # Determine overall confidence
            confidence = self._calculate_confidence(
                hallucination_result,
                retrieval_result,
                quality_score
            )
            
            # Check validity
            is_valid = self._check_validity(
                hallucination_result,
                quality_score,
                confidence
            )
            
            # Generate warnings and recommendations
            warnings = self._generate_warnings(
                hallucination_result,
                quality_score,
                confidence,
                retrieval_result
            )
            
            recommendations = self._generate_recommendations(
                hallucination_result,
                quality_score,
                retrieval_result
            )
            
            # Extract source information
            sources = self._extract_sources(retrieval_result)
            
            # Build metadata
            metadata = {
                "validation_timestamp": datetime.utcnow().isoformat(),
                "retrieval_time_ms": retrieval_result.total_time_ms,
                "context_length": len(context),
                "answer_length": len(answer),
                "num_sources": len(sources),
                "total_candidates": retrieval_result.total_candidates,
                "final_results": retrieval_result.final_results
            }
            
            validation_result = ValidationResult(
                answer=answer,
                query=retrieval_result.query,
                is_valid=is_valid,
                hallucination_result=hallucination_result,
                confidence=confidence,
                quality_score=quality_score,
                sources=sources,
                warnings=warnings,
                recommendations=recommendations,
                metadata=metadata
            )
            
            self._log_validation(validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            
            # Return failed validation result
            return self._create_failed_validation(
                answer,
                retrieval_result.query,
                str(e)
            )
    
    def _extract_context(
        self,
        retrieval_result: RetrievalResult,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Extract context text from retrieval result
        
        Args:
            retrieval_result: RetrievalResult instance
            additional_context: Optional additional context
            
        Returns:
            Combined context string
        """
        # Primary context from repacked context
        context_parts = []
        
        if retrieval_result.final_context and retrieval_result.final_context.text:
            context_parts.append(retrieval_result.final_context.text)
        
        # Add additional context if provided
        if additional_context:
            context_parts.append(additional_context)
        
        # Combine context parts
        combined_context = "\n\n".join(context_parts)
        
        # Truncate if too long (to stay within limits)
        max_context_length = self.config.max_context_length
        if len(combined_context) > max_context_length:
            logger.warning(
                f"Context truncated from {len(combined_context)} to {max_context_length} chars"
            )
            combined_context = combined_context[:max_context_length]
        
        return combined_context
    
    def _calculate_quality_score(
        self,
        answer: str,
        retrieval_result: RetrievalResult,
        hallucination_result: HallucinationResult
    ) -> float:
        """
        Calculate overall quality score for the answer
        
        Args:
            answer: Generated answer
            retrieval_result: Retrieval result
            hallucination_result: Hallucination detection result
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        scores = []
        
        # Factor 1: Hallucination risk (inverse)
        hallucination_score = 1.0 - hallucination_result.risk_score
        scores.append((hallucination_score, 0.4))  # 40% weight
        
        # Factor 2: Confidence from hallucination detection
        confidence_score = hallucination_result.confidence
        scores.append((confidence_score, 0.2))  # 20% weight
        
        # Factor 3: Source quality (based on relevance scores)
        source_score = self._calculate_source_quality(retrieval_result)
        scores.append((source_score, 0.2))  # 20% weight
        
        # Factor 4: Answer completeness
        completeness_score = self._calculate_completeness(answer)
        scores.append((completeness_score, 0.2))  # 20% weight
        
        # Weighted average
        total_score = sum(score * weight for score, weight in scores)
        
        return max(0.0, min(1.0, total_score))
    
    def _calculate_source_quality(self, retrieval_result: RetrievalResult) -> float:
        """
        Calculate quality of sources
        
        Args:
            retrieval_result: Retrieval result
            
        Returns:
            Source quality score
        """
        if not retrieval_result.final_context or not retrieval_result.final_context.relevance_scores:
            return 0.5  # Default middle score
        
        # Average of relevance scores
        relevance_scores = retrieval_result.final_context.relevance_scores
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            # Normalize to 0-1 range (assuming scores are typically 0-1)
            return max(0.0, min(1.0, avg_relevance))
        
        return 0.5
    
    def _calculate_completeness(self, answer: str) -> float:
        """
        Calculate answer completeness score
        
        Args:
            answer: Generated answer
            
        Returns:
            Completeness score
        """
        # Simple heuristic based on answer length and structure
        answer_length = len(answer)
        
        # Too short answers are likely incomplete
        if answer_length < 50:
            return 0.3
        
        # Check for common completeness indicators
        has_multiple_sentences = answer.count('.') >= 2
        has_reasonable_length = 100 <= answer_length <= 2000
        
        score = 0.5
        if has_multiple_sentences:
            score += 0.25
        if has_reasonable_length:
            score += 0.25
        
        return min(1.0, score)
    
    def _calculate_confidence(
        self,
        hallucination_result: HallucinationResult,
        retrieval_result: RetrievalResult,
        quality_score: float
    ) -> float:
        """
        Calculate overall confidence in the answer
        
        Args:
            hallucination_result: Hallucination detection result
            retrieval_result: Retrieval result
            quality_score: Calculated quality score
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        factors = []
        
        # Hallucination detection confidence
        factors.append((hallucination_result.confidence, 0.3))
        
        # Quality score
        factors.append((quality_score, 0.3))
        
        # Number of sources (more sources = higher confidence)
        num_sources = len(retrieval_result.answer_sources) if retrieval_result.answer_sources else 0
        source_confidence = min(num_sources / 3.0, 1.0)  # Max at 3+ sources
        factors.append((source_confidence, 0.2))
        
        # Retrieval pipeline success (did we find good results?)
        retrieval_success = min(retrieval_result.final_results / 5.0, 1.0)  # Max at 5+ results
        factors.append((retrieval_success, 0.2))
        
        # Weighted average
        total_confidence = sum(factor * weight for factor, weight in factors)
        
        return max(0.0, min(1.0, total_confidence))
    
    def _check_validity(
        self,
        hallucination_result: HallucinationResult,
        quality_score: float,
        confidence: float
    ) -> bool:
        """
        Determine if answer is valid
        
        Args:
            hallucination_result: Hallucination detection result
            quality_score: Quality score
            confidence: Confidence score
            
        Returns:
            True if answer is valid
        """
        # Fail if critical risk
        if hallucination_result.risk_level == RiskLevel.CRITICAL:
            return False
        
        # Fail if quality too low
        if quality_score < self.min_quality_score:
            return False
        
        # Fail if confidence too low
        if confidence < self.min_confidence:
            return False
        
        # Warn but pass for high risk
        if hallucination_result.risk_level == RiskLevel.HIGH:
            logger.warning("Answer has high hallucination risk but passes minimum thresholds")
            return True
        
        return True
    
    def _generate_warnings(
        self,
        hallucination_result: HallucinationResult,
        quality_score: float,
        confidence: float,
        retrieval_result: RetrievalResult
    ) -> List[str]:
        """
        Generate validation warnings
        
        Args:
            hallucination_result: Hallucination detection result
            quality_score: Quality score
            confidence: Confidence score
            retrieval_result: Retrieval result
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Hallucination warnings
        if hallucination_result.risk_level == RiskLevel.HIGH:
            warnings.append("High hallucination risk detected")
        elif hallucination_result.risk_level == RiskLevel.CRITICAL:
            warnings.append("Critical hallucination risk - answer should not be served")
        
        if hallucination_result.is_hallucination:
            warnings.append("Potential hallucination detected")
        
        # Quality warnings
        if quality_score < 0.6:
            warnings.append(f"Low quality score: {quality_score:.2f}")
        
        # Confidence warnings
        if confidence < 0.6:
            warnings.append(f"Low confidence: {confidence:.2f}")
        
        # Source warnings
        if not retrieval_result.answer_sources:
            warnings.append("No sources found for answer")
        elif len(retrieval_result.answer_sources) < 2:
            warnings.append("Limited sources available")
        
        # Retrieval warnings
        if retrieval_result.final_results == 0:
            warnings.append("No results from retrieval pipeline")
        
        return warnings
    
    def _generate_recommendations(
        self,
        hallucination_result: HallucinationResult,
        quality_score: float,
        retrieval_result: RetrievalResult
    ) -> List[str]:
        """
        Generate recommendations for improvement
        
        Args:
            hallucination_result: Hallucination detection result
            quality_score: Quality score
            retrieval_result: Retrieval result
            
        Returns:
            List of recommendation messages
        """
        recommendations = []
        
        # Add suggestions from hallucination detection
        if hallucination_result.suggestions:
            recommendations.extend(hallucination_result.suggestions)
        
        # Quality-based recommendations
        if quality_score < 0.7:
            recommendations.append("Consider regenerating answer with different prompt")
        
        # Source-based recommendations
        if not retrieval_result.answer_sources:
            recommendations.append("Retrieve more relevant sources before generating answer")
        
        # Context-based recommendations
        if retrieval_result.final_context.total_tokens < 100:
            recommendations.append("Insufficient context - expand search parameters")
        
        return recommendations
    
    def _extract_sources(self, retrieval_result: RetrievalResult) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieval result
        
        Args:
            retrieval_result: Retrieval result
            
        Returns:
            List of source dictionaries
        """
        if retrieval_result.answer_sources:
            return retrieval_result.answer_sources
        
        # Fallback: extract from context
        sources = []
        if retrieval_result.final_context and retrieval_result.final_context.sources:
            for i, source in enumerate(retrieval_result.final_context.sources):
                sources.append({
                    "source_id": i + 1,
                    "document_id": source.get("document_id", "unknown"),
                    "section_title": source.get("section_title", f"Section {i+1}"),
                    "relevance_score": source.get("relevance_score", 0.0),
                    "token_count": source.get("token_count", 0)
                })
        
        return sources
    
    def _create_failed_validation(
        self,
        answer: str,
        query: str,
        error: str
    ) -> ValidationResult:
        """
        Create a failed validation result
        
        Args:
            answer: Answer that failed validation
            query: Original query
            error: Error message
            
        Returns:
            ValidationResult indicating failure
        """
        # Create minimal hallucination result
        hallucination_result = HallucinationResult(
            is_hallucination=True,
            risk_score=1.0,
            risk_level=RiskLevel.CRITICAL,
            confidence=0.0,
            metadata={"error": error, "validation_failed": True}
        )
        
        return ValidationResult(
            answer=answer,
            query=query,
            is_valid=False,
            hallucination_result=hallucination_result,
            confidence=0.0,
            quality_score=0.0,
            warnings=[f"Validation failed: {error}"],
            recommendations=["Manual review required"],
            metadata={"validation_error": error}
        )
    
    def _log_validation(self, result: ValidationResult) -> None:
        """
        Log validation result
        
        Args:
            result: Validation result to log
        """
        if not self.config.log_validations:
            return
        
        log_data = {
            "query": result.query[:100],
            "is_valid": result.is_valid,
            "risk_level": result.hallucination_result.risk_level.value,
            "risk_score": result.hallucination_result.risk_score,
            "confidence": result.confidence,
            "quality_score": result.quality_score,
            "num_warnings": len(result.warnings),
            "should_serve": result.should_serve()
        }
        
        if result.is_valid:
            logger.info(f"Validation passed: {log_data}")
        else:
            logger.warning(f"Validation failed: {log_data}")
    
    async def batch_validate(
        self,
        validations: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None
    ) -> List[ValidationResult]:
        """
        Perform batch validation
        
        Args:
            validations: List of dicts with 'answer' and 'retrieval_result' keys
            max_concurrent: Maximum concurrent validations
            
        Returns:
            List of ValidationResult
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_validations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_with_semaphore(validation: Dict[str, Any]) -> ValidationResult:
            async with semaphore:
                return await self.validate(
                    answer=validation["answer"],
                    retrieval_result=validation["retrieval_result"],
                    additional_context=validation.get("additional_context")
                )
        
        tasks = [validate_with_semaphore(v) for v in validations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validation {i} failed: {result}")
                final_results.append(
                    self._create_failed_validation(
                        validations[i]["answer"],
                        validations[i]["retrieval_result"].query,
                        str(result)
                    )
                )
            else:
                final_results.append(result)
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get validator statistics
        
        Returns:
            Dictionary of statistics
        """
        detector_stats = self.detector.get_stats()
        
        return {
            "detector_stats": detector_stats,
            "config": {
                "min_quality_score": self.min_quality_score,
                "min_confidence": self.min_confidence
            }
        }

