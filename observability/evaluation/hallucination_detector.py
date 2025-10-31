"""
HallBayes Hallucination Detector Module
Provides async wrapper around HallBayes with caching and error handling
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from observability.evaluation.config import HallBayesConfig, RiskLevel, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """
    Result of hallucination detection
    
    Attributes:
        is_hallucination: Whether hallucination was detected
        risk_score: Risk score between 0.0 and 1.0
        risk_level: Categorized risk level
        confidence: Confidence in the detection
        detected_issues: List of specific issues found
        suggestions: Suggestions for improvement
        sla_certificate: Optional SLA certificate
        processing_time_ms: Time taken for detection
        metadata: Additional metadata
    """
    is_hallucination: bool
    risk_score: float
    risk_level: RiskLevel
    confidence: float
    detected_issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    sla_certificate: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_hallucination": self.is_hallucination,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "detected_issues": self.detected_issues,
            "suggestions": self.suggestions,
            "sla_certificate": self.sla_certificate,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }


@dataclass
class CacheEntry:
    """Cache entry for validation results"""
    result: HallucinationResult
    timestamp: datetime
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        expiry_time = self.timestamp + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time


class HallucinationDetector:
    """
    Async wrapper around HallBayes for hallucination detection
    
    Features:
    - Async/await support for non-blocking operations
    - Result caching for performance
    - Graceful error handling with fallbacks
    - Retry logic for transient failures
    - Comprehensive logging
    """
    
    def __init__(self, config: Optional[HallBayesConfig] = None):
        """
        Initialize the hallucination detector
        
        Args:
            config: Configuration for the detector (uses default if None)
        """
        self.config = config or DEFAULT_CONFIG
        self._cache: Dict[str, CacheEntry] = {}
        self._hallbayes_client = None
        self._initialized = False
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "average_time_ms": 0.0
        }
        
        logger.info(
            f"HallucinationDetector initialized with backend: {self.config.backend.value}"
        )
    
    async def initialize(self) -> None:
        """
        Initialize HallBayes client (lazy initialization)
        
        This is separated from __init__ to allow async initialization
        """
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Import HallBayes here to avoid initialization errors
                # Note: Actual HallBayes API may differ - this is a placeholder
                # based on typical patterns. Update when actual library is available.
                logger.info("Initializing HallBayes client...")
                
                # Placeholder for actual HallBayes initialization
                # When the actual library is available, replace this with:
                # from hallbayes import HallBayesClient
                # self._hallbayes_client = HallBayesClient(
                #     backend=self.config.backend.value,
                #     api_key=self.config.api_key,
                #     model_name=self.config.model_name
                # )
                
                # For now, we'll create a mock client for testing
                self._hallbayes_client = self._create_mock_client()
                
                self._initialized = True
                logger.info("HallBayes client initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize HallBayes client: {e}")
                if not self.config.fail_gracefully:
                    raise
                # Continue with mock client for graceful degradation
                self._hallbayes_client = self._create_mock_client()
                self._initialized = True
    
    def _create_mock_client(self) -> Any:
        """
        Create a mock client for testing/development
        
        This will be replaced with actual HallBayes client when available
        """
        class MockHallBayesClient:
            """Mock client for development and testing"""
            
            async def check_hallucination(
                self, 
                answer: str, 
                context: str,
                **kwargs
            ) -> Dict[str, Any]:
                """Mock hallucination check"""
                # Simple heuristic: longer answers without context are riskier
                answer_length = len(answer)
                context_length = len(context)
                
                if context_length == 0:
                    risk_score = 0.8
                else:
                    # Simple ratio-based risk
                    ratio = answer_length / max(context_length, 1)
                    risk_score = min(ratio * 0.3, 0.9)
                
                return {
                    "hallucination_detected": risk_score > 0.5,
                    "risk_score": risk_score,
                    "confidence": 0.75,
                    "issues": [],
                    "suggestions": []
                }
        
        return MockHallBayesClient()
    
    async def detect(
        self,
        answer: str,
        context: str,
        query: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> HallucinationResult:
        """
        Detect hallucinations in an answer given the context
        
        Args:
            answer: The generated answer to validate
            context: The source context used to generate the answer
            query: Optional original query
            use_cache: Whether to use cached results
            **kwargs: Additional parameters for HallBayes
            
        Returns:
            HallucinationResult with detection results
            
        Raises:
            Exception: If validation fails and fail_gracefully is False
        """
        start_time = time.time()
        
        # Ensure client is initialized
        await self.initialize()
        
        # Check cache if enabled
        if use_cache and self.config.enable_caching:
            cache_key = self._generate_cache_key(answer, context, query)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for validation (key: {cache_key[:16]}...)")
                return cached_result
            
            self._stats["cache_misses"] += 1
        
        # Perform validation with retry logic
        result = await self._validate_with_retry(answer, context, query, **kwargs)
        
        # Update processing time
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Cache result if enabled
        if use_cache and self.config.enable_caching:
            self._cache_result(cache_key, result)
        
        # Update statistics
        self._update_stats(result.processing_time_ms)
        
        return result
    
    async def _validate_with_retry(
        self,
        answer: str,
        context: str,
        query: Optional[str],
        **kwargs
    ) -> HallucinationResult:
        """
        Perform validation with retry logic
        
        Args:
            answer: Answer to validate
            context: Source context
            query: Optional query
            **kwargs: Additional parameters
            
        Returns:
            HallucinationResult
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self._perform_validation(answer, context, query, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Validation succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                self._stats["errors"] += 1
                
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"Validation attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {self.config.retry_delay_seconds}s..."
                    )
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    logger.error(f"Validation failed after {attempt + 1} attempts: {e}")
        
        # All retries failed
        if self.config.fail_gracefully:
            logger.warning("Using default risk score due to validation failure")
            return self._create_default_result(last_exception)
        else:
            raise last_exception
    
    async def _perform_validation(
        self,
        answer: str,
        context: str,
        query: Optional[str],
        **kwargs
    ) -> HallucinationResult:
        """
        Perform actual validation using HallBayes
        
        Args:
            answer: Answer to validate
            context: Source context
            query: Optional query
            **kwargs: Additional parameters
            
        Returns:
            HallucinationResult
        """
        # Check if validation should be performed
        if not self.config.should_validate(len(context)):
            logger.warning(
                f"Context length {len(context)} outside valid range "
                f"[{self.config.min_context_length}, {self.config.max_context_length}]"
            )
            return self._create_default_result(
                ValueError("Context length outside valid range")
            )
        
        # Run validation with timeout
        try:
            validation_task = self._hallbayes_client.check_hallucination(
                answer=answer,
                context=context,
                query=query,
                **kwargs
            )
            
            raw_result = await asyncio.wait_for(
                validation_task,
                timeout=self.config.timeout_seconds
            )
            
            # Parse and structure the result
            return self._parse_result(raw_result, answer, context, query)
            
        except asyncio.TimeoutError:
            logger.error(f"Validation timeout after {self.config.timeout_seconds}s")
            raise TimeoutError("Hallucination detection timed out")
    
    def _parse_result(
        self,
        raw_result: Dict[str, Any],
        answer: str,
        context: str,
        query: Optional[str]
    ) -> HallucinationResult:
        """
        Parse raw HallBayes result into HallucinationResult
        
        Args:
            raw_result: Raw result from HallBayes
            answer: Original answer
            context: Original context
            query: Original query
            
        Returns:
            Structured HallucinationResult
        """
        # Extract core fields
        risk_score = float(raw_result.get("risk_score", self.config.default_risk_score))
        is_hallucination = bool(raw_result.get("hallucination_detected", risk_score > 0.5))
        confidence = float(raw_result.get("confidence", 0.5))
        
        # Determine risk level
        risk_level = self.config.get_risk_level(risk_score)
        
        # Extract detailed issues if available
        detected_issues = []
        if self.config.detailed_analysis and "issues" in raw_result:
            detected_issues = raw_result["issues"]
        
        # Extract suggestions if available
        suggestions = []
        if self.config.include_suggestions and "suggestions" in raw_result:
            suggestions = raw_result["suggestions"]
        
        # Extract SLA certificate if available
        sla_certificate = None
        if self.config.generate_sla_certificates and "sla_certificate" in raw_result:
            sla_certificate = raw_result["sla_certificate"]
        
        # Build metadata
        metadata = {
            "answer_length": len(answer),
            "context_length": len(context),
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "backend": self.config.backend.value,
            "model": self.config.model_name
        }
        
        return HallucinationResult(
            is_hallucination=is_hallucination,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            detected_issues=detected_issues,
            suggestions=suggestions,
            sla_certificate=sla_certificate,
            metadata=metadata
        )
    
    def _create_default_result(self, error: Optional[Exception] = None) -> HallucinationResult:
        """
        Create default result when validation fails
        
        Args:
            error: Optional exception that caused the failure
            
        Returns:
            Default HallucinationResult
        """
        risk_score = self.config.default_risk_score
        
        return HallucinationResult(
            is_hallucination=risk_score > 0.5,
            risk_score=risk_score,
            risk_level=self.config.get_risk_level(risk_score),
            confidence=0.0,
            detected_issues=[],
            suggestions=["Validation failed - manual review recommended"],
            metadata={
                "error": str(error) if error else "Unknown error",
                "fallback": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _generate_cache_key(
        self,
        answer: str,
        context: str,
        query: Optional[str]
    ) -> str:
        """
        Generate cache key for validation
        
        Args:
            answer: Answer text
            context: Context text
            query: Query text
            
        Returns:
            Hash-based cache key
        """
        # Create deterministic key from inputs
        key_data = {
            "answer": answer,
            "context": context[:1000],  # Limit context for key generation
            "query": query or "",
            "backend": self.config.backend.value,
            "model": self.config.model_name or "default"
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[HallucinationResult]:
        """
        Get result from cache if available and not expired
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None
        """
        if cache_key not in self._cache:
            return None
        
        entry = self._cache[cache_key]
        
        if entry.is_expired():
            del self._cache[cache_key]
            return None
        
        return entry.result
    
    def _cache_result(self, cache_key: str, result: HallucinationResult) -> None:
        """
        Cache validation result
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        entry = CacheEntry(
            result=result,
            timestamp=datetime.utcnow(),
            ttl_seconds=self.config.cache_ttl_seconds
        )
        
        self._cache[cache_key] = entry
        
        # Simple cache eviction: remove oldest entries if cache is too large
        max_cache_size = 1000
        if len(self._cache) > max_cache_size:
            # Remove oldest 10%
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp
            )
            for key in sorted_keys[:max_cache_size // 10]:
                del self._cache[key]
    
    def _update_stats(self, processing_time_ms: float) -> None:
        """
        Update statistics
        
        Args:
            processing_time_ms: Processing time for this validation
        """
        self._stats["total_validations"] += 1
        
        # Update running average
        n = self._stats["total_validations"]
        old_avg = self._stats["average_time_ms"]
        self._stats["average_time_ms"] = old_avg + (processing_time_ms - old_avg) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics
        
        Returns:
            Dictionary of statistics
        """
        cache_hit_rate = 0.0
        if self._stats["total_validations"] > 0:
            cache_hit_rate = self._stats["cache_hits"] / self._stats["total_validations"]
        
        return {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "initialized": self._initialized,
            "config": self.config.to_dict()
        }
    
    def clear_cache(self) -> None:
        """Clear the validation cache"""
        self._cache.clear()
        logger.info("Validation cache cleared")
    
    async def batch_detect(
        self,
        validations: List[Dict[str, str]],
        max_concurrent: Optional[int] = None
    ) -> List[HallucinationResult]:
        """
        Perform batch validation with concurrency control
        
        Args:
            validations: List of dicts with 'answer', 'context', 'query' keys
            max_concurrent: Maximum concurrent validations (uses config default if None)
            
        Returns:
            List of HallucinationResult
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_validations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_with_semaphore(validation: Dict[str, str]) -> HallucinationResult:
            async with semaphore:
                return await self.detect(
                    answer=validation["answer"],
                    context=validation["context"],
                    query=validation.get("query")
                )
        
        tasks = [validate_with_semaphore(v) for v in validations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validation {i} failed: {result}")
                final_results.append(self._create_default_result(result))
            else:
                final_results.append(result)
        
        return final_results

