"""
Unit Tests for HallucinationDetector
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from observability.evaluation.hallucination_detector import (
    HallucinationDetector,
    HallucinationResult,
    CacheEntry
)
from observability.evaluation.config import (
    HallBayesConfig,
    RiskLevel,
    LLMBackend
)


@pytest.fixture
def config():
    """Create test configuration"""
    return HallBayesConfig(
        backend=LLMBackend.OPENAI,
        enable_caching=True,
        cache_ttl_seconds=60,
        timeout_seconds=5,
        fail_gracefully=True,
        max_retries=2,
        retry_delay_seconds=0.1  # Short delay for tests
    )


@pytest.fixture
def detector(config):
    """Create detector instance"""
    return HallucinationDetector(config)


class TestHallucinationDetectorInitialization:
    """Test detector initialization"""
    
    def test_initialization_with_config(self, config):
        """Test initialization with custom config"""
        detector = HallucinationDetector(config)
        
        assert detector.config == config
        assert detector._cache == {}
        assert not detector._initialized
        assert detector._stats["total_validations"] == 0
    
    def test_initialization_with_defaults(self):
        """Test initialization with default config"""
        detector = HallucinationDetector()
        
        assert detector.config is not None
        assert detector.config.backend == LLMBackend.OPENAI
        assert detector._cache == {}
    
    @pytest.mark.asyncio
    async def test_lazy_initialization(self, detector):
        """Test lazy initialization"""
        assert not detector._initialized
        
        await detector.initialize()
        
        assert detector._initialized
        assert detector._hallbayes_client is not None
    
    @pytest.mark.asyncio
    async def test_initialization_idempotent(self, detector):
        """Test that initialization can be called multiple times safely"""
        await detector.initialize()
        first_client = detector._hallbayes_client
        
        await detector.initialize()
        second_client = detector._hallbayes_client
        
        assert first_client is second_client


class TestHallucinationDetection:
    """Test hallucination detection"""
    
    @pytest.mark.asyncio
    async def test_basic_detection(self, detector):
        """Test basic hallucination detection"""
        answer = "The sky is blue because of Rayleigh scattering."
        context = "Light scattering in the atmosphere causes the sky to appear blue."
        query = "Why is the sky blue?"
        
        result = await detector.detect(answer, context, query)
        
        assert isinstance(result, HallucinationResult)
        assert isinstance(result.risk_score, float)
        assert 0.0 <= result.risk_score <= 1.0
        assert isinstance(result.risk_level, RiskLevel)
        assert isinstance(result.confidence, float)
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_detection_with_empty_context(self, detector):
        """Test detection with empty context (high risk expected)"""
        answer = "The answer is 42."
        context = ""
        
        result = await detector.detect(answer, context)
        
        assert result.risk_score > 0.5  # Should be high risk
    
    @pytest.mark.asyncio
    async def test_detection_with_long_context(self, detector):
        """Test detection with long context (lower risk expected)"""
        answer = "The answer is based on the provided information."
        context = "This is a very long context. " * 100
        
        result = await detector.detect(answer, context)
        
        assert result.risk_score < 0.7  # Should be lower risk with context
    
    @pytest.mark.asyncio
    async def test_detection_updates_stats(self, detector):
        """Test that detection updates statistics"""
        initial_count = detector._stats["total_validations"]
        
        await detector.detect("answer", "context")
        
        assert detector._stats["total_validations"] == initial_count + 1
        assert detector._stats["average_time_ms"] > 0


class TestCaching:
    """Test caching functionality"""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, detector):
        """Test cache hit on repeated detection"""
        answer = "Test answer"
        context = "Test context"
        
        # First call - cache miss
        result1 = await detector.detect(answer, context, use_cache=True)
        cache_misses_1 = detector._stats["cache_misses"]
        
        # Second call - cache hit
        result2 = await detector.detect(answer, context, use_cache=True)
        cache_hits = detector._stats["cache_hits"]
        
        assert result1.risk_score == result2.risk_score
        assert cache_hits > 0
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, config):
        """Test detection with caching disabled"""
        config.enable_caching = False
        detector = HallucinationDetector(config)
        
        result1 = await detector.detect("answer", "context", use_cache=True)
        result2 = await detector.detect("answer", "context", use_cache=True)
        
        # Should not use cache
        assert detector._stats["cache_hits"] == 0
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, detector):
        """Test cache entry expiration"""
        answer = "Test answer"
        context = "Test context"
        
        # Add entry to cache
        cache_key = detector._generate_cache_key(answer, context, None)
        expired_entry = CacheEntry(
            result=HallucinationResult(
                is_hallucination=False,
                risk_score=0.1,
                risk_level=RiskLevel.LOW,
                confidence=0.9
            ),
            timestamp=datetime.utcnow() - timedelta(hours=2),
            ttl_seconds=60
        )
        detector._cache[cache_key] = expired_entry
        
        # Should be treated as cache miss
        result = await detector.detect(answer, context, use_cache=True)
        
        # Cache entry should be removed
        assert cache_key not in detector._cache or not detector._cache[cache_key].is_expired()
    
    def test_clear_cache(self, detector):
        """Test cache clearing"""
        detector._cache["key1"] = "value1"
        detector._cache["key2"] = "value2"
        
        detector.clear_cache()
        
        assert len(detector._cache) == 0


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_graceful_failure(self, config):
        """Test graceful failure when validation fails"""
        config.fail_gracefully = True
        detector = HallucinationDetector(config)
        
        # Mock client to raise error
        await detector.initialize()
        original_method = detector._hallbayes_client.check_hallucination
        detector._hallbayes_client.check_hallucination = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await detector.detect("answer", "context")
        
        # Should return default result, not raise
        assert result.risk_score == config.default_risk_score
        assert result.confidence == 0.0
        assert "error" in result.metadata
        
        # Restore original method
        detector._hallbayes_client.check_hallucination = original_method
    
    @pytest.mark.asyncio
    async def test_retry_logic(self, config):
        """Test retry logic on failures"""
        config.max_retries = 2
        config.retry_delay_seconds = 0.05
        detector = HallucinationDetector(config)
        
        await detector.initialize()
        
        # Mock to fail twice then succeed
        call_count = 0
        async def mock_check(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return {
                "hallucination_detected": False,
                "risk_score": 0.2,
                "confidence": 0.8,
                "issues": [],
                "suggestions": []
            }
        
        detector._hallbayes_client.check_hallucination = mock_check
        
        result = await detector.detect("answer", "context")
        
        # Should succeed after retries
        assert call_count == 3
        assert result.risk_score == 0.2
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, config):
        """Test timeout handling"""
        config.timeout_seconds = 0.1
        config.fail_gracefully = True
        detector = HallucinationDetector(config)
        
        await detector.initialize()
        
        # Mock to take too long
        async def mock_check(*args, **kwargs):
            await asyncio.sleep(1.0)
            return {"risk_score": 0.1}
        
        detector._hallbayes_client.check_hallucination = mock_check
        
        result = await detector.detect("answer", "context")
        
        # Should return default result due to timeout
        assert result.risk_score == config.default_risk_score


class TestBatchDetection:
    """Test batch detection"""
    
    @pytest.mark.asyncio
    async def test_batch_detect(self, detector):
        """Test batch detection"""
        validations = [
            {"answer": "Answer 1", "context": "Context 1"},
            {"answer": "Answer 2", "context": "Context 2"},
            {"answer": "Answer 3", "context": "Context 3"}
        ]
        
        results = await detector.batch_detect(validations, max_concurrent=2)
        
        assert len(results) == 3
        assert all(isinstance(r, HallucinationResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_detect_with_errors(self, detector):
        """Test batch detection with some failures"""
        await detector.initialize()
        
        # Mock to fail for specific inputs
        original_method = detector._hallbayes_client.check_hallucination
        async def mock_check(answer, context, **kwargs):
            if "fail" in answer:
                raise Exception("Validation error")
            return await original_method(answer, context, **kwargs)
        
        detector._hallbayes_client.check_hallucination = mock_check
        
        validations = [
            {"answer": "Answer 1", "context": "Context 1"},
            {"answer": "fail Answer 2", "context": "Context 2"},
            {"answer": "Answer 3", "context": "Context 3"}
        ]
        
        results = await detector.batch_detect(validations)
        
        # Should return results for all, with defaults for failures
        assert len(results) == 3
        assert results[1].confidence == 0.0  # Failed validation


class TestStatistics:
    """Test statistics tracking"""
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, detector):
        """Test that statistics are tracked correctly"""
        initial_stats = detector.get_stats()
        
        # Perform some detections
        await detector.detect("answer1", "context1")
        await detector.detect("answer2", "context2")
        await detector.detect("answer1", "context1")  # Cache hit
        
        final_stats = detector.get_stats()
        
        assert final_stats["total_validations"] > initial_stats["total_validations"]
        assert final_stats["cache_hits"] > initial_stats["cache_hits"]
        assert final_stats["average_time_ms"] > 0
    
    def test_get_stats_structure(self, detector):
        """Test stats structure"""
        stats = detector.get_stats()
        
        assert "total_validations" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert "cache_size" in stats
        assert "initialized" in stats
        assert "config" in stats


class TestCacheKeyGeneration:
    """Test cache key generation"""
    
    def test_deterministic_keys(self, detector):
        """Test that same inputs produce same keys"""
        answer = "Test answer"
        context = "Test context"
        query = "Test query"
        
        key1 = detector._generate_cache_key(answer, context, query)
        key2 = detector._generate_cache_key(answer, context, query)
        
        assert key1 == key2
    
    def test_different_inputs_different_keys(self, detector):
        """Test that different inputs produce different keys"""
        key1 = detector._generate_cache_key("answer1", "context", "query")
        key2 = detector._generate_cache_key("answer2", "context", "query")
        
        assert key1 != key2


class TestResultParsing:
    """Test result parsing"""
    
    @pytest.mark.asyncio
    async def test_parse_result(self, detector):
        """Test parsing of raw HallBayes result"""
        raw_result = {
            "hallucination_detected": True,
            "risk_score": 0.75,
            "confidence": 0.85,
            "issues": [{"type": "factual", "description": "Unsupported claim"}],
            "suggestions": ["Add citation"]
        }
        
        result = detector._parse_result(raw_result, "answer", "context", "query")
        
        assert result.is_hallucination is True
        assert result.risk_score == 0.75
        assert result.confidence == 0.85
        assert result.risk_level == RiskLevel.HIGH
        assert len(result.detected_issues) > 0
        assert len(result.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_parse_result_with_defaults(self, detector):
        """Test parsing with missing fields"""
        raw_result = {}
        
        result = detector._parse_result(raw_result, "answer", "context", "query")
        
        assert isinstance(result.risk_score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.risk_level, RiskLevel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

