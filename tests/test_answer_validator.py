"""
Unit Tests for AnswerValidator
"""
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass

from observability.evaluation.answer_validator import (
    AnswerValidator,
    ValidationResult
)
from observability.evaluation.hallucination_detector import (
    HallucinationDetector,
    HallucinationResult
)
from observability.evaluation.config import (
    HallBayesConfig,
    RiskLevel,
    LLMBackend
)
from retrieval_pipeline.pipeline import RetrievalResult
from retrieval_pipeline.context.parent_retriever import RepackedContext


@pytest.fixture
def config():
    """Create test configuration"""
    return HallBayesConfig(
        backend=LLMBackend.OPENAI,
        enable_caching=True,
        timeout_seconds=5
    )


@pytest.fixture
def mock_detector(config):
    """Create mock detector"""
    detector = HallucinationDetector(config)
    
    # Mock the detect method
    async def mock_detect(answer, context, query=None, **kwargs):
        # Return simple result based on answer length
        risk_score = 0.3 if len(answer) > 50 else 0.7
        return HallucinationResult(
            is_hallucination=risk_score > 0.5,
            risk_score=risk_score,
            risk_level=RiskLevel.MEDIUM if risk_score > 0.5 else RiskLevel.LOW,
            confidence=0.8,
            detected_issues=[],
            suggestions=[]
        )
    
    detector.detect = mock_detect
    return detector


@pytest.fixture
def validator(mock_detector):
    """Create validator instance"""
    return AnswerValidator(
        detector=mock_detector,
        min_quality_score=0.5,
        min_confidence=0.5
    )


@pytest.fixture
def mock_retrieval_result():
    """Create mock retrieval result"""
    return RetrievalResult(
        query="What is machine learning?",
        final_context=RepackedContext(
            text="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            sources=[
                {
                    "document_id": "doc1",
                    "section_title": "Introduction to ML",
                    "relevance_score": 0.9,
                    "token_count": 50
                }
            ],
            total_tokens=150,
            context_windows=[{"start_pos": 0, "end_pos": 100}],
            relevance_scores=[0.9]
        ),
        answer_sources=[
            {
                "source_id": 1,
                "document_id": "doc1",
                "section_title": "Introduction to ML",
                "relevance_score": 0.9,
                "token_count": 50,
                "excerpt": "Machine learning is..."
            }
        ],
        total_time_ms=250.0,
        embedding_time_ms=50.0,
        search_time_ms=100.0,
        reranking_time_ms=75.0,
        context_time_ms=25.0,
        total_candidates=10,
        final_results=3
    )


class TestAnswerValidatorInitialization:
    """Test validator initialization"""
    
    def test_initialization_with_detector(self, mock_detector):
        """Test initialization with provided detector"""
        validator = AnswerValidator(detector=mock_detector)
        
        assert validator.detector is mock_detector
        assert validator.min_quality_score == 0.5
        assert validator.min_confidence == 0.5
    
    def test_initialization_with_config(self, config):
        """Test initialization with config"""
        validator = AnswerValidator(config=config)
        
        assert validator.config == config
        assert validator.detector is not None
    
    def test_initialization_defaults(self):
        """Test initialization with defaults"""
        validator = AnswerValidator()
        
        assert validator.detector is not None
        assert validator.config is not None


class TestAnswerValidation:
    """Test answer validation"""
    
    @pytest.mark.asyncio
    async def test_basic_validation(self, validator, mock_retrieval_result):
        """Test basic answer validation"""
        answer = "Machine learning is a branch of AI that allows computers to learn from data without explicit programming."
        
        result = await validator.validate(answer, mock_retrieval_result)
        
        assert isinstance(result, ValidationResult)
        assert result.answer == answer
        assert result.query == mock_retrieval_result.query
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.quality_score, float)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.quality_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_validation_with_good_answer(self, validator, mock_retrieval_result):
        """Test validation with good quality answer"""
        answer = "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It uses algorithms to find patterns and make decisions."
        
        result = await validator.validate(answer, mock_retrieval_result)
        
        # Long answer with context should have good scores
        assert result.quality_score >= 0.4
        assert result.confidence >= 0.3
        assert len(result.sources) > 0
    
    @pytest.mark.asyncio
    async def test_validation_with_short_answer(self, validator, mock_retrieval_result):
        """Test validation with short answer"""
        answer = "ML is AI."
        
        result = await validator.validate(answer, mock_retrieval_result)
        
        # Short answer should have warnings
        assert len(result.warnings) > 0
    
    @pytest.mark.asyncio
    async def test_validation_with_additional_context(self, validator, mock_retrieval_result):
        """Test validation with additional context"""
        answer = "Machine learning enables predictive analytics."
        additional_context = "Predictive analytics uses ML to forecast future trends."
        
        result = await validator.validate(
            answer,
            mock_retrieval_result,
            additional_context=additional_context
        )
        
        assert isinstance(result, ValidationResult)
    
    @pytest.mark.asyncio
    async def test_validation_with_empty_context(self, validator):
        """Test validation with empty context"""
        empty_result = RetrievalResult(
            query="Test query",
            final_context=RepackedContext(
                text="",
                sources=[],
                total_tokens=0,
                context_windows=[],
                relevance_scores=[]
            ),
            answer_sources=[],
            total_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=0,
            reranking_time_ms=0,
            context_time_ms=0,
            total_candidates=0,
            final_results=0
        )
        
        result = await validator.validate("Some answer", empty_result)
        
        # Should have warnings about no sources
        assert any("source" in w.lower() for w in result.warnings)


class TestQualityScoring:
    """Test quality scoring"""
    
    def test_quality_score_calculation(self, validator, mock_retrieval_result):
        """Test quality score calculation"""
        answer = "Machine learning is an AI technique."
        
        hallucination_result = HallucinationResult(
            is_hallucination=False,
            risk_score=0.2,
            risk_level=RiskLevel.LOW,
            confidence=0.9
        )
        
        quality_score = validator._calculate_quality_score(
            answer,
            mock_retrieval_result,
            hallucination_result
        )
        
        assert 0.0 <= quality_score <= 1.0
    
    def test_source_quality_calculation(self, validator, mock_retrieval_result):
        """Test source quality calculation"""
        source_quality = validator._calculate_source_quality(mock_retrieval_result)
        
        assert 0.0 <= source_quality <= 1.0
        assert source_quality > 0.5  # Good relevance scores
    
    def test_completeness_calculation(self, validator):
        """Test completeness calculation"""
        # Short answer
        short_score = validator._calculate_completeness("Short.")
        
        # Medium answer
        medium_score = validator._calculate_completeness(
            "This is a reasonably complete answer with multiple sentences. "
            "It provides detailed information."
        )
        
        # Long answer
        long_score = validator._calculate_completeness("Long answer. " * 50)
        
        assert short_score < medium_score
        assert medium_score <= 1.0


class TestConfidenceCalculation:
    """Test confidence calculation"""
    
    def test_confidence_calculation(self, validator, mock_retrieval_result):
        """Test confidence calculation"""
        hallucination_result = HallucinationResult(
            is_hallucination=False,
            risk_score=0.2,
            risk_level=RiskLevel.LOW,
            confidence=0.9
        )
        
        confidence = validator._calculate_confidence(
            hallucination_result,
            mock_retrieval_result,
            quality_score=0.8
        )
        
        assert 0.0 <= confidence <= 1.0
    
    def test_confidence_with_no_sources(self, validator):
        """Test confidence calculation with no sources"""
        no_source_result = RetrievalResult(
            query="Test",
            final_context=RepackedContext(
                text="",
                sources=[],
                total_tokens=0,
                context_windows=[],
                relevance_scores=[]
            ),
            answer_sources=[],
            total_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=0,
            reranking_time_ms=0,
            context_time_ms=0,
            total_candidates=0,
            final_results=0
        )
        
        hallucination_result = HallucinationResult(
            is_hallucination=True,
            risk_score=0.8,
            risk_level=RiskLevel.HIGH,
            confidence=0.3
        )
        
        confidence = validator._calculate_confidence(
            hallucination_result,
            no_source_result,
            quality_score=0.3
        )
        
        # Should be low confidence
        assert confidence < 0.5


class TestValidityChecking:
    """Test validity checking"""
    
    def test_valid_answer(self, validator):
        """Test that good answer passes validity check"""
        hallucination_result = HallucinationResult(
            is_hallucination=False,
            risk_score=0.2,
            risk_level=RiskLevel.LOW,
            confidence=0.9
        )
        
        is_valid = validator._check_validity(
            hallucination_result,
            quality_score=0.8,
            confidence=0.8
        )
        
        assert is_valid is True
    
    def test_critical_risk_invalid(self, validator):
        """Test that critical risk answers are invalid"""
        hallucination_result = HallucinationResult(
            is_hallucination=True,
            risk_score=0.9,
            risk_level=RiskLevel.CRITICAL,
            confidence=0.5
        )
        
        is_valid = validator._check_validity(
            hallucination_result,
            quality_score=0.8,
            confidence=0.8
        )
        
        assert is_valid is False
    
    def test_low_quality_invalid(self, validator):
        """Test that low quality answers are invalid"""
        hallucination_result = HallucinationResult(
            is_hallucination=False,
            risk_score=0.3,
            risk_level=RiskLevel.LOW,
            confidence=0.9
        )
        
        is_valid = validator._check_validity(
            hallucination_result,
            quality_score=0.2,  # Below threshold
            confidence=0.8
        )
        
        assert is_valid is False
    
    def test_low_confidence_invalid(self, validator):
        """Test that low confidence answers are invalid"""
        hallucination_result = HallucinationResult(
            is_hallucination=False,
            risk_score=0.3,
            risk_level=RiskLevel.LOW,
            confidence=0.9
        )
        
        is_valid = validator._check_validity(
            hallucination_result,
            quality_score=0.8,
            confidence=0.2  # Below threshold
        )
        
        assert is_valid is False


class TestWarningsAndRecommendations:
    """Test warning and recommendation generation"""
    
    def test_high_risk_warning(self, validator, mock_retrieval_result):
        """Test warning for high risk"""
        hallucination_result = HallucinationResult(
            is_hallucination=True,
            risk_score=0.8,
            risk_level=RiskLevel.HIGH,
            confidence=0.6
        )
        
        warnings = validator._generate_warnings(
            hallucination_result,
            quality_score=0.7,
            confidence=0.6,
            retrieval_result=mock_retrieval_result
        )
        
        assert any("high" in w.lower() and "risk" in w.lower() for w in warnings)
    
    def test_low_quality_warning(self, validator, mock_retrieval_result):
        """Test warning for low quality"""
        hallucination_result = HallucinationResult(
            is_hallucination=False,
            risk_score=0.3,
            risk_level=RiskLevel.LOW,
            confidence=0.8
        )
        
        warnings = validator._generate_warnings(
            hallucination_result,
            quality_score=0.4,  # Low quality
            confidence=0.8,
            retrieval_result=mock_retrieval_result
        )
        
        assert any("quality" in w.lower() for w in warnings)
    
    def test_recommendations_generation(self, validator, mock_retrieval_result):
        """Test recommendation generation"""
        hallucination_result = HallucinationResult(
            is_hallucination=False,
            risk_score=0.4,
            risk_level=RiskLevel.MEDIUM,
            confidence=0.6,
            suggestions=["Add citations"]
        )
        
        recommendations = validator._generate_recommendations(
            hallucination_result,
            quality_score=0.6,
            retrieval_result=mock_retrieval_result
        )
        
        # Should include suggestions from hallucination result
        assert "Add citations" in recommendations


class TestBatchValidation:
    """Test batch validation"""
    
    @pytest.mark.asyncio
    async def test_batch_validate(self, validator, mock_retrieval_result):
        """Test batch validation"""
        validations = [
            {"answer": "Answer 1 with sufficient length to pass quality checks.", "retrieval_result": mock_retrieval_result},
            {"answer": "Answer 2 with sufficient length to pass quality checks.", "retrieval_result": mock_retrieval_result},
            {"answer": "Answer 3 with sufficient length to pass quality checks.", "retrieval_result": mock_retrieval_result}
        ]
        
        results = await validator.batch_validate(validations, max_concurrent=2)
        
        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)


class TestValidationResult:
    """Test ValidationResult class"""
    
    def test_should_serve_valid_answer(self):
        """Test should_serve with valid answer"""
        result = ValidationResult(
            answer="Good answer",
            query="Test query",
            is_valid=True,
            hallucination_result=HallucinationResult(
                is_hallucination=False,
                risk_score=0.2,
                risk_level=RiskLevel.LOW,
                confidence=0.9
            ),
            confidence=0.8,
            quality_score=0.8
        )
        
        assert result.should_serve() is True
    
    def test_should_not_serve_critical_risk(self):
        """Test should_serve with critical risk"""
        result = ValidationResult(
            answer="Risky answer",
            query="Test query",
            is_valid=True,
            hallucination_result=HallucinationResult(
                is_hallucination=True,
                risk_score=0.9,
                risk_level=RiskLevel.CRITICAL,
                confidence=0.5
            ),
            confidence=0.8,
            quality_score=0.8
        )
        
        assert result.should_serve() is False
    
    def test_should_not_serve_invalid(self):
        """Test should_serve with invalid answer"""
        result = ValidationResult(
            answer="Invalid answer",
            query="Test query",
            is_valid=False,
            hallucination_result=HallucinationResult(
                is_hallucination=False,
                risk_score=0.3,
                risk_level=RiskLevel.LOW,
                confidence=0.9
            ),
            confidence=0.8,
            quality_score=0.8
        )
        
        assert result.should_serve() is False
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = ValidationResult(
            answer="Test answer",
            query="Test query",
            is_valid=True,
            hallucination_result=HallucinationResult(
                is_hallucination=False,
                risk_score=0.2,
                risk_level=RiskLevel.LOW,
                confidence=0.9
            ),
            confidence=0.8,
            quality_score=0.8,
            sources=[{"source_id": 1}],
            warnings=["Test warning"],
            recommendations=["Test recommendation"]
        )
        
        result_dict = result.to_dict()
        
        assert "answer" in result_dict
        assert "query" in result_dict
        assert "is_valid" in result_dict
        assert "hallucination_result" in result_dict
        assert "confidence" in result_dict
        assert "quality_score" in result_dict
    
    def test_get_risk_summary(self):
        """Test risk summary generation"""
        result = ValidationResult(
            answer="Test answer",
            query="Test query",
            is_valid=True,
            hallucination_result=HallucinationResult(
                is_hallucination=False,
                risk_score=0.2,
                risk_level=RiskLevel.LOW,
                confidence=0.9
            ),
            confidence=0.8,
            quality_score=0.8
        )
        
        risk_summary = result.get_risk_summary()
        
        assert "risk_level" in risk_summary
        assert "risk_score" in risk_summary
        assert "is_hallucination" in risk_summary
        assert "should_serve" in risk_summary


class TestContextExtraction:
    """Test context extraction"""
    
    def test_extract_context(self, validator, mock_retrieval_result):
        """Test context extraction from retrieval result"""
        context = validator._extract_context(mock_retrieval_result)
        
        assert len(context) > 0
        assert "machine learning" in context.lower()
    
    def test_extract_context_with_additional(self, validator, mock_retrieval_result):
        """Test context extraction with additional context"""
        additional = "Additional information about ML."
        context = validator._extract_context(mock_retrieval_result, additional)
        
        assert "additional information" in context.lower()
    
    def test_extract_context_truncation(self, validator):
        """Test context truncation when too long"""
        long_context = RetrievalResult(
            query="Test",
            final_context=RepackedContext(
                text="Very long text. " * 1000,
                sources=[],
                total_tokens=10000,
                context_windows=[],
                relevance_scores=[]
            ),
            answer_sources=[],
            total_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=0,
            reranking_time_ms=0,
            context_time_ms=0
        )
        
        context = validator._extract_context(long_context)
        
        # Should be truncated
        assert len(context) <= validator.config.max_context_length


class TestStatistics:
    """Test statistics"""
    
    def test_get_stats(self, validator):
        """Test get_stats method"""
        stats = validator.get_stats()
        
        assert "detector_stats" in stats
        assert "config" in stats
        assert "min_quality_score" in stats["config"]
        assert "min_confidence" in stats["config"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

