"""
Evaluation Module for Observability
Provides hallucination detection and answer validation capabilities
"""
from observability.evaluation.config import (
    HallBayesConfig,
    RiskLevel,
    LLMBackend,
    DEFAULT_CONFIG
)
from observability.evaluation.hallucination_detector import (
    HallucinationDetector,
    HallucinationResult,
    CacheEntry
)
from observability.evaluation.answer_validator import (
    AnswerValidator,
    ValidationResult
)

__all__ = [
    # Configuration
    "HallBayesConfig",
    "RiskLevel",
    "LLMBackend",
    "DEFAULT_CONFIG",
    
    # Hallucination Detection
    "HallucinationDetector",
    "HallucinationResult",
    "CacheEntry",
    
    # Answer Validation
    "AnswerValidator",
    "ValidationResult",
]

__version__ = "1.0.0"

