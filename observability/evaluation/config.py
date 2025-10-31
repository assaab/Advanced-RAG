"""
HallBayes Configuration Module
Provides configuration management for hallucination detection
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level enumeration for hallucination detection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LLMBackend(str, Enum):
    """Supported LLM backends for hallucination detection"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass
class HallBayesConfig:
    """
    Configuration for HallBayes integration
    
    Attributes:
        backend: LLM backend to use for detection
        model_name: Specific model name (None for auto-select)
        api_key: API key for the backend (if required)
        
        Risk thresholds (0.0 to 1.0):
        - low_risk_threshold: Below this = low risk
        - medium_risk_threshold: Below this = medium risk
        - high_risk_threshold: Below this = high risk
        - Above high_risk_threshold = critical risk
        
        Performance settings:
        - enable_caching: Cache validation results
        - cache_ttl_seconds: Cache time-to-live
        - async_validation: Run validation asynchronously
        - timeout_seconds: Maximum time for validation
        
        Feature flags:
        - generate_sla_certificates: Generate SLA certificates (expensive)
        - detailed_analysis: Include detailed hallucination analysis
        
        Error handling:
        - fail_gracefully: Don't fail pipeline on validation errors
        - default_risk_score: Score to use when validation fails
        - max_retries: Maximum retry attempts on failure
    """
    # Backend configuration
    backend: LLMBackend = LLMBackend.OPENAI
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    
    # Risk thresholds (0.0 to 1.0)
    low_risk_threshold: float = 0.25
    medium_risk_threshold: float = 0.50
    high_risk_threshold: float = 0.75
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    async_validation: bool = True
    timeout_seconds: int = 30
    max_concurrent_validations: int = 5
    
    # Feature flags
    generate_sla_certificates: bool = False
    detailed_analysis: bool = True
    include_suggestions: bool = True
    
    # Error handling
    fail_gracefully: bool = True
    default_risk_score: float = 0.5
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Logging
    log_level: str = "INFO"
    log_validations: bool = True
    
    # Additional settings
    confidence_threshold: float = 0.7
    min_context_length: int = 50
    max_context_length: int = 8000
    
    def __post_init__(self):
        """Validate configuration values"""
        # Validate thresholds
        if not (0.0 <= self.low_risk_threshold <= 1.0):
            raise ValueError("low_risk_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.medium_risk_threshold <= 1.0):
            raise ValueError("medium_risk_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.high_risk_threshold <= 1.0):
            raise ValueError("high_risk_threshold must be between 0.0 and 1.0")
        
        # Ensure proper threshold ordering
        if not (self.low_risk_threshold < self.medium_risk_threshold < self.high_risk_threshold):
            raise ValueError(
                "Thresholds must be ordered: low < medium < high"
            )
        
        # Validate performance settings
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        # Set API key from environment if not provided
        if not self.api_key:
            if self.backend == LLMBackend.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.backend == LLMBackend.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.backend == LLMBackend.HUGGINGFACE:
                self.api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    def get_risk_level(self, score: float) -> RiskLevel:
        """
        Map risk score to risk level
        
        Args:
            score: Risk score between 0.0 and 1.0
            
        Returns:
            Corresponding risk level
        """
        if score < self.low_risk_threshold:
            return RiskLevel.LOW
        elif score < self.medium_risk_threshold:
            return RiskLevel.MEDIUM
        elif score < self.high_risk_threshold:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def should_validate(self, context_length: int) -> bool:
        """
        Determine if validation should be performed based on context
        
        Args:
            context_length: Length of context to validate
            
        Returns:
            True if validation should be performed
        """
        return self.min_context_length <= context_length <= self.max_context_length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "backend": self.backend.value,
            "model_name": self.model_name,
            "risk_thresholds": {
                "low": self.low_risk_threshold,
                "medium": self.medium_risk_threshold,
                "high": self.high_risk_threshold,
            },
            "performance": {
                "caching": self.enable_caching,
                "cache_ttl": self.cache_ttl_seconds,
                "async": self.async_validation,
                "timeout": self.timeout_seconds,
            },
            "features": {
                "sla_certificates": self.generate_sla_certificates,
                "detailed_analysis": self.detailed_analysis,
                "suggestions": self.include_suggestions,
            },
            "error_handling": {
                "fail_gracefully": self.fail_gracefully,
                "default_risk_score": self.default_risk_score,
                "max_retries": self.max_retries,
            }
        }
    
    @classmethod
    def from_env(cls, **overrides) -> "HallBayesConfig":
        """
        Create configuration from environment variables with optional overrides
        
        Environment variables:
        - HALLBAYES_BACKEND: LLM backend (openai, anthropic, etc.)
        - HALLBAYES_MODEL_NAME: Model name
        - HALLBAYES_LOW_RISK_THRESHOLD: Low risk threshold
        - HALLBAYES_MEDIUM_RISK_THRESHOLD: Medium risk threshold
        - HALLBAYES_HIGH_RISK_THRESHOLD: High risk threshold
        - HALLBAYES_ENABLE_CACHING: Enable caching (true/false)
        - HALLBAYES_CACHE_TTL: Cache TTL in seconds
        - HALLBAYES_TIMEOUT: Timeout in seconds
        - HALLBAYES_GENERATE_SLA: Generate SLA certificates (true/false)
        
        Args:
            **overrides: Override specific configuration values
            
        Returns:
            HallBayesConfig instance
        """
        config_dict = {}
        
        # Backend
        if backend_str := os.getenv("HALLBAYES_BACKEND"):
            config_dict["backend"] = LLMBackend(backend_str.lower())
        
        # Model
        if model_name := os.getenv("HALLBAYES_MODEL_NAME"):
            config_dict["model_name"] = model_name
        
        # Risk thresholds
        if low_threshold := os.getenv("HALLBAYES_LOW_RISK_THRESHOLD"):
            config_dict["low_risk_threshold"] = float(low_threshold)
        if medium_threshold := os.getenv("HALLBAYES_MEDIUM_RISK_THRESHOLD"):
            config_dict["medium_risk_threshold"] = float(medium_threshold)
        if high_threshold := os.getenv("HALLBAYES_HIGH_RISK_THRESHOLD"):
            config_dict["high_risk_threshold"] = float(high_threshold)
        
        # Performance
        if caching := os.getenv("HALLBAYES_ENABLE_CACHING"):
            config_dict["enable_caching"] = caching.lower() in ("true", "1", "yes")
        if cache_ttl := os.getenv("HALLBAYES_CACHE_TTL"):
            config_dict["cache_ttl_seconds"] = int(cache_ttl)
        if timeout := os.getenv("HALLBAYES_TIMEOUT"):
            config_dict["timeout_seconds"] = int(timeout)
        
        # Features
        if generate_sla := os.getenv("HALLBAYES_GENERATE_SLA"):
            config_dict["generate_sla_certificates"] = generate_sla.lower() in ("true", "1", "yes")
        
        # Apply overrides
        config_dict.update(overrides)
        
        return cls(**config_dict)


# Default configuration instance
DEFAULT_CONFIG = HallBayesConfig()

