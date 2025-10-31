"""
Unit Tests for HallBayes Configuration
"""
import pytest
import os
from unittest.mock import patch

from observability.evaluation.config import (
    HallBayesConfig,
    RiskLevel,
    LLMBackend,
    DEFAULT_CONFIG
)


class TestHallBayesConfig:
    """Test HallBayesConfig class"""
    
    def test_default_initialization(self):
        """Test default configuration"""
        config = HallBayesConfig()
        
        assert config.backend == LLMBackend.OPENAI
        assert config.model_name is None
        assert config.low_risk_threshold == 0.25
        assert config.medium_risk_threshold == 0.50
        assert config.high_risk_threshold == 0.75
        assert config.enable_caching is True
        assert config.fail_gracefully is True
    
    def test_custom_initialization(self):
        """Test custom configuration"""
        config = HallBayesConfig(
            backend=LLMBackend.ANTHROPIC,
            model_name="claude-3",
            low_risk_threshold=0.2,
            medium_risk_threshold=0.5,
            high_risk_threshold=0.8,
            enable_caching=False
        )
        
        assert config.backend == LLMBackend.ANTHROPIC
        assert config.model_name == "claude-3"
        assert config.low_risk_threshold == 0.2
        assert config.enable_caching is False
    
    def test_threshold_validation(self):
        """Test threshold validation"""
        # Invalid: thresholds out of range
        with pytest.raises(ValueError):
            HallBayesConfig(low_risk_threshold=-0.1)
        
        with pytest.raises(ValueError):
            HallBayesConfig(high_risk_threshold=1.5)
        
        # Invalid: thresholds not ordered correctly
        with pytest.raises(ValueError):
            HallBayesConfig(
                low_risk_threshold=0.7,
                medium_risk_threshold=0.5,
                high_risk_threshold=0.8
            )
    
    def test_performance_validation(self):
        """Test performance setting validation"""
        with pytest.raises(ValueError):
            HallBayesConfig(cache_ttl_seconds=-1)
        
        with pytest.raises(ValueError):
            HallBayesConfig(timeout_seconds=0)
        
        with pytest.raises(ValueError):
            HallBayesConfig(max_retries=-1)


class TestRiskLevel:
    """Test RiskLevel enum"""
    
    def test_risk_levels(self):
        """Test risk level values"""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"
    
    def test_get_risk_level(self):
        """Test risk level mapping"""
        config = HallBayesConfig(
            low_risk_threshold=0.25,
            medium_risk_threshold=0.50,
            high_risk_threshold=0.75
        )
        
        assert config.get_risk_level(0.1) == RiskLevel.LOW
        assert config.get_risk_level(0.3) == RiskLevel.MEDIUM
        assert config.get_risk_level(0.6) == RiskLevel.HIGH
        assert config.get_risk_level(0.9) == RiskLevel.CRITICAL
        
        # Edge cases
        assert config.get_risk_level(0.25) == RiskLevel.MEDIUM
        assert config.get_risk_level(0.50) == RiskLevel.HIGH
        assert config.get_risk_level(0.75) == RiskLevel.CRITICAL


class TestLLMBackend:
    """Test LLMBackend enum"""
    
    def test_backend_values(self):
        """Test backend values"""
        assert LLMBackend.OPENAI.value == "openai"
        assert LLMBackend.ANTHROPIC.value == "anthropic"
        assert LLMBackend.HUGGINGFACE.value == "huggingface"
        assert LLMBackend.OLLAMA.value == "ollama"


class TestConfigValidation:
    """Test configuration validation methods"""
    
    def test_should_validate_valid_context(self):
        """Test should_validate with valid context"""
        config = HallBayesConfig(
            min_context_length=50,
            max_context_length=8000
        )
        
        assert config.should_validate(100) is True
        assert config.should_validate(5000) is True
    
    def test_should_validate_invalid_context(self):
        """Test should_validate with invalid context"""
        config = HallBayesConfig(
            min_context_length=50,
            max_context_length=8000
        )
        
        assert config.should_validate(10) is False  # Too short
        assert config.should_validate(10000) is False  # Too long


class TestConfigSerialization:
    """Test configuration serialization"""
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = HallBayesConfig(
            backend=LLMBackend.OPENAI,
            model_name="gpt-4",
            low_risk_threshold=0.2,
            enable_caching=True
        )
        
        config_dict = config.to_dict()
        
        assert "backend" in config_dict
        assert config_dict["backend"] == "openai"
        assert "model_name" in config_dict
        assert "risk_thresholds" in config_dict
        assert "performance" in config_dict
        assert "features" in config_dict
        assert "error_handling" in config_dict
    
    def test_to_dict_structure(self):
        """Test dictionary structure"""
        config = HallBayesConfig()
        config_dict = config.to_dict()
        
        # Check nested structure
        assert "low" in config_dict["risk_thresholds"]
        assert "medium" in config_dict["risk_thresholds"]
        assert "high" in config_dict["risk_thresholds"]
        
        assert "caching" in config_dict["performance"]
        assert "async" in config_dict["performance"]
        assert "timeout" in config_dict["performance"]


class TestConfigFromEnvironment:
    """Test configuration from environment variables"""
    
    @patch.dict(os.environ, {
        "HALLBAYES_BACKEND": "anthropic",
        "HALLBAYES_MODEL_NAME": "claude-3",
        "HALLBAYES_LOW_RISK_THRESHOLD": "0.2",
        "HALLBAYES_MEDIUM_RISK_THRESHOLD": "0.5",
        "HALLBAYES_HIGH_RISK_THRESHOLD": "0.8",
        "HALLBAYES_ENABLE_CACHING": "true",
        "HALLBAYES_CACHE_TTL": "7200",
        "HALLBAYES_TIMEOUT": "10",
        "HALLBAYES_GENERATE_SLA": "false"
    })
    def test_from_env(self):
        """Test configuration from environment variables"""
        config = HallBayesConfig.from_env()
        
        assert config.backend == LLMBackend.ANTHROPIC
        assert config.model_name == "claude-3"
        assert config.low_risk_threshold == 0.2
        assert config.medium_risk_threshold == 0.5
        assert config.high_risk_threshold == 0.8
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 7200
        assert config.timeout_seconds == 10
        assert config.generate_sla_certificates is False
    
    @patch.dict(os.environ, {
        "HALLBAYES_BACKEND": "openai",
        "OPENAI_API_KEY": "test-key-123"
    })
    def test_from_env_with_api_key(self):
        """Test API key loading from environment"""
        config = HallBayesConfig.from_env()
        
        assert config.backend == LLMBackend.OPENAI
        assert config.api_key == "test-key-123"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_with_defaults(self):
        """Test from_env with no environment variables"""
        config = HallBayesConfig.from_env()
        
        # Should use defaults
        assert config.backend == LLMBackend.OPENAI
        assert config.model_name is None
    
    def test_from_env_with_overrides(self):
        """Test from_env with overrides"""
        config = HallBayesConfig.from_env(
            backend=LLMBackend.OLLAMA,
            model_name="llama2",
            enable_caching=False
        )
        
        # Overrides should take precedence
        assert config.backend == LLMBackend.OLLAMA
        assert config.model_name == "llama2"
        assert config.enable_caching is False


class TestAPIKeyLoading:
    """Test API key loading"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_openai_key_loading(self):
        """Test OpenAI API key loading"""
        config = HallBayesConfig(backend=LLMBackend.OPENAI)
        
        assert config.api_key == "test-openai-key"
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    def test_anthropic_key_loading(self):
        """Test Anthropic API key loading"""
        config = HallBayesConfig(backend=LLMBackend.ANTHROPIC)
        
        assert config.api_key == "test-anthropic-key"
    
    @patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test-hf-key"})
    def test_huggingface_key_loading(self):
        """Test HuggingFace API key loading"""
        config = HallBayesConfig(backend=LLMBackend.HUGGINGFACE)
        
        assert config.api_key == "test-hf-key"
    
    def test_explicit_key_override(self):
        """Test explicit API key overrides environment"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            config = HallBayesConfig(
                backend=LLMBackend.OPENAI,
                api_key="explicit-key"
            )
            
            assert config.api_key == "explicit-key"


class TestDefaultConfig:
    """Test default configuration constant"""
    
    def test_default_config_exists(self):
        """Test that DEFAULT_CONFIG is available"""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, HallBayesConfig)
    
    def test_default_config_values(self):
        """Test default configuration values"""
        assert DEFAULT_CONFIG.backend == LLMBackend.OPENAI
        assert DEFAULT_CONFIG.enable_caching is True
        assert DEFAULT_CONFIG.fail_gracefully is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

