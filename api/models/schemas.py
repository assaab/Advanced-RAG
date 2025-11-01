"""
API Request and Response Schemas
Pydantic models for FastAPI endpoints
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class LLMBackendEnum(str, Enum):
    """LLM backend options"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


class PromptTypeEnum(str, Enum):
    """Prompt type options"""
    STANDARD = "standard"
    CONCISE = "concise"
    DETAILED = "detailed"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    EXPLANATION = "explanation"


class RiskLevelEnum(str, Enum):
    """Risk level categories"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Request Models
class QueryRequest(BaseModel):
    """
    Request model for query endpoint
    
    Attributes:
        query: User query text
        user_id: Optional user identifier
        filters: Optional search filters
        llm_backend: LLM backend to use
        llm_model: Specific model name (optional)
        prompt_type: Type of prompt to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        include_validation: Whether to validate answer
        include_sources: Whether to include source documents
    """
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    user_id: Optional[str] = Field(None, max_length=255, description="User identifier")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    
    # LLM configuration
    llm_backend: LLMBackendEnum = Field(
        LLMBackendEnum.OPENAI,
        description="LLM backend to use"
    )
    llm_model: Optional[str] = Field(None, description="Specific model name")
    prompt_type: Optional[PromptTypeEnum] = Field(
        None,
        description="Prompt type (auto-selected if not provided)"
    )
    temperature: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        512,
        ge=50,
        le=2048,
        description="Maximum tokens to generate"
    )
    
    # Validation options
    include_validation: bool = Field(
        True,
        description="Enable hallucination validation"
    )
    include_sources: bool = Field(
        True,
        description="Include source documents in response"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How do model-training costs change over time?",
                "user_id": "user_123",
                "llm_backend": "openai",
                "temperature": 0.3,
                "include_validation": True,
                "include_sources": True
            }
        }


class ValidationRequest(BaseModel):
    """
    Request model for standalone validation endpoint
    
    Attributes:
        answer: Answer text to validate
        context: Context text used to generate answer
        query: Original query (optional)
    """
    answer: str = Field(..., min_length=1, description="Answer to validate")
    context: str = Field(..., min_length=1, description="Context used")
    query: Optional[str] = Field(None, description="Original query")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "Model training costs have decreased significantly...",
                "context": "According to recent studies, GPU prices have fallen...",
                "query": "How do model-training costs change over time?"
            }
        }


# Response Models
class SourceModel(BaseModel):
    """Source document information"""
    source_id: int = Field(..., description="Source identifier")
    document_id: str = Field(..., description="Document ID")
    section_title: Optional[str] = Field(None, description="Section title")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    token_count: int = Field(..., ge=0, description="Token count")


class HallucinationResultModel(BaseModel):
    """Hallucination detection result"""
    is_hallucination: bool = Field(..., description="Whether hallucination detected")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score")
    risk_level: RiskLevelEnum = Field(..., description="Risk level category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in detection")
    detected_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detected issues"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Improvement suggestions"
    )


class ValidationResultModel(BaseModel):
    """Validation result"""
    is_valid: bool = Field(..., description="Whether answer is valid")
    hallucination_result: HallucinationResultModel
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )


class PerformanceMetricsModel(BaseModel):
    """Performance metrics"""
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    validation_time_ms: Optional[float] = Field(
        None,
        description="Validation time in milliseconds"
    )
    total_time_ms: float = Field(..., description="Total time in milliseconds")


class TokenUsageModel(BaseModel):
    """Token usage information"""
    prompt_tokens: int = Field(..., ge=0, description="Prompt tokens")
    completion_tokens: int = Field(..., ge=0, description="Completion tokens")
    total_tokens: int = Field(..., ge=0, description="Total tokens")


class QueryResponse(BaseModel):
    """
    Response model for query endpoint
    
    Attributes:
        query_id: Unique query identifier
        query: Original query text
        answer: Generated answer
        sources: List of source documents
        validation: Validation result (if enabled)
        performance: Performance metrics
        token_usage: Token usage information
        metadata: Additional metadata
    """
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceModel] = Field(
        default_factory=list,
        description="Source documents"
    )
    validation: Optional[ValidationResultModel] = Field(
        None,
        description="Validation result"
    )
    performance: PerformanceMetricsModel = Field(..., description="Performance metrics")
    token_usage: TokenUsageModel = Field(..., description="Token usage")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query_id": "query_abc123",
                "query": "How do model-training costs change over time?",
                "answer": "Model training costs have decreased significantly...",
                "sources": [
                    {
                        "source_id": 1,
                        "document_id": "doc_123",
                        "section_title": "Cost Analysis",
                        "relevance_score": 0.95,
                        "token_count": 250
                    }
                ],
                "validation": {
                    "is_valid": True,
                    "hallucination_result": {
                        "is_hallucination": False,
                        "risk_score": 0.15,
                        "risk_level": "low",
                        "confidence": 0.92,
                        "detected_issues": [],
                        "suggestions": []
                    },
                    "confidence": 0.88,
                    "quality_score": 0.91,
                    "warnings": [],
                    "recommendations": []
                },
                "performance": {
                    "retrieval_time_ms": 245.5,
                    "generation_time_ms": 1320.8,
                    "validation_time_ms": 892.3,
                    "total_time_ms": 2458.6
                },
                "token_usage": {
                    "prompt_tokens": 1250,
                    "completion_tokens": 180,
                    "total_tokens": 1430
                },
                "metadata": {
                    "backend": "openai",
                    "model": "gpt-3.5-turbo",
                    "regeneration_attempts": 0
                }
            }
        }


class ValidationResponse(BaseModel):
    """Response model for validation endpoint"""
    validation_result: ValidationResultModel
    processing_time_ms: float = Field(..., description="Processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "validation_result": {
                    "is_valid": True,
                    "hallucination_result": {
                        "is_hallucination": False,
                        "risk_score": 0.15,
                        "risk_level": "low",
                        "confidence": 0.92,
                        "detected_issues": [],
                        "suggestions": []
                    },
                    "confidence": 0.88,
                    "quality_score": 0.91,
                    "warnings": [],
                    "recommendations": []
                },
                "processing_time_ms": 892.3
            }
        }


class QueryLogResponse(BaseModel):
    """Response model for query log retrieval"""
    id: str
    query_text: str
    final_answer: Optional[str] = None
    hallucination_score: Optional[float] = None
    hallucination_risk_level: Optional[str] = None
    quality_score: Optional[float] = None
    validation_confidence: Optional[float] = None
    llm_backend: Optional[str] = None
    llm_model: Optional[str] = None
    total_tokens: Optional[int] = None
    total_time_ms: Optional[int] = None
    user_rating: Optional[int] = None
    created_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "id": "query_abc123",
                "query_text": "How do model-training costs change?",
                "final_answer": "Model training costs have decreased...",
                "hallucination_score": 0.15,
                "hallucination_risk_level": "low",
                "quality_score": 0.91,
                "validation_confidence": 0.88,
                "llm_backend": "openai",
                "llm_model": "gpt-3.5-turbo",
                "total_tokens": 1430,
                "total_time_ms": 2458,
                "user_rating": 5,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class QueryStatsResponse(BaseModel):
    """Response model for query statistics"""
    total_queries: int = Field(..., description="Total number of queries")
    avg_hallucination_score: Optional[float] = Field(
        None,
        description="Average hallucination score"
    )
    avg_quality_score: Optional[float] = Field(
        None,
        description="Average quality score"
    )
    risk_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of risk levels"
    )
    backend_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of LLM backends"
    )
    avg_total_time_ms: Optional[float] = Field(
        None,
        description="Average total time"
    )
    avg_tokens: Optional[float] = Field(None, description="Average tokens used")
    
    class Config:
        schema_extra = {
            "example": {
                "total_queries": 1523,
                "avg_hallucination_score": 0.23,
                "avg_quality_score": 0.85,
                "risk_distribution": {
                    "low": 1205,
                    "medium": 268,
                    "high": 45,
                    "critical": 5
                },
                "backend_distribution": {
                    "openai": 1200,
                    "ollama": 323
                },
                "avg_total_time_ms": 2350.5,
                "avg_tokens": 1450.2
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    error_code: Optional[str] = Field(None, description="Error code")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Query processing failed",
                "detail": "Retrieval pipeline timeout after 30 seconds",
                "error_code": "TIMEOUT_ERROR"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Component status"
    )
    timestamp: datetime = Field(..., description="Check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "database": "healthy",
                    "opensearch": "healthy",
                    "llm_backend": "healthy"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

