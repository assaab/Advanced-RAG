"""
Main FastAPI Application
Advanced-RAG API with integrated answer generation and hallucination validation
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.endpoints.query import router as query_router
from api.models.schemas import HealthCheckResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events
    
    Startup:
    - Initialize database connections
    - Initialize LLM clients
    - Load models
    
    Shutdown:
    - Close database connections
    - Cleanup resources
    """
    logger.info("Starting Advanced-RAG API...")
    
    # Startup logic here
    # In production, initialize shared resources:
    # - Database connection pools
    # - OpenSearch clients
    # - LLM clients
    # - Model loading
    
    yield
    
    # Shutdown logic here
    logger.info("Shutting down Advanced-RAG API...")
    # Close connections, cleanup resources


# Create FastAPI application
app = FastAPI(
    title="Advanced-RAG API",
    description="""
    Advanced Retrieval-Augmented Generation API with integrated hallucination detection.
    
    ## Features
    
    - **Multi-stage Retrieval**: Token-level embeddings with MaxSim search
    - **Cascade Reranking**: TILDE → MonoT5 → RankLLaMA pipeline
    - **Answer Generation**: Support for OpenAI, Ollama, and Anthropic
    - **Hallucination Detection**: Real-time validation using HallBayes
    - **Query Logging**: Comprehensive analytics and feedback tracking
    
    ## Key Endpoints
    
    - `POST /api/v1/query` - Process query with full RAG pipeline
    - `POST /api/v1/validate` - Validate existing answers
    - `GET /api/v1/query/{id}` - Retrieve query logs
    - `GET /api/v1/stats` - Get query statistics
    - `GET /health` - Health check
    
    ## Usage Example
    
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/api/v1/query",
        json={
            "query": "How do model-training costs change over time?",
            "llm_backend": "openai",
            "include_validation": True
        }
    )
    
    result = response.json()
    print(f"Answer: {result['answer']}")
    print(f"Risk Level: {result['validation']['hallucination_result']['risk_level']}")
    ```
    """,
    version="1.0.0",
    contact={
        "name": "Advanced-RAG Team",
        "url": "https://github.com/your-org/advanced-rag"
    },
    license_info={
        "name": "MIT License"
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "errors": exc.errors(),
            "error_code": "VALIDATION_ERROR"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "error_code": "INTERNAL_ERROR"
        }
    )


# Include routers
app.include_router(query_router)


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["health"],
    summary="Health check",
    description="Check API health and component status"
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint
    
    Returns status of API and its components
    """
    # In production, check actual component health
    components = {
        "database": "healthy",
        "opensearch": "healthy",
        "llm_backend": "healthy"
    }
    
    # Check if any components are unhealthy
    overall_status = "healthy" if all(
        status == "healthy" for status in components.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version="1.0.0",
        components=components,
        timestamp=datetime.utcnow()
    )


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Advanced-RAG API",
        "version": "1.0.0",
        "description": "Advanced Retrieval-Augmented Generation with Hallucination Detection",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "query": "/api/v1/query",
            "validate": "/api/v1/validate",
            "stats": "/api/v1/stats"
        }
    }


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} - {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

