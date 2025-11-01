"""
Query API Endpoints
Handles query processing with answer generation and validation
"""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query as QueryParam
from fastapi.responses import JSONResponse

from api.models.schemas import (
    QueryRequest,
    QueryResponse,
    ValidationRequest,
    ValidationResponse,
    QueryLogResponse,
    QueryStatsResponse,
    ErrorResponse,
    SourceModel,
    HallucinationResultModel,
    ValidationResultModel,
    PerformanceMetricsModel,
    TokenUsageModel
)
from llm_generation import (
    IntegratedValidator,
    ValidatedAnswerConfig,
    GenerationConfig,
    LLMBackend,
    PromptTemplateLibrary,
    PromptType
)
from retrieval_pipeline.pipeline import RetrievalPipeline, PipelineConfig
from storage.postgresql.database import DatabaseManager
from storage.postgresql.models import QueryLog
from storage.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["query"])


# Dependency injection helpers
async def get_retrieval_pipeline() -> RetrievalPipeline:
    """Get retrieval pipeline instance"""
    # In production, this should be initialized once and reused
    opensearch_client = OpenSearchClient()
    db_manager = DatabaseManager()
    config = PipelineConfig()
    
    return RetrievalPipeline(opensearch_client, db_manager, config)


async def get_integrated_validator() -> IntegratedValidator:
    """Get integrated validator instance"""
    # In production, this should be initialized once and reused
    config = ValidatedAnswerConfig()
    return IntegratedValidator(config=config)


async def get_database_manager() -> DatabaseManager:
    """Get database manager instance"""
    return DatabaseManager()


# Main query endpoint
@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Process query with answer generation and validation",
    description="Retrieves relevant documents, generates answer, and validates for hallucinations"
)
async def process_query(
    request: QueryRequest,
    pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
    validator: IntegratedValidator = Depends(get_integrated_validator),
    db: DatabaseManager = Depends(get_database_manager)
) -> QueryResponse:
    """
    Process user query with full RAG pipeline
    
    Steps:
    1. Retrieve relevant documents
    2. Generate answer using LLM
    3. Validate answer for hallucinations
    4. Log query and results
    5. Return response
    """
    try:
        query_id = str(uuid.uuid4())
        logger.info(f"Processing query {query_id}: {request.query[:100]}...")
        
        # Step 1: Retrieve relevant documents
        retrieval_result = await pipeline.retrieve(
            query=request.query,
            filters=request.filters
        )
        
        # Step 2 & 3: Generate and validate answer
        generation_config = GenerationConfig(
            backend=LLMBackend(request.llm_backend.value),
            model_name=request.llm_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        validated_answer_config = ValidatedAnswerConfig(
            generation_config=generation_config,
            require_validation=request.include_validation
        )
        
        # Select prompt template
        prompt_template = None
        if request.prompt_type:
            prompt_template = PromptTemplateLibrary.get_template(
                PromptType(request.prompt_type.value)
            )
        
        validated_answer = await validator.generate_and_validate(
            retrieval_result=retrieval_result,
            prompt_template=prompt_template,
            custom_config=validated_answer_config,
            include_sources=request.include_sources
        )
        
        # Step 4: Log to database
        query_log = await _create_query_log(
            query_id=query_id,
            request=request,
            retrieval_result=retrieval_result,
            validated_answer=validated_answer
        )
        
        await db.log_query(query_log)
        
        # Step 5: Build response
        response = _build_query_response(
            query_id=query_id,
            request=request,
            retrieval_result=retrieval_result,
            validated_answer=validated_answer
        )
        
        logger.info(
            f"Query {query_id} completed: "
            f"valid={validated_answer.is_valid} "
            f"risk={validated_answer.validation_result.hallucination_result.risk_level.value} "
            f"time={validated_answer.total_time_ms:.0f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "detail": str(e),
                "error_code": "PROCESSING_ERROR"
            }
        )


def _build_query_response(
    query_id: str,
    request: QueryRequest,
    retrieval_result,
    validated_answer
) -> QueryResponse:
    """Build query response from results"""
    
    # Build sources list
    sources = []
    if request.include_sources and validated_answer.validation_result.sources:
        for source in validated_answer.validation_result.sources:
            sources.append(SourceModel(
                source_id=source.get("source_id", 0),
                document_id=source.get("document_id", "unknown"),
                section_title=source.get("section_title"),
                relevance_score=source.get("relevance_score", 0.0),
                token_count=source.get("token_count", 0)
            ))
    
    # Build validation result
    validation = None
    if request.include_validation:
        hall_result = validated_answer.validation_result.hallucination_result
        validation = ValidationResultModel(
            is_valid=validated_answer.is_valid,
            hallucination_result=HallucinationResultModel(
                is_hallucination=hall_result.is_hallucination,
                risk_score=hall_result.risk_score,
                risk_level=hall_result.risk_level.value,
                confidence=hall_result.confidence,
                detected_issues=hall_result.detected_issues,
                suggestions=hall_result.suggestions
            ),
            confidence=validated_answer.validation_result.confidence,
            quality_score=validated_answer.validation_result.quality_score,
            warnings=validated_answer.validation_result.warnings,
            recommendations=validated_answer.validation_result.recommendations
        )
    
    # Build performance metrics
    performance = PerformanceMetricsModel(
        retrieval_time_ms=retrieval_result.total_time_ms,
        generation_time_ms=validated_answer.generation_result.generation_time_ms,
        validation_time_ms=(
            validated_answer.total_time_ms - 
            validated_answer.generation_result.generation_time_ms - 
            retrieval_result.total_time_ms
        ) if request.include_validation else None,
        total_time_ms=validated_answer.total_time_ms + retrieval_result.total_time_ms
    )
    
    # Build token usage
    token_usage = TokenUsageModel(
        prompt_tokens=validated_answer.generation_result.prompt_tokens,
        completion_tokens=validated_answer.generation_result.completion_tokens,
        total_tokens=validated_answer.generation_result.total_tokens
    )
    
    # Build metadata
    metadata = {
        "backend": request.llm_backend.value,
        "model": validated_answer.generation_result.model,
        "regeneration_attempts": validated_answer.regeneration_attempts,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return QueryResponse(
        query_id=query_id,
        query=request.query,
        answer=validated_answer.answer,
        sources=sources,
        validation=validation,
        performance=performance,
        token_usage=token_usage,
        metadata=metadata
    )


async def _create_query_log(
    query_id: str,
    request: QueryRequest,
    retrieval_result,
    validated_answer
) -> QueryLog:
    """Create query log entry"""
    
    hall_result = validated_answer.validation_result.hallucination_result
    
    query_log = QueryLog(
        id=query_id,
        query_text=request.query,
        user_id=request.user_id,
        
        # Results
        retrieved_chunks=[
            chunk.get("chunk_id") 
            for chunk in (retrieval_result.answer_sources or [])
        ],
        final_answer=validated_answer.answer,
        confidence_score=int(validated_answer.validation_result.confidence * 5),  # Scale to 1-5
        
        # Performance metrics
        retrieval_time_ms=int(retrieval_result.total_time_ms),
        generation_time_ms=int(validated_answer.generation_result.generation_time_ms),
        total_time_ms=int(validated_answer.total_time_ms + retrieval_result.total_time_ms),
        
        # Hallucination detection
        hallucination_score=hall_result.risk_score,
        hallucination_risk_level=hall_result.risk_level.value,
        is_hallucination=hall_result.is_hallucination,
        quality_score=validated_answer.validation_result.quality_score,
        validation_confidence=validated_answer.validation_result.confidence,
        sla_certificate=hall_result.sla_certificate,
        validation_warnings=validated_answer.validation_result.warnings,
        sources_used=[
            source.get("document_id") 
            for source in (validated_answer.validation_result.sources or [])
        ],
        
        # LLM metadata
        llm_backend=request.llm_backend.value,
        llm_model=validated_answer.generation_result.model,
        prompt_tokens=validated_answer.generation_result.prompt_tokens,
        completion_tokens=validated_answer.generation_result.completion_tokens,
        total_tokens=validated_answer.generation_result.total_tokens,
        regeneration_attempts=validated_answer.regeneration_attempts
    )
    
    return query_log


# Standalone validation endpoint
@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validate existing answer for hallucinations",
    description="Validates a pre-generated answer against provided context"
)
async def validate_answer(
    request: ValidationRequest,
    validator: IntegratedValidator = Depends(get_integrated_validator)
) -> ValidationResponse:
    """
    Validate existing answer for hallucinations
    
    Useful for validating answers from external sources or testing
    """
    try:
        import time
        start_time = time.time()
        
        # Create minimal retrieval result for validation
        from retrieval_pipeline.context.parent_retriever import RepackedContext
        from retrieval_pipeline.pipeline import RetrievalResult
        
        repacked_context = RepackedContext(
            text=request.context,
            sources=[],
            total_tokens=len(request.context.split()),
            context_windows=[],
            relevance_scores=[]
        )
        
        retrieval_result = RetrievalResult(
            query=request.query or "validation query",
            final_context=repacked_context,
            answer_sources=[],
            total_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=0,
            reranking_time_ms=0,
            context_time_ms=0
        )
        
        # Perform validation
        validation_result = await validator.validate_existing(
            answer=request.answer,
            retrieval_result=retrieval_result
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build response
        hall_result = validation_result.hallucination_result
        
        response = ValidationResponse(
            validation_result=ValidationResultModel(
                is_valid=validation_result.is_valid,
                hallucination_result=HallucinationResultModel(
                    is_hallucination=hall_result.is_hallucination,
                    risk_score=hall_result.risk_score,
                    risk_level=hall_result.risk_level.value,
                    confidence=hall_result.confidence,
                    detected_issues=hall_result.detected_issues,
                    suggestions=hall_result.suggestions
                ),
                confidence=validation_result.confidence,
                quality_score=validation_result.quality_score,
                warnings=validation_result.warnings,
                recommendations=validation_result.recommendations
            ),
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Validation failed",
                "detail": str(e),
                "error_code": "VALIDATION_ERROR"
            }
        )


# Get query log by ID
@router.get(
    "/query/{query_id}",
    response_model=QueryLogResponse,
    summary="Get query log by ID",
    description="Retrieve details of a previously processed query"
)
async def get_query_log(
    query_id: str,
    db: DatabaseManager = Depends(get_database_manager)
) -> QueryLogResponse:
    """Get query log by ID"""
    try:
        async with db.get_session() as session:
            query_log = await session.get(QueryLog, query_id)
            
            if not query_log:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Query not found",
                        "detail": f"No query found with ID: {query_id}",
                        "error_code": "NOT_FOUND"
                    }
                )
            
            return QueryLogResponse(**query_log.to_dict())
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve query log: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve query log",
                "detail": str(e),
                "error_code": "DATABASE_ERROR"
            }
        )


# Get query statistics
@router.get(
    "/stats",
    response_model=QueryStatsResponse,
    summary="Get query statistics",
    description="Get aggregated statistics about query processing"
)
async def get_query_stats(
    days: int = QueryParam(7, ge=1, le=365, description="Number of days to analyze"),
    db: DatabaseManager = Depends(get_database_manager)
) -> QueryStatsResponse:
    """Get query statistics for the specified time period"""
    try:
        from sqlalchemy import func, select
        
        # Calculate date range
        start_date = datetime.utcnow() - timedelta(days=days)
        
        async with db.get_session() as session:
            # Get total queries
            total_query = select(func.count(QueryLog.id)).where(
                QueryLog.created_at >= start_date
            )
            total_result = await session.execute(total_query)
            total_queries = total_result.scalar() or 0
            
            # Get average hallucination score
            avg_hall_query = select(func.avg(QueryLog.hallucination_score)).where(
                QueryLog.created_at >= start_date,
                QueryLog.hallucination_score.isnot(None)
            )
            avg_hall_result = await session.execute(avg_hall_query)
            avg_hallucination_score = avg_hall_result.scalar()
            
            # Get average quality score
            avg_quality_query = select(func.avg(QueryLog.quality_score)).where(
                QueryLog.created_at >= start_date,
                QueryLog.quality_score.isnot(None)
            )
            avg_quality_result = await session.execute(avg_quality_query)
            avg_quality_score = avg_quality_result.scalar()
            
            # Get risk level distribution
            risk_query = select(
                QueryLog.hallucination_risk_level,
                func.count(QueryLog.id)
            ).where(
                QueryLog.created_at >= start_date,
                QueryLog.hallucination_risk_level.isnot(None)
            ).group_by(QueryLog.hallucination_risk_level)
            
            risk_result = await session.execute(risk_query)
            risk_distribution = {row[0]: row[1] for row in risk_result}
            
            # Get backend distribution
            backend_query = select(
                QueryLog.llm_backend,
                func.count(QueryLog.id)
            ).where(
                QueryLog.created_at >= start_date,
                QueryLog.llm_backend.isnot(None)
            ).group_by(QueryLog.llm_backend)
            
            backend_result = await session.execute(backend_query)
            backend_distribution = {row[0]: row[1] for row in backend_result}
            
            # Get average time
            avg_time_query = select(func.avg(QueryLog.total_time_ms)).where(
                QueryLog.created_at >= start_date,
                QueryLog.total_time_ms.isnot(None)
            )
            avg_time_result = await session.execute(avg_time_query)
            avg_total_time_ms = avg_time_result.scalar()
            
            # Get average tokens
            avg_tokens_query = select(func.avg(QueryLog.total_tokens)).where(
                QueryLog.created_at >= start_date,
                QueryLog.total_tokens.isnot(None)
            )
            avg_tokens_result = await session.execute(avg_tokens_query)
            avg_tokens = avg_tokens_result.scalar()
            
            return QueryStatsResponse(
                total_queries=total_queries,
                avg_hallucination_score=avg_hallucination_score,
                avg_quality_score=avg_quality_score,
                risk_distribution=risk_distribution,
                backend_distribution=backend_distribution,
                avg_total_time_ms=avg_total_time_ms,
                avg_tokens=avg_tokens
            )
            
    except Exception as e:
        logger.error(f"Failed to retrieve query stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve statistics",
                "detail": str(e),
                "error_code": "DATABASE_ERROR"
            }
        )


# Get hallucination details for a query
@router.get(
    "/query/{query_id}/hallucination",
    summary="Get hallucination detection details",
    description="Get detailed hallucination detection results for a specific query"
)
async def get_hallucination_details(
    query_id: str,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get detailed hallucination information for a query"""
    try:
        async with db.get_session() as session:
            query_log = await session.get(QueryLog, query_id)
            
            if not query_log:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Query not found",
                        "detail": f"No query found with ID: {query_id}",
                        "error_code": "NOT_FOUND"
                    }
                )
            
            return {
                "query_id": query_id,
                "query_text": query_log.query_text,
                "risk_summary": query_log.get_risk_summary(),
                "validation_warnings": query_log.validation_warnings,
                "sla_certificate": query_log.sla_certificate,
                "is_high_quality": query_log.is_high_quality()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve hallucination details: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve hallucination details",
                "detail": str(e),
                "error_code": "DATABASE_ERROR"
            }
        )

