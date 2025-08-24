"""
Main Retrieval Pipeline
Orchestrates the complete advanced RAG pipeline with all enhancements
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import logging

from retrieval_pipeline.embeddings.multi_vector_embedder import MultiVectorEmbedder, QueryEmbedding
from retrieval_pipeline.search.maxsim_searcher import MaxSimSearcher, SearchConfig, MaxSimResult
from retrieval_pipeline.reranking.cascade_reranker import CascadeReranker, CascadeConfig, CascadeResult
from retrieval_pipeline.context.parent_retriever import ParentRetriever, ReverseRepacker, ContextConfig, RepackedContext
from storage.opensearch.client import OpenSearchClient
from storage.postgresql.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    colbert_dim: int = 128
    
    # Search configuration
    search_config: Optional[SearchConfig] = None
    
    # Reranking configuration
    cascade_config: Optional[CascadeConfig] = None
    
    # Context configuration
    context_config: Optional[ContextConfig] = None
    
    # Performance settings
    enable_caching: bool = True
    parallel_processing: bool = True
    timeout_seconds: int = 30


@dataclass
class RetrievalResult:
    """Complete retrieval result"""
    query: str
    final_context: RepackedContext
    answer_sources: List[Dict[str, Any]]
    
    # Performance metrics
    total_time_ms: float
    embedding_time_ms: float
    search_time_ms: float
    reranking_time_ms: float
    context_time_ms: float
    
    # Intermediate results for debugging
    query_embedding: Optional[QueryEmbedding] = None
    maxsim_results: Optional[List[MaxSimResult]] = None
    cascade_results: Optional[List[CascadeResult]] = None
    
    # Metadata
    total_candidates: int = 0
    final_results: int = 0
    pipeline_version: str = "v1.0"


class RetrievalPipeline:
    """
    Advanced RAG Retrieval Pipeline
    
    Complete flow:
    1. Multi-vector embedding (ColBERT/ColPali)
    2. MaxSim search with late interaction
    3. Three-stage cascade reranking (TILDE → MonoT5 → RankLLaMA)
    4. Parent retrieval + reverse repacking
    5. Context optimization for LLM generation
    """
    
    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        db_manager: DatabaseManager,
        config: Optional[PipelineConfig] = None
    ):
        self.opensearch = opensearch_client
        self.db = db_manager
        self.config = config or PipelineConfig()
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Cache for performance
        self._embedding_cache = {} if self.config.enable_caching else None
        
        logger.info("Advanced RAG Pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        # Multi-vector embedder
        self.embedder = MultiVectorEmbedder(
            model_name=self.config.embedding_model,
            colbert_dim=self.config.colbert_dim
        )
        
        # MaxSim searcher
        self.searcher = MaxSimSearcher(self.opensearch)
        
        # Cascade reranker
        cascade_config = self.config.cascade_config or CascadeConfig()
        self.reranker = CascadeReranker(cascade_config)
        
        # Parent retriever and repacker
        self.parent_retriever = ParentRetriever(self.db)
        self.repacker = ReverseRepacker()
        
        logger.info("All pipeline components initialized")
    
    async def retrieve(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Execute complete retrieval pipeline
        
        Args:
            query: User query
            filters: Optional filters (document_ids, categories, etc.)
        
        Returns:
            Complete retrieval result with context and metadata
        """
        start_time = time.time()
        
        try:
            # Stage 1: Multi-vector embedding
            embedding_start = time.time()
            query_embedding = await self._embed_query(query)
            embedding_time = (time.time() - embedding_start) * 1000
            
            # Stage 2: MaxSim search
            search_start = time.time()
            maxsim_results = await self._maxsim_search(query_embedding, filters)
            search_time = (time.time() - search_start) * 1000
            
            # Stage 3: Cascade reranking
            reranking_start = time.time()
            cascade_results = await self._cascade_rerank(query, maxsim_results)
            reranking_time = (time.time() - reranking_start) * 1000
            
            # Stage 4: Parent retrieval and context repacking
            context_start = time.time()
            final_context = await self._retrieve_and_repack_context(cascade_results)
            context_time = (time.time() - context_start) * 1000
            
            # Build final result
            total_time = (time.time() - start_time) * 1000
            
            result = RetrievalResult(
                query=query,
                final_context=final_context,
                answer_sources=self._extract_sources(final_context),
                
                # Performance metrics
                total_time_ms=total_time,
                embedding_time_ms=embedding_time,
                search_time_ms=search_time,
                reranking_time_ms=reranking_time,
                context_time_ms=context_time,
                
                # Intermediate results
                query_embedding=query_embedding,
                maxsim_results=maxsim_results,
                cascade_results=cascade_results,
                
                # Metadata
                total_candidates=len(maxsim_results) if maxsim_results else 0,
                final_results=len(cascade_results) if cascade_results else 0
            )
            
            logger.info(f"Pipeline completed: {query[:50]}... in {total_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed for query '{query[:50]}...': {e}")
            # Return empty result on failure
            return RetrievalResult(
                query=query,
                final_context=RepackedContext(
                    text="", sources=[], total_tokens=0, 
                    context_windows=[], relevance_scores=[]
                ),
                answer_sources=[],
                total_time_ms=(time.time() - start_time) * 1000,
                embedding_time_ms=0,
                search_time_ms=0,
                reranking_time_ms=0,
                context_time_ms=0
            )
    
    async def _embed_query(self, query: str) -> QueryEmbedding:
        """Embed query with caching"""
        if self._embedding_cache and query in self._embedding_cache:
            return self._embedding_cache[query]
        
        query_embedding = await self.embedder.embed_query(query)
        
        if self._embedding_cache:
            self._embedding_cache[query] = query_embedding
        
        return query_embedding
    
    async def _maxsim_search(
        self, 
        query_embedding: QueryEmbedding, 
        filters: Optional[Dict[str, Any]]
    ) -> List[MaxSimResult]:
        """Perform MaxSim search with filters"""
        search_config = self.config.search_config or SearchConfig()
        
        # Apply filters to search config if provided
        if filters:
            if "document_ids" in filters:
                # Would need to modify SearchConfig to support document filtering
                pass
            if "chunk_type" in filters:
                search_config.chunk_type = filters["chunk_type"]
        
        maxsim_results = await self.searcher.search(query_embedding, search_config)
        
        logger.info(f"MaxSim search returned {len(maxsim_results)} results")
        return maxsim_results
    
    async def _cascade_rerank(
        self, 
        query: str, 
        maxsim_results: List[MaxSimResult]
    ) -> List[CascadeResult]:
        """Perform three-stage cascade reranking"""
        if not maxsim_results:
            return []
        
        cascade_results = await self.reranker.rerank(query, maxsim_results)
        
        logger.info(f"Cascade reranking: {len(maxsim_results)} → {len(cascade_results)} results")
        return cascade_results
    
    async def _retrieve_and_repack_context(
        self, 
        cascade_results: List[CascadeResult]
    ) -> RepackedContext:
        """Retrieve parent contexts and repack for LLM"""
        if not cascade_results:
            return RepackedContext(
                text="", sources=[], total_tokens=0, 
                context_windows=[], relevance_scores=[]
            )
        
        # Retrieve parent contexts
        context_config = self.config.context_config or ContextConfig()
        parent_contexts = await self.parent_retriever.retrieve_parent_contexts(
            cascade_results, context_config
        )
        
        # Reverse repack contexts
        final_context = self.repacker.repack_contexts(parent_contexts, context_config)
        
        logger.info(f"Context repacking: {len(parent_contexts)} parents → {final_context.total_tokens} tokens")
        return final_context
    
    def _extract_sources(self, context: RepackedContext) -> List[Dict[str, Any]]:
        """Extract source information for answer attribution"""
        sources = []
        
        for i, source in enumerate(context.sources):
            sources.append({
                "source_id": i + 1,
                "document_id": source["document_id"],
                "section_title": source.get("section_title", f"Section {i+1}"),
                "relevance_score": source["relevance_score"],
                "token_count": source["token_count"],
                "excerpt": context.text[
                    context.context_windows[i]["start_pos"]:
                    context.context_windows[i]["start_pos"] + 200
                ] + "..." if i < len(context.context_windows) else ""
            })
        
        return sources
    
    # Additional utility methods
    
    async def explain_retrieval(self, query: str) -> Dict[str, Any]:
        """Detailed explanation of retrieval process for debugging"""
        result = await self.retrieve(query)
        
        explanation = {
            "query": query,
            "pipeline_stages": {
                "embedding": {
                    "time_ms": result.embedding_time_ms,
                    "token_count": result.query_embedding.token_count if result.query_embedding else 0,
                    "model": self.config.embedding_model
                },
                "maxsim_search": {
                    "time_ms": result.search_time_ms,
                    "candidates_found": result.total_candidates,
                    "search_method": "MaxSim with late interaction"
                },
                "cascade_reranking": {
                    "time_ms": result.reranking_time_ms,
                    "final_results": result.final_results,
                    "stages": ["TILDE", "MonoT5", "RankLLaMA"]
                },
                "context_processing": {
                    "time_ms": result.context_time_ms,
                    "total_tokens": result.final_context.total_tokens,
                    "num_sources": len(result.final_context.sources)
                }
            },
            "performance": {
                "total_time_ms": result.total_time_ms,
                "tokens_per_second": result.final_context.total_tokens / (result.total_time_ms / 1000) if result.total_time_ms > 0 else 0
            },
            "final_context_summary": self.repacker.get_context_summary(result.final_context)
        }
        
        return explanation
    
    async def batch_retrieve(
        self, 
        queries: List[str], 
        max_concurrent: int = 3
    ) -> List[RetrievalResult]:
        """Process multiple queries in parallel"""
        if not self.config.parallel_processing:
            # Sequential processing
            results = []
            for query in queries:
                result = await self.retrieve(query)
                results.append(result)
            return results
        
        # Parallel processing with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_query(query: str) -> RetrievalResult:
            async with semaphore:
                return await self.retrieve(query)
        
        tasks = [process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i} failed: {result}")
                # Create empty result
                final_results.append(RetrievalResult(
                    query=queries[i],
                    final_context=RepackedContext(
                        text="", sources=[], total_tokens=0, 
                        context_windows=[], relevance_scores=[]
                    ),
                    answer_sources=[],
                    total_time_ms=0,
                    embedding_time_ms=0,
                    search_time_ms=0,
                    reranking_time_ms=0,
                    context_time_ms=0
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline configuration and statistics"""
        return {
            "config": {
                "embedding_model": self.config.embedding_model,
                "colbert_dim": self.config.colbert_dim,
                "caching_enabled": self.config.enable_caching,
                "parallel_processing": self.config.parallel_processing
            },
            "component_info": {
                "embedder": self.embedder.get_model_info(),
                "search": {"method": "MaxSim with late interaction"},
                "reranking": {"stages": 3, "models": ["TILDE", "MonoT5", "RankLLaMA"]},
                "context": {"parent_retrieval": True, "reverse_repacking": True}
            },
            "cache_stats": {
                "embedding_cache_size": len(self._embedding_cache) if self._embedding_cache else 0
            }
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components"""
        health = {}
        
        try:
            health["opensearch"] = await self.opensearch.health_check()
        except Exception:
            health["opensearch"] = False
        
        try:
            health["database"] = await self.db.health_check()
        except Exception:
            health["database"] = False
        
        try:
            # Test embedding
            test_result = await self.embedder.embed_query("test query")
            health["embedder"] = len(test_result.token_embeddings) > 0
        except Exception:
            health["embedder"] = False
        
        health["overall"] = all(health.values())
        return health
