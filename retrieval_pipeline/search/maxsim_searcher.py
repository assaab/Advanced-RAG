"""
MaxSim Searcher
Late Interaction search using ColBERT-style MaxSim scoring
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from storage.opensearch.client import OpenSearchClient, SearchResult
from retrieval_pipeline.embeddings.multi_vector_embedder import QueryEmbedding

logger = logging.getLogger(__name__)


@dataclass
class MaxSimResult:
    """MaxSim search result with detailed scoring"""
    chunk_id: str
    document_id: str
    maxsim_score: float
    vector_scores: List[float]  # Scores for each query token
    text: str
    metadata: Dict[str, Any]
    parent_chunk_id: Optional[str] = None


@dataclass
class SearchConfig:
    """Search configuration"""
    candidate_pool_size: int = 1000  # Initial candidates
    final_results: int = 100  # Final results after MaxSim
    chunk_type: Optional[str] = "child"  # Search in child chunks
    use_prefilter: bool = True  # Use vector search as prefilter
    alpha: float = 1.0  # MaxSim weighting parameter


class MaxSimSearcher:
    """
    Advanced searcher using MaxSim scoring for late interaction
    Supports both pure MaxSim and hybrid approaches
    """
    
    def __init__(self, opensearch_client: OpenSearchClient):
        self.opensearch = opensearch_client
        self.default_config = SearchConfig()
    
    async def search(
        self,
        query_embedding: QueryEmbedding,
        config: Optional[SearchConfig] = None
    ) -> List[MaxSimResult]:
        """
        Perform MaxSim search with late interaction scoring
        
        Process:
        1. Get candidate pool using vector search (prefilter)
        2. Compute MaxSim scores for all candidates
        3. Rank by MaxSim scores
        4. Return top-k results
        """
        config = config or self.default_config
        
        # Step 1: Get candidate pool
        candidates = await self._get_candidate_pool(query_embedding, config)
        
        if not candidates:
            logger.warning("No candidates found in initial search")
            return []
        
        # Step 2: Compute MaxSim scores
        maxsim_results = await self._compute_maxsim_scores(
            query_embedding, candidates, config
        )
        
        # Step 3: Sort by MaxSim scores and return top-k
        maxsim_results.sort(key=lambda x: x.maxsim_score, reverse=True)
        
        return maxsim_results[:config.final_results]
    
    async def _get_candidate_pool(
        self,
        query_embedding: QueryEmbedding,
        config: SearchConfig
    ) -> List[SearchResult]:
        """
        Get initial candidate pool using vector search
        Uses pooled embedding as approximation for efficiency
        """
        if config.use_prefilter:
            # Use pooled embedding for initial filtering
            candidates = await self.opensearch.vector_search(
                query_embedding=query_embedding.pooled_embedding,
                k=config.candidate_pool_size,
                chunk_type=config.chunk_type
            )
        else:
            # Get all available chunks (expensive, for small corpora only)
            candidates = await self._get_all_chunks(config.chunk_type)
            candidates = candidates[:config.candidate_pool_size]
        
        logger.info(f"Retrieved {len(candidates)} candidates for MaxSim scoring")
        return candidates
    
    async def _compute_maxsim_scores(
        self,
        query_embedding: QueryEmbedding,
        candidates: List[SearchResult],
        config: SearchConfig
    ) -> List[MaxSimResult]:
        """
        Compute MaxSim scores for all candidates
        This is where the late interaction magic happens
        """
        results = []
        
        # Get detailed embeddings for candidates
        candidate_embeddings = await self._get_candidate_embeddings(
            [c.chunk_id for c in candidates]
        )
        
        for candidate in candidates:
            chunk_id = candidate.chunk_id
            
            if chunk_id not in candidate_embeddings:
                logger.warning(f"No embeddings found for chunk {chunk_id}")
                continue
            
            doc_embeddings = candidate_embeddings[chunk_id]["token_embeddings"]
            
            # Compute MaxSim score
            maxsim_score, token_scores = self._maxsim_scoring(
                query_embedding.token_embeddings,
                doc_embeddings,
                config.alpha
            )
            
            # Create result
            result = MaxSimResult(
                chunk_id=chunk_id,
                document_id=candidate.document_id,
                maxsim_score=maxsim_score,
                vector_scores=token_scores,
                text=candidate.text,
                metadata=candidate.metadata,
                parent_chunk_id=candidate_embeddings[chunk_id].get("parent_chunk_id")
            )
            
            results.append(result)
        
        return results
    
    def _maxsim_scoring(
        self,
        query_embeddings: List[List[float]],
        doc_embeddings: List[List[float]],
        alpha: float = 1.0
    ) -> Tuple[float, List[float]]:
        """
        Core MaxSim scoring function
        
        MaxSim(q, d) = (1/|q|) * Σ_i max_j (q_i · d_j)
        
        For each query token, find the most similar document token,
        then average across all query tokens.
        """
        if not query_embeddings or not doc_embeddings:
            return 0.0, []
        
        # Convert to numpy for efficient computation
        query_matrix = np.array(query_embeddings)  # [q_len, dim]
        doc_matrix = np.array(doc_embeddings)      # [d_len, dim]
        
        # Compute similarity matrix [q_len, d_len]
        similarity_matrix = np.dot(query_matrix, doc_matrix.T)
        
        # For each query token, find max similarity with any document token
        max_similarities = np.max(similarity_matrix, axis=1)  # [q_len]
        
        # Apply alpha weighting (for term importance)
        weighted_similarities = alpha * max_similarities
        
        # Average across query tokens
        maxsim_score = np.mean(weighted_similarities).item()
        
        # Return individual token scores for analysis
        token_scores = max_similarities.tolist()
        
        return maxsim_score, token_scores
    
    async def _get_candidate_embeddings(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve multi-vector embeddings for candidate chunks
        This queries the embedding storage to get token-level embeddings
        """
        if not chunk_ids:
            return {}
        
        # Query OpenSearch for embeddings
        # In practice, this would be optimized with batch queries
        embeddings = {}
        
        # Batch query for efficiency
        batch_size = 50
        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_embeddings = await self._fetch_embeddings_batch(batch_ids)
            embeddings.update(batch_embeddings)
        
        return embeddings
    
    async def _fetch_embeddings_batch(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch embeddings for a batch of chunk IDs"""
        try:
            # Query embeddings index
            query = {
                "query": {
                    "terms": {"chunk_id": chunk_ids}
                },
                "size": len(chunk_ids) * 100,  # Account for multi-vector storage
                "_source": ["chunk_id", "parent_chunk_id", "embedding", "vector_id"]
            }
            
            response = await self.opensearch.client.search(
                index=self.opensearch.embedding_index,
                body=query
            )
            
            # Group embeddings by chunk_id
            chunk_embeddings = {}
            
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                chunk_id = source["chunk_id"]
                embedding = source["embedding"]
                
                if chunk_id not in chunk_embeddings:
                    chunk_embeddings[chunk_id] = {
                        "token_embeddings": [],
                        "parent_chunk_id": source.get("parent_chunk_id")
                    }
                
                chunk_embeddings[chunk_id]["token_embeddings"].append(embedding)
            
            return chunk_embeddings
            
        except Exception as e:
            logger.error(f"Failed to fetch embeddings for batch: {e}")
            return {}
    
    async def _get_all_chunks(self, chunk_type: Optional[str]) -> List[SearchResult]:
        """Get all chunks of a specific type (for exhaustive search)"""
        try:
            query = {
                "query": {"match_all": {}},
                "size": 10000  # Adjust based on corpus size
            }
            
            if chunk_type:
                query["query"] = {"term": {"chunk_type": chunk_type}}
            
            response = await self.opensearch.client.search(
                index=self.opensearch.chunk_index,
                body=query
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append(SearchResult(
                    chunk_id=source["chunk_id"],
                    document_id=source["document_id"],
                    score=1.0,  # Placeholder
                    text=source["text"],
                    metadata=source.get("metadata", {})
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get all chunks: {e}")
            return []
    
    async def hybrid_maxsim_search(
        self,
        query_embedding: QueryEmbedding,
        text_query: str,
        config: Optional[SearchConfig] = None,
        text_weight: float = 0.3
    ) -> List[MaxSimResult]:
        """
        Hybrid search combining MaxSim with text search
        Useful for keyword-rich queries
        """
        config = config or self.default_config
        
        # Get MaxSim results
        maxsim_results = await self.search(query_embedding, config)
        
        # Get text search results
        text_results = await self._text_search(text_query, config.final_results * 2)
        
        # Combine scores
        combined_results = self._combine_maxsim_text_scores(
            maxsim_results, text_results, text_weight
        )
        
        return combined_results[:config.final_results]
    
    async def _text_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform text-based search"""
        try:
            query_body = {
                "size": k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "section_title"],
                        "type": "best_fields"
                    }
                }
            }
            
            response = await self.opensearch.client.search(
                index=self.opensearch.chunk_index,
                body=query_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "chunk_id": source["chunk_id"],
                    "text_score": hit["_score"],
                    "text": source["text"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def _combine_maxsim_text_scores(
        self,
        maxsim_results: List[MaxSimResult],
        text_results: List[Dict[str, Any]],
        text_weight: float
    ) -> List[MaxSimResult]:
        """Combine MaxSim and text search scores"""
        
        # Create text score lookup
        text_scores = {r["chunk_id"]: r["text_score"] for r in text_results}
        
        # Normalize MaxSim scores
        if maxsim_results:
            max_maxsim = max(r.maxsim_score for r in maxsim_results)
            min_maxsim = min(r.maxsim_score for r in maxsim_results)
            maxsim_range = max_maxsim - min_maxsim if max_maxsim != min_maxsim else 1.0
        
        # Normalize text scores
        if text_results:
            max_text = max(r["text_score"] for r in text_results)
            min_text = min(r["text_score"] for r in text_results)
            text_range = max_text - min_text if max_text != min_text else 1.0
        
        # Combine scores
        for result in maxsim_results:
            # Normalize MaxSim score
            normalized_maxsim = (result.maxsim_score - min_maxsim) / maxsim_range
            
            # Get normalized text score
            text_score = text_scores.get(result.chunk_id, 0)
            normalized_text = (text_score - min_text) / text_range if text_score > 0 else 0
            
            # Combined score
            result.maxsim_score = (1 - text_weight) * normalized_maxsim + text_weight * normalized_text
        
        # Sort by combined score
        maxsim_results.sort(key=lambda x: x.maxsim_score, reverse=True)
        
        return maxsim_results
    
    async def explain_maxsim_score(self, chunk_id: str, query_embedding: QueryEmbedding) -> Dict[str, Any]:
        """
        Explain MaxSim scoring for debugging and analysis
        Returns detailed breakdown of scoring
        """
        # Get chunk embeddings
        chunk_embeddings = await self._get_candidate_embeddings([chunk_id])
        
        if chunk_id not in chunk_embeddings:
            return {"error": "Chunk embeddings not found"}
        
        doc_embeddings = chunk_embeddings[chunk_id]["token_embeddings"]
        
        # Compute detailed scoring
        query_matrix = np.array(query_embedding.token_embeddings)
        doc_matrix = np.array(doc_embeddings)
        
        similarity_matrix = np.dot(query_matrix, doc_matrix.T)
        max_similarities = np.max(similarity_matrix, axis=1)
        max_indices = np.argmax(similarity_matrix, axis=1)
        
        return {
            "chunk_id": chunk_id,
            "maxsim_score": float(np.mean(max_similarities)),
            "query_tokens": len(query_embedding.token_embeddings),
            "doc_tokens": len(doc_embeddings),
            "token_scores": max_similarities.tolist(),
            "best_matches": max_indices.tolist(),
            "score_distribution": {
                "min": float(np.min(max_similarities)),
                "max": float(np.max(max_similarities)),
                "mean": float(np.mean(max_similarities)),
                "std": float(np.std(max_similarities))
            }
        }
