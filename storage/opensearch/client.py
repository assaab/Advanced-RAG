"""
OpenSearch Client
Handles vector storage and similarity search
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError
import os
import json
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result container"""
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


@dataclass
class VectorDocument:
    """Vector document for indexing"""
    chunk_id: str
    document_id: str
    parent_chunk_id: Optional[str]
    text: str
    embeddings: List[float]  # Multi-vector embeddings will be stored as multiple docs
    chunk_type: str
    metadata: Dict[str, Any]


class OpenSearchClient:
    """Async OpenSearch client for vector operations"""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or os.getenv("OPENSEARCH_HOST", "localhost")
        self.port = port or int(os.getenv("OPENSEARCH_PORT", "9200"))
        self.username = os.getenv("OPENSEARCH_USERNAME", "admin")
        self.password = os.getenv("OPENSEARCH_PASSWORD", "admin")
        
        self.client = AsyncOpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_auth=(self.username, self.password),
            use_ssl=False,
            verify_certs=False,
            connection_class=None,
        )
        
        # Index names
        self.embedding_index = "document_embeddings"
        self.chunk_index = "document_chunks"
    
    async def create_indices(self):
        """Create OpenSearch indices with proper mappings"""
        
        # Embedding index for multi-vector storage
        embedding_mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "parent_chunk_id": {"type": "keyword"},
                    "vector_id": {"type": "keyword"},  # For multi-vector support
                    "chunk_type": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 768,  # Default BERT-like dimension
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
                        }
                    },
                    "metadata": {"type": "object"}
                }
            },
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512
                }
            }
        }
        
        # Chunk index for metadata and text search
        chunk_mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "parent_chunk_id": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard"
                    },
                    "section_title": {"type": "text"},
                    "token_count": {"type": "integer"},
                    "position": {"type": "integer"},
                    "metadata": {"type": "object"}
                }
            }
        }
        
        # Create indices
        try:
            await self.client.indices.create(
                index=self.embedding_index,
                body=embedding_mapping
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
        
        try:
            await self.client.indices.create(
                index=self.chunk_index,
                body=chunk_mapping
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
    
    async def index_embeddings(self, vector_docs: List[VectorDocument]):
        """Index multi-vector embeddings"""
        if not vector_docs:
            return
        
        bulk_body = []
        
        for doc in vector_docs:
            # For multi-vector embeddings, we'll create multiple documents
            # Each token embedding becomes a separate document
            if isinstance(doc.embeddings[0], list):  # Multi-vector case (ColBERT style)
                for i, embedding in enumerate(doc.embeddings):
                    bulk_body.extend([
                        {"index": {"_index": self.embedding_index, "_id": f"{doc.chunk_id}_vec_{i}"}},
                        {
                            "chunk_id": doc.chunk_id,
                            "document_id": doc.document_id,
                            "parent_chunk_id": doc.parent_chunk_id,
                            "vector_id": f"{doc.chunk_id}_vec_{i}",
                            "chunk_type": doc.chunk_type,
                            "text": doc.text,
                            "embedding": embedding,
                            "metadata": {**doc.metadata, "vector_index": i}
                        }
                    ])
            else:  # Single vector case
                bulk_body.extend([
                    {"index": {"_index": self.embedding_index, "_id": f"{doc.chunk_id}_vec_0"}},
                    {
                        "chunk_id": doc.chunk_id,
                        "document_id": doc.document_id,
                        "parent_chunk_id": doc.parent_chunk_id,
                        "vector_id": f"{doc.chunk_id}_vec_0",
                        "chunk_type": doc.chunk_type,
                        "text": doc.text,
                        "embedding": doc.embeddings,
                        "metadata": doc.metadata
                    }
                ])
        
        if bulk_body:
            await self.client.bulk(body=bulk_body)
    
    async def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunk metadata for text search"""
        if not chunks:
            return
        
        bulk_body = []
        for chunk in chunks:
            bulk_body.extend([
                {"index": {"_index": self.chunk_index, "_id": chunk["chunk_id"]}},
                chunk
            ])
        
        if bulk_body:
            await self.client.bulk(body=bulk_body)
    
    async def vector_search(
        self,
        query_embedding: List[float],
        k: int = 100,
        chunk_type: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform vector similarity search"""
        
        # Build query
        knn_query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            }
        }
        
        # Add filters
        filters = []
        if chunk_type:
            filters.append({"term": {"chunk_type": chunk_type}})
        if document_ids:
            filters.append({"terms": {"document_id": document_ids}})
        
        if filters:
            knn_query["query"] = {
                "bool": {
                    "must": [knn_query["query"]],
                    "filter": filters
                }
            }
        
        # Execute search
        response = await self.client.search(
            index=self.embedding_index,
            body=knn_query
        )
        
        # Parse results
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(SearchResult(
                chunk_id=source["chunk_id"],
                document_id=source["document_id"],
                score=hit["_score"],
                text=source["text"],
                metadata=source.get("metadata", {})
            ))
        
        return results
    
    async def maxsim_search(
        self,
        query_embeddings: List[List[float]],  # Multi-vector query
        k: int = 100,
        chunk_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform MaxSim search for multi-vector embeddings (ColBERT style)
        This is a simplified version - full implementation would use custom scoring
        """
        
        # For each query vector, find top matches
        all_results = {}  # chunk_id -> max_score
        
        for query_vec in query_embeddings:
            vec_results = await self.vector_search(
                query_embedding=query_vec,
                k=k * 2,  # Get more to ensure diversity
                chunk_type=chunk_type
            )
            
            # Update max scores per chunk
            for result in vec_results:
                current_score = all_results.get(result.chunk_id, 0)
                if result.score > current_score:
                    all_results[result.chunk_id] = result.score
        
        # Convert back to SearchResult objects (simplified)
        final_results = []
        chunk_scores = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Get full chunk data for top results
        if chunk_scores:
            chunk_ids = [chunk_id for chunk_id, _ in chunk_scores]
            chunk_data = await self._get_chunks_by_ids(chunk_ids)
            
            for chunk_id, score in chunk_scores:
                if chunk_id in chunk_data:
                    chunk = chunk_data[chunk_id]
                    final_results.append(SearchResult(
                        chunk_id=chunk_id,
                        document_id=chunk["document_id"],
                        score=score,
                        text=chunk["text"],
                        metadata=chunk.get("metadata", {})
                    ))
        
        return final_results
    
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 100,
        alpha: float = 0.5  # Weight between text and vector search
    ) -> List[SearchResult]:
        """Perform hybrid text + vector search"""
        
        # Vector search
        vector_results = await self.vector_search(query_embedding, k=k)
        
        # Text search
        text_query = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "section_title"],
                    "type": "best_fields"
                }
            }
        }
        
        text_response = await self.client.search(
            index=self.chunk_index,
            body=text_query
        )
        
        # Combine results with weighted scoring
        combined_scores = {}
        
        # Add vector scores
        for result in vector_results:
            combined_scores[result.chunk_id] = {
                "vector_score": result.score,
                "text_score": 0,
                "result": result
            }
        
        # Add text scores
        for hit in text_response["hits"]["hits"]:
            chunk_id = hit["_id"]
            text_score = hit["_score"]
            
            if chunk_id in combined_scores:
                combined_scores[chunk_id]["text_score"] = text_score
            else:
                source = hit["_source"]
                combined_scores[chunk_id] = {
                    "vector_score": 0,
                    "text_score": text_score,
                    "result": SearchResult(
                        chunk_id=chunk_id,
                        document_id=source["document_id"],
                        score=text_score,
                        text=source["text"],
                        metadata=source.get("metadata", {})
                    )
                }
        
        # Calculate combined scores and sort
        final_results = []
        for chunk_id, scores in combined_scores.items():
            combined_score = (alpha * scores["vector_score"] + (1 - alpha) * scores["text_score"])
            result = scores["result"]
            result.score = combined_score
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]
    
    async def _get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get chunk metadata by IDs"""
        if not chunk_ids:
            return {}
        
        query = {
            "query": {
                "terms": {"chunk_id": chunk_ids}
            }
        }
        
        response = await self.client.search(
            index=self.chunk_index,
            body=query
        )
        
        chunks = {}
        for hit in response["hits"]["hits"]:
            chunks[hit["_source"]["chunk_id"]] = hit["_source"]
        
        return chunks
    
    async def delete_document(self, document_id: str):
        """Delete all embeddings and chunks for a document"""
        delete_query = {"query": {"term": {"document_id": document_id}}}
        
        await self.client.delete_by_query(
            index=self.embedding_index,
            body=delete_query
        )
        
        await self.client.delete_by_query(
            index=self.chunk_index,
            body=delete_query
        )
    
    async def health_check(self) -> bool:
        """Check OpenSearch connectivity"""
        try:
            response = await self.client.cluster.health()
            return response.get("status") in ["green", "yellow"]
        except Exception:
            return False
    
    async def close(self):
        """Close OpenSearch connection"""
        if self.client:
            await self.client.close()
