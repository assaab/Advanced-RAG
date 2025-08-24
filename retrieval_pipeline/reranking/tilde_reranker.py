"""
TILDEv2 Sparse Reranker
Ultra-fast sparse reranking for initial filtering (1000 → 100)
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class TildeConfig:
    """Configuration for TILDE reranker"""
    model_name: str = "ielab/TILDE"  # Or use a lightweight alternative
    max_query_length: int = 128
    max_doc_length: int = 512
    batch_size: int = 32
    device: Optional[str] = None
    cache_size: int = 10000


@dataclass
class SparseVector:
    """Sparse vector representation"""
    indices: List[int]
    values: List[float]
    vocab_size: int
    
    def to_dense(self) -> List[float]:
        """Convert to dense vector"""
        dense = [0.0] * self.vocab_size
        for idx, val in zip(self.indices, self.values):
            if idx < self.vocab_size:
                dense[idx] = val
        return dense


@dataclass
class TildeResult:
    """TILDE reranking result"""
    chunk_id: str
    document_id: str
    tilde_score: float
    sparse_scores: Dict[str, float]  # term -> score
    original_score: float
    text: str
    metadata: Dict[str, Any]


class TildeReranker:
    """
    TILDEv2 Sparse Reranker - Ultra-fast sparse term-based reranking
    
    TILDE (Term Independent Likelihood moDEl) creates sparse representations
    that can be scored very quickly without neural network inference.
    
    Process:
    1. Convert query to sparse term weights (once per query)
    2. Score documents using pre-computed sparse representations
    3. Fast dot product scoring without neural networks
    """
    
    def __init__(self, config: Optional[TildeConfig] = None):
        self.config = config or TildeConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.vocab_size = 0
        
        # Caches for performance
        self.query_cache = {}  # query -> sparse vector
        self.doc_cache = {}    # doc_id -> sparse vector
        
        # Initialize model
        self._load_model()
        
        logger.info(f"TILDE reranker initialized on {self.device}")
    
    def _load_model(self):
        """Load TILDE model or create lightweight alternative"""
        try:
            # Try to load actual TILDE model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            self.vocab_size = len(self.tokenizer.vocab)
            
            logger.info(f"Loaded TILDE model: {self.config.model_name}")
            
        except Exception as e:
            logger.warning(f"Could not load TILDE model: {e}")
            # Fallback to lightweight implementation
            self._create_lightweight_tilde()
    
    def _create_lightweight_tilde(self):
        """Create lightweight TILDE-like implementation"""
        from transformers import AutoTokenizer, AutoModel
        
        # Use a standard model as base
        model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        self.vocab_size = len(self.tokenizer.vocab)
        
        logger.info(f"Created lightweight TILDE implementation with {model_name}")
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 100
    ) -> List[TildeResult]:
        """
        Perform TILDE sparse reranking
        
        Args:
            query: Search query
            candidates: List of candidate documents from MaxSim search
            top_k: Number of results to return
        
        Returns:
            Reranked results with TILDE scores
        """
        if not candidates:
            return []
        
        logger.info(f"TILDE reranking {len(candidates)} candidates → top {top_k}")
        
        # Convert query to sparse representation
        query_sparse = await self._get_query_sparse_vector(query)
        
        # Score all candidates
        results = []
        
        # Process in batches for memory efficiency
        batch_size = self.config.batch_size
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_results = await self._score_batch(query_sparse, batch)
            results.extend(batch_results)
        
        # Sort by TILDE score
        results.sort(key=lambda x: x.tilde_score, reverse=True)
        
        logger.info(f"TILDE reranking completed: {len(candidates)} → {min(top_k, len(results))}")
        return results[:top_k]
    
    async def _get_query_sparse_vector(self, query: str) -> SparseVector:
        """Convert query to sparse vector representation"""
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Tokenize query
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=self.config.max_query_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Create sparse representation
            # In a full TILDE implementation, this would use specialized layers
            # Here we approximate with attention-weighted token importance
            last_hidden = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"].squeeze(0)
            
            # Simple importance scoring (in practice, TILDE uses learned weights)
            token_importance = torch.norm(last_hidden, dim=1) * attention_mask.float()
            
            # Get token ids and create sparse vector
            token_ids = inputs["input_ids"].squeeze(0)
            
            # Create sparse representation
            indices = []
            values = []
            
            for i, (token_id, importance) in enumerate(zip(token_ids, token_importance)):
                if importance > 0 and token_id not in [self.tokenizer.pad_token_id, 
                                                      self.tokenizer.cls_token_id, 
                                                      self.tokenizer.sep_token_id]:
                    indices.append(token_id.item())
                    values.append(importance.item())
        
        # Normalize values
        if values:
            max_val = max(values)
            values = [v / max_val for v in values]
        
        sparse_vector = SparseVector(
            indices=indices,
            values=values,
            vocab_size=self.vocab_size
        )
        
        # Cache result
        if len(self.query_cache) < self.config.cache_size:
            self.query_cache[query] = sparse_vector
        
        return sparse_vector
    
    async def _score_batch(
        self,
        query_sparse: SparseVector,
        batch: List[Dict[str, Any]]
    ) -> List[TildeResult]:
        """Score a batch of candidates against query sparse vector"""
        results = []
        
        for candidate in batch:
            # Get document sparse vector
            doc_sparse = await self._get_document_sparse_vector(
                candidate["text"], 
                candidate["chunk_id"]
            )
            
            # Compute sparse dot product score
            tilde_score = self._compute_sparse_score(query_sparse, doc_sparse)
            
            # Extract term-level scores for analysis
            sparse_scores = self._get_term_scores(query_sparse, doc_sparse)
            
            result = TildeResult(
                chunk_id=candidate["chunk_id"],
                document_id=candidate["document_id"],
                tilde_score=tilde_score,
                sparse_scores=sparse_scores,
                original_score=candidate.get("maxsim_score", 0.0),
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            
            results.append(result)
        
        return results
    
    async def _get_document_sparse_vector(self, text: str, doc_id: str) -> SparseVector:
        """Convert document to sparse vector representation"""
        if doc_id in self.doc_cache:
            return self.doc_cache[doc_id]
        
        # Tokenize document
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_doc_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Create sparse representation (similar to query)
            last_hidden = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"].squeeze(0)
            
            # Token importance for documents (can use different weighting)
            token_importance = torch.norm(last_hidden, dim=1) * attention_mask.float()
            
            # Apply TF-IDF like weighting (simplified)
            token_ids = inputs["input_ids"].squeeze(0)
            token_counts = defaultdict(int)
            
            # Count token frequencies
            for token_id in token_ids:
                if token_id not in [self.tokenizer.pad_token_id, 
                                  self.tokenizer.cls_token_id, 
                                  self.tokenizer.sep_token_id]:
                    token_counts[token_id.item()] += 1
            
            # Create sparse representation with TF weighting
            indices = []
            values = []
            doc_length = len([t for t in token_ids if t != self.tokenizer.pad_token_id])
            
            for i, (token_id, importance) in enumerate(zip(token_ids, token_importance)):
                if importance > 0 and token_id not in [self.tokenizer.pad_token_id, 
                                                      self.tokenizer.cls_token_id, 
                                                      self.tokenizer.sep_token_id]:
                    token_id_int = token_id.item()
                    
                    # Apply TF weighting
                    tf = token_counts[token_id_int] / doc_length
                    weighted_importance = importance.item() * (1 + math.log(1 + tf))
                    
                    indices.append(token_id_int)
                    values.append(weighted_importance)
        
        # Normalize values
        if values:
            max_val = max(values)
            values = [v / max_val for v in values]
        
        sparse_vector = SparseVector(
            indices=indices,
            values=values,
            vocab_size=self.vocab_size
        )
        
        # Cache result
        if len(self.doc_cache) < self.config.cache_size:
            self.doc_cache[doc_id] = sparse_vector
        
        return sparse_vector
    
    def _compute_sparse_score(self, query_sparse: SparseVector, doc_sparse: SparseVector) -> float:
        """Compute sparse dot product score between query and document"""
        if not query_sparse.indices or not doc_sparse.indices:
            return 0.0
        
        # Create lookup for document terms
        doc_terms = dict(zip(doc_sparse.indices, doc_sparse.values))
        
        # Compute dot product
        score = 0.0
        for q_idx, q_val in zip(query_sparse.indices, query_sparse.values):
            if q_idx in doc_terms:
                score += q_val * doc_terms[q_idx]
        
        return score
    
    def _get_term_scores(self, query_sparse: SparseVector, doc_sparse: SparseVector) -> Dict[str, float]:
        """Get individual term contribution scores"""
        term_scores = {}
        
        # Create lookup for document terms
        doc_terms = dict(zip(doc_sparse.indices, doc_sparse.values))
        
        # Get term scores
        for q_idx, q_val in zip(query_sparse.indices, query_sparse.values):
            if q_idx in doc_terms:
                token = self.tokenizer.decode([q_idx])
                term_scores[token] = q_val * doc_terms[q_idx]
        
        return term_scores
    
    def precompute_document_vectors(self, documents: List[Dict[str, Any]]) -> Dict[str, SparseVector]:
        """
        Precompute sparse vectors for documents (for indexing)
        This would typically be done offline during document processing
        """
        doc_vectors = {}
        
        logger.info(f"Precomputing sparse vectors for {len(documents)} documents")
        
        for doc in documents:
            doc_id = doc["chunk_id"]
            text = doc["text"]
            
            # This would normally be done in batches
            sparse_vector = asyncio.run(self._get_document_sparse_vector(text, doc_id))
            doc_vectors[doc_id] = sparse_vector
        
        logger.info(f"Precomputed {len(doc_vectors)} sparse vectors")
        return doc_vectors
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.config.model_name,
            "vocab_size": self.vocab_size,
            "device": self.device,
            "cache_sizes": {
                "query_cache": len(self.query_cache),
                "doc_cache": len(self.doc_cache)
            }
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.query_cache.clear()
        self.doc_cache.clear()
        logger.info("TILDE caches cleared")


class FastTildeReranker:
    """
    Simplified, ultra-fast TILDE-like implementation
    Uses pre-computed term weights for maximum speed
    """
    
    def __init__(self, term_weights: Optional[Dict[str, float]] = None):
        self.term_weights = term_weights or self._create_default_weights()
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer for term extraction"""
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def _create_default_weights(self) -> Dict[str, float]:
        """Create default term importance weights"""
        # In practice, these would be learned from training data
        return {
            # High importance terms
            "what": 2.0, "how": 2.0, "why": 2.0, "when": 2.0, "where": 2.0, "which": 1.8,
            # Medium importance terms
            "is": 1.0, "are": 1.0, "was": 1.0, "were": 1.0, "can": 1.5, "could": 1.5,
            # Low importance (common) terms
            "the": 0.1, "a": 0.1, "an": 0.1, "and": 0.2, "or": 0.3, "but": 0.4,
            # Default for unknown terms
            "__DEFAULT__": 1.0
        }
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 100
    ) -> List[TildeResult]:
        """Ultra-fast sparse reranking"""
        if not candidates:
            return []
        
        # Tokenize query
        query_terms = self._extract_terms(query)
        query_weights = {term: self.term_weights.get(term.lower(), self.term_weights["__DEFAULT__"]) 
                        for term in query_terms}
        
        results = []
        
        for candidate in candidates:
            # Extract document terms
            doc_terms = self._extract_terms(candidate["text"])
            doc_term_counts = {term: doc_terms.count(term) for term in set(doc_terms)}
            
            # Compute sparse score
            score = 0.0
            term_scores = {}
            
            for term, weight in query_weights.items():
                if term in doc_term_counts:
                    term_score = weight * math.log(1 + doc_term_counts[term])
                    score += term_score
                    term_scores[term] = term_score
            
            result = TildeResult(
                chunk_id=candidate["chunk_id"],
                document_id=candidate["document_id"],
                tilde_score=score,
                sparse_scores=term_scores,
                original_score=candidate.get("maxsim_score", 0.0),
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            results.append(result)
        
        # Sort and return top-k
        results.sort(key=lambda x: x.tilde_score, reverse=True)
        return results[:top_k]
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from text"""
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text.lower())
            return [token.replace("##", "") for token in tokens if len(token) > 1]
        else:
            # Fallback to simple tokenization
            import re
            return re.findall(r'\b\w+\b', text.lower())
