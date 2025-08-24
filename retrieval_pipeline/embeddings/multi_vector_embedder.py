"""
Multi-Vector Embedder
ColBERT/ColPali style token-level embeddings for late interaction
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for multi-vector embedding results"""
    chunk_id: str
    text: str
    token_embeddings: List[List[float]]  # List of token vectors
    pooled_embedding: List[float]  # Optional pooled representation
    token_count: int
    model_name: str


@dataclass
class QueryEmbedding:
    """Container for query embeddings"""
    query: str
    token_embeddings: List[List[float]]
    pooled_embedding: List[float]
    token_count: int


class MultiVectorEmbedder:
    """
    Multi-vector embedder supporting ColBERT-style late interaction
    Creates token-level embeddings for fine-grained matching
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
        device: Optional[str] = None,
        colbert_dim: int = 128  # ColBERT compression dimension
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.colbert_dim = colbert_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Special tokens
        self.query_marker = "[Q]"
        self.document_marker = "[D]"
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Add special tokens if needed
            special_tokens = [self.query_marker, self.document_marker]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded model {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    async def embed_query(self, query: str) -> QueryEmbedding:
        """
        Create multi-vector embeddings for a query
        Adds query marker for specialized processing
        """
        # Prepare query with marker
        marked_query = f"{self.query_marker} {query.strip()}"
        
        # Get token embeddings
        token_embeddings, pooled_embedding = await self._encode_text(marked_query)
        
        return QueryEmbedding(
            query=query,
            token_embeddings=token_embeddings,
            pooled_embedding=pooled_embedding,
            token_count=len(token_embeddings)
        )
    
    async def embed_documents(self, texts: List[str], chunk_ids: List[str]) -> List[EmbeddingResult]:
        """
        Create multi-vector embeddings for documents
        Adds document marker for specialized processing
        """
        if len(texts) != len(chunk_ids):
            raise ValueError("Number of texts and chunk_ids must match")
        
        results = []
        
        # Process in batches to manage memory
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = chunk_ids[i:i + batch_size]
            
            # Add document markers
            marked_texts = [f"{self.document_marker} {text.strip()}" for text in batch_texts]
            
            # Get embeddings for batch
            batch_results = await self._encode_batch(marked_texts, batch_ids)
            results.extend(batch_results)
        
        return results
    
    async def _encode_text(self, text: str) -> Tuple[List[List[float]], List[float]]:
        """Encode single text into token embeddings and pooled embedding"""
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Get token embeddings (last hidden state)
            token_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
            
            # Apply ColBERT-style processing
            if self.colbert_dim < token_embeddings.size(-1):
                # Linear compression to ColBERT dimension
                # In a full implementation, this would be a learned linear layer
                token_embeddings = token_embeddings[:, :self.colbert_dim]
            
            # Normalize token embeddings for cosine similarity
            token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)
            
            # Create pooled embedding (mean pooling, excluding padding)
            attention_mask = inputs["attention_mask"].squeeze(0)
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            pooled_embedding = torch.sum(token_embeddings * mask_expanded, 0) / torch.clamp(mask_expanded.sum(0), min=1e-9)
            pooled_embedding = F.normalize(pooled_embedding, p=2, dim=0)
            
            # Convert to lists
            token_embeddings_list = token_embeddings.cpu().numpy().tolist()
            pooled_embedding_list = pooled_embedding.cpu().numpy().tolist()
            
            # Filter out padding tokens
            actual_length = attention_mask.sum().item()
            token_embeddings_list = token_embeddings_list[:actual_length]
            
            return token_embeddings_list, pooled_embedding_list
    
    async def _encode_batch(self, texts: List[str], chunk_ids: List[str]) -> List[EmbeddingResult]:
        """Encode batch of texts"""
        results = []
        
        for text, chunk_id in zip(texts, chunk_ids):
            try:
                token_embeddings, pooled_embedding = await self._encode_text(text)
                
                result = EmbeddingResult(
                    chunk_id=chunk_id,
                    text=text,
                    token_embeddings=token_embeddings,
                    pooled_embedding=pooled_embedding,
                    token_count=len(token_embeddings),
                    model_name=self.model_name
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to embed chunk {chunk_id}: {e}")
                # Create empty result to maintain order
                results.append(EmbeddingResult(
                    chunk_id=chunk_id,
                    text=text,
                    token_embeddings=[],
                    pooled_embedding=[0.0] * (self.colbert_dim or 768),
                    token_count=0,
                    model_name=self.model_name
                ))
        
        return results
    
    def compute_maxsim_score(self, query_embeddings: List[List[float]], doc_embeddings: List[List[float]]) -> float:
        """
        Compute MaxSim score between query and document token embeddings
        MaxSim: For each query token, find max similarity with any document token
        """
        if not query_embeddings or not doc_embeddings:
            return 0.0
        
        # Convert to tensors
        query_tensor = torch.tensor(query_embeddings, dtype=torch.float32)  # [q_len, dim]
        doc_tensor = torch.tensor(doc_embeddings, dtype=torch.float32)      # [d_len, dim]
        
        # Compute similarity matrix [q_len, d_len]
        similarity_matrix = torch.mm(query_tensor, doc_tensor.t())
        
        # MaxSim: max similarity for each query token, then average
        max_similarities = torch.max(similarity_matrix, dim=1)[0]  # [q_len]
        maxsim_score = torch.mean(max_similarities).item()
        
        return maxsim_score
    
    def compute_colbert_score(self, query_embeddings: List[List[float]], doc_embeddings: List[List[float]]) -> float:
        """
        Compute ColBERT-style score (MaxSim + optional additional processing)
        This is the core late interaction scoring mechanism
        """
        # For now, ColBERT score is the same as MaxSim
        # In a full implementation, you might add query term weighting, etc.
        return self.compute_maxsim_score(query_embeddings, doc_embeddings)
    
    async def embed_and_index_chunks(self, chunks: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """
        Embed chunks for indexing
        Input: List of chunk dictionaries with 'id', 'text' keys
        """
        if not chunks:
            return []
        
        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]
        
        return await self.embed_documents(texts, chunk_ids)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "colbert_dim": self.colbert_dim,
            "device": self.device,
            "embedding_dim": self.colbert_dim or 768
        }
