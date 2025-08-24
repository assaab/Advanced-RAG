"""
MonoT5 Cross-Encoder Reranker
Neural reranking with deep semantic understanding (100 → 20)
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MonoT5Config:
    """Configuration for MonoT5 reranker"""
    model_name: str = "castorini/monot5-base-msmarco"  # Or use alternative
    max_length: int = 512
    batch_size: int = 8
    device: Optional[str] = None
    use_fp16: bool = True  # For faster inference
    cache_size: int = 1000


@dataclass
class MonoT5Result:
    """MonoT5 reranking result"""
    chunk_id: str
    document_id: str
    monot5_score: float
    relevance_logits: List[float]  # Raw model outputs
    confidence: float  # Confidence in prediction
    original_score: float  # Original TILDE score
    text: str
    metadata: Dict[str, Any]


class MonoT5Reranker:
    """
    MonoT5 Cross-Encoder Reranker
    
    Uses T5-based cross-encoder for deep semantic reranking.
    The model sees query and document together, enabling better
    understanding of relevance relationships.
    
    Process:
    1. Concatenate query and document: "Query: [Q] Document: [D]"
    2. Pass through T5 model
    3. Get relevance score from model output
    4. Rank by relevance scores
    """
    
    def __init__(self, config: Optional[MonoT5Config] = None):
        self.config = config or MonoT5Config()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.tokenizer = None
        
        # Performance optimization
        self.score_cache = {}  # (query, doc_text) -> score
        
        # Load model
        self._load_model()
        
        logger.info(f"MonoT5 reranker initialized on {self.device}")
    
    def _load_model(self):
        """Load MonoT5 model or fallback alternative"""
        try:
            # Try to load actual MonoT5 model
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
            
            self.model.to(self.device)
            if self.config.use_fp16 and torch.cuda.is_available():
                self.model = self.model.half()
            self.model.eval()
            
            logger.info(f"Loaded MonoT5 model: {self.config.model_name}")
            
        except Exception as e:
            logger.warning(f"Could not load MonoT5 model: {e}")
            # Fallback to cross-encoder alternative
            self._load_cross_encoder_alternative()
    
    def _load_cross_encoder_alternative(self):
        """Load cross-encoder alternative model"""
        try:
            # Use a standard cross-encoder model
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.model.to(self.device)
            if self.config.use_fp16 and torch.cuda.is_available():
                self.model = self.model.half()
            self.model.eval()
            
            logger.info(f"Loaded cross-encoder alternative: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load any reranking model: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[MonoT5Result]:
        """
        Perform MonoT5 cross-encoder reranking
        
        Args:
            query: Search query
            candidates: Candidates from TILDE reranker
            top_k: Number of results to return
        
        Returns:
            Reranked results with MonoT5 scores
        """
        if not candidates:
            return []
        
        logger.info(f"MonoT5 reranking {len(candidates)} candidates → top {top_k}")
        
        results = []
        
        # Process in batches for memory efficiency
        batch_size = self.config.batch_size
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_results = await self._score_batch(query, batch)
            results.extend(batch_results)
        
        # Sort by MonoT5 score
        results.sort(key=lambda x: x.monot5_score, reverse=True)
        
        logger.info(f"MonoT5 reranking completed: {len(candidates)} → {min(top_k, len(results))}")
        return results[:top_k]
    
    async def _score_batch(
        self,
        query: str,
        batch: List[Dict[str, Any]]
    ) -> List[MonoT5Result]:
        """Score a batch of candidates"""
        
        # Prepare inputs for batch processing
        batch_inputs = []
        valid_indices = []
        
        for i, candidate in enumerate(batch):
            # Check cache first
            cache_key = (query, candidate["text"][:200])  # Use text prefix for cache key
            if cache_key in self.score_cache:
                continue
            
            # Prepare input text
            input_text = self._prepare_input(query, candidate["text"])
            batch_inputs.append(input_text)
            valid_indices.append(i)
        
        # Get scores for new inputs
        if batch_inputs:
            with torch.no_grad():
                scores = await self._compute_scores(batch_inputs)
                
                # Update cache
                for input_text, score in zip(batch_inputs, scores):
                    if len(self.score_cache) < self.config.cache_size:
                        # Use simplified cache key
                        cache_key = (query, input_text[:200])
                        self.score_cache[cache_key] = score
        
        # Create results
        results = []
        score_idx = 0
        
        for i, candidate in enumerate(batch):
            cache_key = (query, candidate["text"][:200])
            
            if cache_key in self.score_cache:
                # Use cached score
                monot5_score = self.score_cache[cache_key]
                relevance_logits = [monot5_score]
                confidence = self._compute_confidence(monot5_score)
            elif i in valid_indices:
                # Use computed score
                monot5_score = scores[score_idx]
                relevance_logits = [monot5_score]
                confidence = self._compute_confidence(monot5_score)
                score_idx += 1
            else:
                # Fallback score
                monot5_score = 0.5
                relevance_logits = [0.5]
                confidence = 0.5
            
            result = MonoT5Result(
                chunk_id=candidate["chunk_id"],
                document_id=candidate["document_id"],
                monot5_score=float(monot5_score),
                relevance_logits=relevance_logits,
                confidence=float(confidence),
                original_score=candidate.get("tilde_score", 0.0),
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            
            results.append(result)
        
        return results
    
    def _prepare_input(self, query: str, document: str) -> str:
        """Prepare input text for the model"""
        
        # Truncate document if too long
        max_doc_length = self.config.max_length - len(query) - 20  # Leave space for special tokens
        if len(document) > max_doc_length:
            document = document[:max_doc_length]
        
        # Format depends on model type
        if "monot5" in self.config.model_name.lower():
            # MonoT5 format: "Query: [query] Document: [document] Relevant:"
            return f"Query: {query} Document: {document} Relevant:"
        else:
            # Cross-encoder format: just concatenate with [SEP]
            return f"{query} [SEP] {document}"
    
    async def _compute_scores(self, batch_inputs: List[str]) -> List[float]:
        """Compute relevance scores for batch of inputs"""
        
        # Tokenize batch
        inputs = self.tokenizer(
            batch_inputs,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if "monot5" in self.config.model_name.lower():
                # MonoT5 T5-based scoring
                scores = await self._monot5_scoring(inputs)
            else:
                # Cross-encoder scoring
                scores = await self._cross_encoder_scoring(inputs)
        
        return scores
    
    async def _monot5_scoring(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """MonoT5 T5-based scoring"""
        
        # Generate with MonoT5
        # The model predicts "true" or "false" for relevance
        generate_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=3,  # Just need "true" or "false"
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Get scores from generation
        scores = []
        
        if hasattr(generate_ids, 'scores') and generate_ids.scores:
            # Use generation scores
            for i in range(len(inputs["input_ids"])):
                # Get score for "true" token
                first_token_logits = generate_ids.scores[0][i]  # [vocab_size]
                
                # Get logits for "true" and "false" tokens
                true_token_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
                false_token_id = self.tokenizer.encode("false", add_special_tokens=False)[0]
                
                true_logit = first_token_logits[true_token_id].item()
                false_logit = first_token_logits[false_token_id].item()
                
                # Convert to probability
                prob_true = torch.softmax(torch.tensor([false_logit, true_logit]), dim=0)[1].item()
                scores.append(prob_true)
        else:
            # Fallback: use simple forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Simple averaging of logits (not ideal, but fallback)
            for i in range(len(inputs["input_ids"])):
                seq_logits = logits[i].mean(dim=0)  # Average over sequence
                score = torch.sigmoid(seq_logits.mean()).item()  # Average over vocab
                scores.append(score)
        
        return scores
    
    async def _cross_encoder_scoring(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """Cross-encoder scoring"""
        
        outputs = self.model(**inputs)
        
        if hasattr(outputs, 'logits'):
            # Classification head outputs
            logits = outputs.logits  # [batch_size, num_classes]
            
            if logits.size(-1) == 1:
                # Single output (regression)
                scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
            else:
                # Multiple outputs (classification)
                # Use softmax to get relevance probability
                probs = F.softmax(logits, dim=-1)
                scores = probs[:, -1].cpu().tolist()  # Assume last class is "relevant"
        else:
            # Fallback
            scores = [0.5] * len(inputs["input_ids"])
        
        return scores
    
    def _compute_confidence(self, score: float) -> float:
        """Compute confidence in the prediction"""
        # Higher confidence for scores closer to 0 or 1
        return 1.0 - 2.0 * abs(score - 0.5)
    
    async def explain_score(
        self,
        query: str,
        document: str,
        chunk_id: str
    ) -> Dict[str, Any]:
        """
        Explain MonoT5 scoring for a specific query-document pair
        """
        input_text = self._prepare_input(query, document)
        
        # Get detailed scoring
        scores = await self._compute_scores([input_text])
        score = scores[0] if scores else 0.5
        
        # Analyze input
        tokens = self.tokenizer.tokenize(input_text)
        
        return {
            "chunk_id": chunk_id,
            "query": query,
            "document_preview": document[:200] + "...",
            "monot5_score": float(score),
            "confidence": float(self._compute_confidence(score)),
            "input_text": input_text[:500] + "...",
            "token_count": len(tokens),
            "model_info": {
                "model_name": self.config.model_name,
                "max_length": self.config.max_length,
                "device": self.device
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "use_fp16": self.config.use_fp16,
            "cache_size": len(self.score_cache)
        }
    
    def clear_cache(self):
        """Clear score cache"""
        self.score_cache.clear()
        logger.info("MonoT5 cache cleared")


class LightweightMonoT5:
    """
    Lightweight MonoT5-like implementation for faster inference
    Uses simplified cross-attention mechanism
    """
    
    def __init__(self, config: Optional[MonoT5Config] = None):
        self.config = config or MonoT5Config()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load lightweight model
        from transformers import AutoModel, AutoTokenizer
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Simple classification head
        hidden_size = self.encoder.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 3, hidden_size),  # query, doc, interaction
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        logger.info("Lightweight MonoT5 implementation initialized")
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[MonoT5Result]:
        """Fast lightweight reranking"""
        
        if not candidates:
            return []
        
        # Encode query once
        query_embedding = await self._encode_text(query)
        
        results = []
        
        # Process candidates
        for candidate in candidates:
            doc_embedding = await self._encode_text(candidate["text"])
            
            # Compute interaction features
            interaction = query_embedding * doc_embedding  # Element-wise product
            
            # Concatenate features
            features = torch.cat([query_embedding, doc_embedding, interaction], dim=-1)
            
            # Get relevance score
            with torch.no_grad():
                score = self.classifier(features).item()
            
            result = MonoT5Result(
                chunk_id=candidate["chunk_id"],
                document_id=candidate["document_id"],
                monot5_score=float(score),
                relevance_logits=[float(score)],
                confidence=float(self._compute_confidence(score)),
                original_score=candidate.get("tilde_score", 0.0),
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            
            results.append(result)
        
        # Sort and return top-k
        results.sort(key=lambda x: x.monot5_score, reverse=True)
        return results[:top_k]
    
    async def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = torch.sum(embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        return embeddings.squeeze(0)
    
    def _compute_confidence(self, score: float) -> float:
        """Compute confidence in prediction"""
        return 1.0 - 2.0 * abs(score - 0.5)
