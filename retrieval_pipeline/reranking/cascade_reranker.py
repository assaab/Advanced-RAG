"""
Three-Stage Cascade Reranker
TILDE → MonoT5 → RankLLaMA for progressive refinement
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from dataclasses import dataclass
import logging
import time

from retrieval_pipeline.search.maxsim_searcher import MaxSimResult
from retrieval_pipeline.reranking.llm_reranker import LLMReranker, LLMRerankConfig, SimpleLLMReranker

logger = logging.getLogger(__name__)


@dataclass
class RerankingStage:
    """Single reranking stage result"""
    stage_name: str
    input_count: int
    output_count: int
    processing_time_ms: float
    scores: List[float]


@dataclass
class CascadeResult:
    """Final cascade reranking result"""
    chunk_id: str
    document_id: str
    final_score: float
    text: str
    metadata: Dict[str, Any]
    parent_chunk_id: Optional[str]
    
    # Detailed scoring breakdown
    maxsim_score: float
    tilde_score: float
    monot5_score: float
    rankllama_score: float
    
    # Stage information
    stages: List[RerankingStage]


@dataclass
class CascadeConfig:
    """Cascade reranking configuration"""
    # Stage 1: TILDE (fast sparse filtering)
    tilde_candidates: int = 100
    tilde_threshold: float = 0.1
    
    # Stage 2: MonoT5 (deep semantic reranking)
    monot5_candidates: int = 20
    monot5_threshold: float = 0.5
    
    # Stage 3: RankLLaMA (LLM-based reasoning)
    rankllama_candidates: int = 5
    
    # Model settings
    tilde_model: str = "ielab/TILDE"
    monot5_model: str = "castorini/monot5-base-msmarco"
    rankllama_model: str = "microsoft/RankZephyr-7B-beta"  # HuggingFace model
    
    # LLM Reranker settings
    llm_model_type: str = "huggingface"  # "huggingface", "openai", "simple"
    llm_temperature: float = 0.1
    llm_max_input_length: int = 4096
    use_llm_reasoning: bool = True
    openai_model: str = "gpt-3.5-turbo"  # For OpenAI API
    
    # Performance settings
    batch_size: int = 8
    use_gpu: bool = True


class TILDEReranker:
    """TILDE sparse reranker for fast filtering"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        # TILDE is complex to implement fully - using approximation with sparse scoring
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load TILDE model (simplified implementation)"""
        try:
            # In practice, TILDE requires specialized implementation
            # Here we use a standard model as approximation
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load TILDE model: {e}")
            raise
    
    async def rerank(self, query: str, candidates: List[MaxSimResult], top_k: int) -> List[MaxSimResult]:
        """Fast sparse reranking using TILDE-style scoring"""
        start_time = time.time()
        
        if len(candidates) <= top_k:
            return candidates
        
        # Compute sparse scores (simplified TILDE approximation)
        scored_candidates = []
        
        for candidate in candidates:
            # Simple term matching + embedding similarity
            sparse_score = self._compute_sparse_score(query, candidate.text)
            
            # Combine with existing MaxSim score
            combined_score = 0.7 * candidate.maxsim_score + 0.3 * sparse_score
            
            candidate.maxsim_score = combined_score  # Update score
            scored_candidates.append(candidate)
        
        # Sort by combined score and take top-k
        scored_candidates.sort(key=lambda x: x.maxsim_score, reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"TILDE reranking: {len(candidates)} → {top_k} in {processing_time:.2f}ms")
        
        return scored_candidates[:top_k]
    
    def _compute_sparse_score(self, query: str, text: str) -> float:
        """Compute sparse similarity score (TILDE approximation)"""
        # Simple term overlap with importance weighting
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Jaccard similarity with term frequency weighting
        intersection = query_terms.intersection(text_terms)
        union = query_terms.union(text_terms)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Boost score for exact phrase matches
        phrase_bonus = 0.0
        if query.lower() in text.lower():
            phrase_bonus = 0.3
        
        return jaccard + phrase_bonus


class MonoT5Reranker:
    """MonoT5 cross-encoder for deep semantic reranking"""
    
    def __init__(self, model_name: str = "castorini/monot5-base-msmarco", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load MonoT5 model"""
        try:
            # Note: MonoT5 requires special setup - this is simplified
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"MonoT5 model load failed, using fallback: {e}")
            # Fallback to standard transformer
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model.to(self.device)
            self.model.eval()
    
    async def rerank(self, query: str, candidates: List[MaxSimResult], top_k: int) -> List[MaxSimResult]:
        """Deep semantic reranking with MonoT5"""
        start_time = time.time()
        
        if len(candidates) <= top_k:
            return candidates
        
        # Process in batches for efficiency
        batch_size = 4
        scored_candidates = []
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_scores = await self._score_batch(query, batch)
            
            for candidate, score in zip(batch, batch_scores):
                candidate.maxsim_score = score  # Update with MonoT5 score
                scored_candidates.append(candidate)
        
        # Sort and take top-k
        scored_candidates.sort(key=lambda x: x.maxsim_score, reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"MonoT5 reranking: {len(candidates)} → {top_k} in {processing_time:.2f}ms")
        
        return scored_candidates[:top_k]
    
    async def _score_batch(self, query: str, candidates: List[MaxSimResult]) -> List[float]:
        """Score a batch of candidates with MonoT5"""
        scores = []
        
        with torch.no_grad():
            for candidate in candidates:
                try:
                    # Format input for T5 (query: {query} document: {doc})
                    input_text = f"Query: {query} Document: {candidate.text[:512]} Relevant:"
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # For MonoT5, we would generate "true"/"false" and use the logits
                    # This is simplified - in practice requires more complex setup
                    outputs = self.model.encoder(**inputs)
                    pooled = torch.mean(outputs.last_hidden_state, dim=1)
                    
                    # Simple scoring (would be replaced with proper MonoT5 scoring)
                    score = torch.sigmoid(pooled.mean()).item()
                    scores.append(score)
                    
                except Exception as e:
                    logger.error(f"MonoT5 scoring failed for candidate: {e}")
                    scores.append(candidate.maxsim_score * 0.8)  # Fallback
        
        return scores


class RankLLaMAReranker:
    """RankLLaMA LLM-based reasoning reranker using dedicated LLM reranker"""
    
    def __init__(self, config: CascadeConfig):
        self.config = config
        
        # Create LLM reranker configuration
        llm_config = LLMRerankConfig(
            model_name=config.rankllama_model,
            model_type=config.llm_model_type,
            max_input_length=config.llm_max_input_length,
            temperature=config.llm_temperature,
            use_reasoning=config.use_llm_reasoning,
            final_top_k=config.rankllama_candidates
        )
        
        # Initialize appropriate LLM reranker
        try:
            if config.llm_model_type == "simple":
                self.llm_reranker = SimpleLLMReranker(llm_config)
            else:
                self.llm_reranker = LLMReranker.create(llm_config)
        except Exception as e:
            logger.warning(f"Failed to load LLM reranker, using simple fallback: {e}")
            self.llm_reranker = SimpleLLMReranker(llm_config)
        
        logger.info(f"RankLLaMA reranker initialized with {config.llm_model_type} model")
    
    async def rerank(self, query: str, candidates: List[MaxSimResult], top_k: int) -> List[MaxSimResult]:
        """LLM-based reasoning reranking"""
        start_time = time.time()
        
        if len(candidates) <= top_k:
            return candidates
        
        # Convert MaxSimResult to format expected by LLM reranker
        llm_candidates = []
        for candidate in candidates:
            llm_candidates.append({
                "chunk_id": candidate.chunk_id,
                "document_id": candidate.document_id,
                "text": candidate.text,
                "metadata": candidate.metadata,
                "monot5_score": candidate.maxsim_score  # Pass previous score
            })
        
        try:
            # Perform LLM reranking
            llm_results = await self.llm_reranker.rerank(query, llm_candidates, top_k)
            
            # Convert back to MaxSimResult format
            reranked_results = []
            for llm_result in llm_results:
                # Find original candidate
                original_candidate = next(
                    (c for c in candidates if c.chunk_id == llm_result.chunk_id),
                    None
                )
                
                if original_candidate:
                    # Update score with LLM score
                    original_candidate.maxsim_score = llm_result.llm_score
                    # Store LLM reasoning in metadata
                    if not hasattr(original_candidate, 'llm_reasoning'):
                        original_candidate.metadata["llm_reasoning"] = llm_result.reasoning
                        original_candidate.metadata["llm_rank"] = llm_result.llm_rank
                        original_candidate.metadata["llm_confidence"] = llm_result.confidence
                    
                    reranked_results.append(original_candidate)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"RankLLaMA reranking: {len(candidates)} → {len(reranked_results)} in {processing_time:.2f}ms")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            # Fallback to original order
            return candidates[:top_k]


class CascadeReranker:
    """Main cascade reranker orchestrating all three stages"""
    
    def __init__(self, config: Optional[CascadeConfig] = None):
        self.config = config or CascadeConfig()
        
        # Initialize stage rerankers
        device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        
        self.tilde_reranker = TILDEReranker(self.config.tilde_model, device)
        self.monot5_reranker = MonoT5Reranker(self.config.monot5_model, device)
        self.rankllama_reranker = RankLLaMAReranker(self.config)
        
        logger.info(f"Cascade reranker initialized with device: {device}")
    
    async def rerank(self, query: str, candidates: List[MaxSimResult]) -> List[CascadeResult]:
        """
        Execute full 3-stage cascade reranking
        
        Pipeline:
        MaxSim Results → TILDE Filter → MonoT5 Semantic → RankLLaMA Reasoning → Final Results
        """
        if not candidates:
            return []
        
        stages = []
        start_time = time.time()
        
        logger.info(f"Starting cascade reranking with {len(candidates)} candidates")
        
        # Stage 1: TILDE sparse filtering (1000 → 100)
        stage1_start = time.time()
        tilde_results = await self.tilde_reranker.rerank(
            query, candidates, self.config.tilde_candidates
        )
        stage1_time = (time.time() - stage1_start) * 1000
        
        stages.append(RerankingStage(
            stage_name="TILDE",
            input_count=len(candidates),
            output_count=len(tilde_results),
            processing_time_ms=stage1_time,
            scores=[r.maxsim_score for r in tilde_results[:5]]  # Sample scores
        ))
        
        if not tilde_results:
            return []
        
        # Stage 2: MonoT5 deep semantic (100 → 20)
        stage2_start = time.time()
        monot5_results = await self.monot5_reranker.rerank(
            query, tilde_results, self.config.monot5_candidates
        )
        stage2_time = (time.time() - stage2_start) * 1000
        
        stages.append(RerankingStage(
            stage_name="MonoT5",
            input_count=len(tilde_results),
            output_count=len(monot5_results),
            processing_time_ms=stage2_time,
            scores=[r.maxsim_score for r in monot5_results[:5]]
        ))
        
        if not monot5_results:
            return []
        
        # Stage 3: RankLLaMA reasoning (20 → 5)
        stage3_start = time.time()
        final_results = await self.rankllama_reranker.rerank(
            query, monot5_results, self.config.rankllama_candidates
        )
        stage3_time = (time.time() - stage3_start) * 1000
        
        stages.append(RerankingStage(
            stage_name="RankLLaMA",
            input_count=len(monot5_results),
            output_count=len(final_results),
            processing_time_ms=stage3_time,
            scores=[r.maxsim_score for r in final_results]
        ))
        
        # Convert to CascadeResult objects
        cascade_results = []
        for result in final_results:
            cascade_result = CascadeResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                final_score=result.maxsim_score,
                text=result.text,
                metadata=result.metadata,
                parent_chunk_id=result.parent_chunk_id,
                
                # Detailed scores (simplified - in practice would track through stages)
                maxsim_score=result.maxsim_score,
                tilde_score=result.maxsim_score,
                monot5_score=result.maxsim_score,
                rankllama_score=result.maxsim_score,
                
                stages=stages.copy()
            )
            cascade_results.append(cascade_result)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Cascade reranking completed: {len(candidates)} → {len(cascade_results)} in {total_time:.2f}ms")
        
        return cascade_results
    
    def get_performance_stats(self, results: List[CascadeResult]) -> Dict[str, Any]:
        """Get performance statistics for the cascade"""
        if not results:
            return {}
        
        # Aggregate stage statistics
        stage_stats = {}
        for result in results[:1]:  # Use first result as representative
            for stage in result.stages:
                if stage.stage_name not in stage_stats:
                    stage_stats[stage.stage_name] = {
                        "avg_time_ms": stage.processing_time_ms,
                        "reduction_ratio": stage.input_count / stage.output_count if stage.output_count > 0 else 0,
                        "output_count": stage.output_count
                    }
        
        return {
            "total_results": len(results),
            "stage_performance": stage_stats,
            "final_scores": [r.final_score for r in results],
            "score_distribution": {
                "min": min(r.final_score for r in results),
                "max": max(r.final_score for r in results),
                "mean": sum(r.final_score for r in results) / len(results)
            }
        }
