"""
Parent Retrieval and Reverse Repacking
Expands child chunks to full parent contexts and optimizes for LLM attention
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging

from storage.postgresql.database import DatabaseManager
from retrieval_pipeline.reranking.cascade_reranker import CascadeResult

logger = logging.getLogger(__name__)


@dataclass
class ParentContext:
    """Parent context with metadata"""
    parent_chunk_id: str
    document_id: str
    parent_text: str
    child_snippets: List[str]
    child_positions: List[int]
    relevance_score: float
    section_title: Optional[str] = None
    token_count: int = 0


@dataclass
class RepackedContext:
    """Repacked context optimized for LLM"""
    text: str
    sources: List[Dict[str, Any]]
    total_tokens: int
    context_windows: List[Dict[str, Any]]  # For attention optimization
    relevance_scores: List[float]


@dataclass
class ContextConfig:
    """Context processing configuration"""
    max_parent_tokens: int = 2000  # Maximum tokens per parent
    max_total_tokens: int = 8000   # Maximum total context length
    min_snippet_tokens: int = 50   # Minimum child snippet size
    overlap_tokens: int = 25       # Overlap between snippets
    preserve_order: bool = True    # Maintain document order
    include_section_headers: bool = True
    merge_adjacent_chunks: bool = True


class ParentRetriever:
    """Retrieves and processes parent contexts for child chunk results"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    async def retrieve_parent_contexts(
        self,
        cascade_results: List[CascadeResult],
        config: Optional[ContextConfig] = None
    ) -> List[ParentContext]:
        """
        Retrieve parent contexts for cascaded results
        Converts child chunks back to full parent contexts
        """
        config = config or ContextConfig()
        
        if not cascade_results:
            return []
        
        # Group results by parent chunk
        parent_groups = self._group_by_parent(cascade_results)
        
        # Fetch parent chunk data
        parent_contexts = []
        
        for parent_id, child_results in parent_groups.items():
            try:
                parent_context = await self._build_parent_context(
                    parent_id, child_results, config
                )
                
                if parent_context:
                    parent_contexts.append(parent_context)
                    
            except Exception as e:
                logger.error(f"Failed to build parent context for {parent_id}: {e}")
        
        # Sort by highest child relevance score
        parent_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Retrieved {len(parent_contexts)} parent contexts")
        return parent_contexts
    
    def _group_by_parent(
        self, 
        cascade_results: List[CascadeResult]
    ) -> Dict[str, List[CascadeResult]]:
        """Group cascade results by parent chunk ID"""
        parent_groups = {}
        
        for result in cascade_results:
            parent_id = result.parent_chunk_id
            
            if not parent_id:
                # Handle orphan chunks (create synthetic parent)
                parent_id = f"synthetic_{result.document_id}_{result.chunk_id}"
            
            if parent_id not in parent_groups:
                parent_groups[parent_id] = []
            
            parent_groups[parent_id].append(result)
        
        return parent_groups
    
    async def _build_parent_context(
        self,
        parent_id: str,
        child_results: List[CascadeResult],
        config: ContextConfig
    ) -> Optional[ParentContext]:
        """Build parent context from child results"""
        
        # Get parent chunk from database
        if parent_id.startswith("synthetic_"):
            # Handle synthetic parent (orphan child)
            return self._create_synthetic_parent(parent_id, child_results, config)
        
        async with self.db.get_session() as session:
            # Fetch parent chunk
            parent_chunk = await session.get("DocumentChunk", parent_id)
            
            if not parent_chunk:
                logger.warning(f"Parent chunk {parent_id} not found")
                return None
            
            # Calculate aggregated relevance score
            relevance_score = max(result.final_score for result in child_results)
            
            # Extract child snippets and positions
            child_snippets = []
            child_positions = []
            
            for result in child_results:
                child_snippets.append(result.text)
                child_positions.append(result.metadata.get("position", 0))
            
            # Estimate token count
            token_count = self._estimate_tokens(parent_chunk.text)
            
            return ParentContext(
                parent_chunk_id=parent_id,
                document_id=parent_chunk.document_id,
                parent_text=parent_chunk.text,
                child_snippets=child_snippets,
                child_positions=child_positions,
                relevance_score=relevance_score,
                section_title=parent_chunk.section_title,
                token_count=token_count
            )
    
    def _create_synthetic_parent(
        self,
        synthetic_id: str,
        child_results: List[CascadeResult],
        config: ContextConfig
    ) -> ParentContext:
        """Create synthetic parent for orphan child chunks"""
        
        # Combine child texts as parent
        combined_text = "\n\n".join(result.text for result in child_results)
        document_id = child_results[0].document_id if child_results else "unknown"
        
        return ParentContext(
            parent_chunk_id=synthetic_id,
            document_id=document_id,
            parent_text=combined_text,
            child_snippets=[result.text for result in child_results],
            child_positions=list(range(len(child_results))),
            relevance_score=max(result.final_score for result in child_results),
            section_title="Combined Context",
            token_count=self._estimate_tokens(combined_text)
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return max(len(text) // 4, 1)


class ReverseRepacker:
    """Optimizes parent contexts for LLM attention and processing"""
    
    def __init__(self):
        pass
    
    def repack_contexts(
        self,
        parent_contexts: List[ParentContext],
        config: Optional[ContextConfig] = None
    ) -> RepackedContext:
        """
        Reverse repack contexts for optimal LLM processing
        
        Strategies:
        1. Merge adjacent contexts from same document
        2. Optimize token usage within limits
        3. Create attention-friendly structure
        4. Preserve relevance hierarchy
        """
        config = config or ContextConfig()
        
        if not parent_contexts:
            return RepackedContext(
                text="", sources=[], total_tokens=0, 
                context_windows=[], relevance_scores=[]
            )
        
        # Step 1: Merge adjacent chunks if enabled
        if config.merge_adjacent_chunks:
            parent_contexts = self._merge_adjacent_contexts(parent_contexts, config)
        
        # Step 2: Optimize token usage
        optimized_contexts = self._optimize_token_usage(parent_contexts, config)
        
        # Step 3: Create final repacked context
        repacked = self._create_repacked_context(optimized_contexts, config)
        
        logger.info(f"Repacked {len(parent_contexts)} contexts into {repacked.total_tokens} tokens")
        return repacked
    
    def _merge_adjacent_contexts(
        self,
        contexts: List[ParentContext],
        config: ContextConfig
    ) -> List[ParentContext]:
        """Merge adjacent contexts from the same document"""
        if not contexts:
            return contexts
        
        merged_contexts = []
        current_group = [contexts[0]]
        
        for i in range(1, len(contexts)):
            current = contexts[i]
            prev = contexts[i-1]
            
            # Check if should merge with previous
            should_merge = (
                current.document_id == prev.document_id and
                abs(max(current.child_positions) - max(prev.child_positions)) <= 2 and
                (current.token_count + prev.token_count) <= config.max_parent_tokens
            )
            
            if should_merge:
                current_group.append(current)
            else:
                # Process current group
                if len(current_group) > 1:
                    merged = self._merge_context_group(current_group)
                    merged_contexts.append(merged)
                else:
                    merged_contexts.append(current_group[0])
                
                current_group = [current]
        
        # Handle final group
        if len(current_group) > 1:
            merged = self._merge_context_group(current_group)
            merged_contexts.append(merged)
        else:
            merged_contexts.extend(current_group)
        
        return merged_contexts
    
    def _merge_context_group(self, contexts: List[ParentContext]) -> ParentContext:
        """Merge a group of contexts into one"""
        if len(contexts) == 1:
            return contexts[0]
        
        # Combine texts
        combined_text = "\n\n---\n\n".join(ctx.parent_text for ctx in contexts)
        
        # Combine child snippets
        all_snippets = []
        all_positions = []
        for ctx in contexts:
            all_snippets.extend(ctx.child_snippets)
            all_positions.extend(ctx.child_positions)
        
        # Use highest relevance score
        max_relevance = max(ctx.relevance_score for ctx in contexts)
        
        # Combine section titles
        section_titles = [ctx.section_title for ctx in contexts if ctx.section_title]
        combined_section = " | ".join(section_titles) if section_titles else None
        
        return ParentContext(
            parent_chunk_id=f"merged_{contexts[0].parent_chunk_id}",
            document_id=contexts[0].document_id,
            parent_text=combined_text,
            child_snippets=all_snippets,
            child_positions=all_positions,
            relevance_score=max_relevance,
            section_title=combined_section,
            token_count=sum(ctx.token_count for ctx in contexts)
        )
    
    def _optimize_token_usage(
        self,
        contexts: List[ParentContext],
        config: ContextConfig
    ) -> List[ParentContext]:
        """Optimize contexts to fit within token limits"""
        optimized = []
        total_tokens = 0
        
        for context in contexts:
            if total_tokens >= config.max_total_tokens:
                break
            
            remaining_tokens = config.max_total_tokens - total_tokens
            
            if context.token_count <= remaining_tokens:
                # Fits entirely
                optimized.append(context)
                total_tokens += context.token_count
            elif remaining_tokens >= config.min_snippet_tokens:
                # Truncate to fit
                truncated = self._truncate_context(context, remaining_tokens)
                optimized.append(truncated)
                total_tokens += truncated.token_count
                break
            else:
                # No space left
                break
        
        return optimized
    
    def _truncate_context(self, context: ParentContext, max_tokens: int) -> ParentContext:
        """Truncate context to fit token limit"""
        # Simple truncation - could be smarter about preserving important parts
        words = context.parent_text.split()
        target_words = max_tokens * 4  # Rough conversion
        
        if len(words) <= target_words:
            return context
        
        # Truncate and add ellipsis
        truncated_words = words[:target_words - 10]  # Leave room for ellipsis
        truncated_text = " ".join(truncated_words) + "\n\n[... content truncated ...]"
        
        return ParentContext(
            parent_chunk_id=context.parent_chunk_id,
            document_id=context.document_id,
            parent_text=truncated_text,
            child_snippets=context.child_snippets[:3],  # Keep first few snippets
            child_positions=context.child_positions[:3],
            relevance_score=context.relevance_score,
            section_title=context.section_title,
            token_count=max_tokens
        )
    
    def _create_repacked_context(
        self,
        contexts: List[ParentContext],
        config: ContextConfig
    ) -> RepackedContext:
        """Create final repacked context structure"""
        if not contexts:
            return RepackedContext(
                text="", sources=[], total_tokens=0, 
                context_windows=[], relevance_scores=[]
            )
        
        # Build context text with structure
        context_parts = []
        sources = []
        context_windows = []
        relevance_scores = []
        
        for i, context in enumerate(contexts):
            # Add section header if enabled
            if config.include_section_headers and context.section_title:
                context_parts.append(f"\n## {context.section_title}\n")
            
            # Add context text
            context_parts.append(context.parent_text)
            
            # Track source information
            sources.append({
                "document_id": context.document_id,
                "parent_chunk_id": context.parent_chunk_id,
                "section_title": context.section_title,
                "relevance_score": context.relevance_score,
                "token_count": context.token_count
            })
            
            # Create context window for attention optimization
            start_pos = sum(len(part) for part in context_parts[:-1])
            context_windows.append({
                "start_pos": start_pos,
                "end_pos": start_pos + len(context_parts[-1]),
                "importance": context.relevance_score,
                "section": context.section_title or f"Context {i+1}"
            })
            
            relevance_scores.append(context.relevance_score)
            
            # Add separator between contexts
            if i < len(contexts) - 1:
                context_parts.append("\n\n---\n\n")
        
        # Combine all parts
        final_text = "".join(context_parts)
        total_tokens = sum(ctx.token_count for ctx in contexts)
        
        return RepackedContext(
            text=final_text,
            sources=sources,
            total_tokens=total_tokens,
            context_windows=context_windows,
            relevance_scores=relevance_scores
        )
    
    def create_attention_mask(
        self,
        repacked_context: RepackedContext,
        query: str
    ) -> List[float]:
        """
        Create attention weights for different parts of the context
        Higher weights for more relevant sections
        """
        if not repacked_context.context_windows:
            return [1.0]
        
        # Base attention weights on relevance scores
        max_relevance = max(repacked_context.relevance_scores) if repacked_context.relevance_scores else 1.0
        min_relevance = min(repacked_context.relevance_scores) if repacked_context.relevance_scores else 0.0
        
        attention_weights = []
        
        for i, window in enumerate(repacked_context.context_windows):
            relevance = repacked_context.relevance_scores[i] if i < len(repacked_context.relevance_scores) else 0.5
            
            # Normalize relevance to [0.3, 1.0] range
            if max_relevance > min_relevance:
                normalized_relevance = 0.3 + 0.7 * (relevance - min_relevance) / (max_relevance - min_relevance)
            else:
                normalized_relevance = 1.0
            
            attention_weights.append(normalized_relevance)
        
        return attention_weights
    
    def get_context_summary(self, repacked_context: RepackedContext) -> Dict[str, Any]:
        """Get summary statistics for the repacked context"""
        return {
            "total_tokens": repacked_context.total_tokens,
            "num_sources": len(repacked_context.sources),
            "num_windows": len(repacked_context.context_windows),
            "relevance_stats": {
                "min": min(repacked_context.relevance_scores) if repacked_context.relevance_scores else 0,
                "max": max(repacked_context.relevance_scores) if repacked_context.relevance_scores else 0,
                "mean": sum(repacked_context.relevance_scores) / len(repacked_context.relevance_scores) if repacked_context.relevance_scores else 0
            },
            "document_coverage": len(set(src["document_id"] for src in repacked_context.sources)),
            "sections": [src["section_title"] for src in repacked_context.sources if src["section_title"]]
        }
