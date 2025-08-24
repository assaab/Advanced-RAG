"""
Hierarchical Chunking for Parent-Child Document Structure
Implements the document preprocessing phase of the advanced RAG pipeline
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
import tiktoken
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    chunk_id: str
    document_id: str
    parent_chunk_id: Optional[str]
    chunk_type: str  # 'parent' or 'child'
    position: int
    section_title: Optional[str] = None
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0


@dataclass
class DocumentChunk:
    """Single document chunk with metadata"""
    text: str
    metadata: ChunkMetadata


@dataclass
class HierarchicalChunks:
    """Complete hierarchical chunking result"""
    document_id: str
    parent_chunks: List[DocumentChunk]
    child_chunks: List[DocumentChunk]
    total_tokens: int
    chunk_relationships: Dict[str, List[str]]  # parent_id -> [child_ids]


class HierarchicalChunker:
    """
    Advanced hierarchical chunker that creates parent-child chunk relationships
    
    Process:
    1. Split document into sections (if structure available)
    2. Create parent chunks (500-1000 tokens) - Full context paragraphs
    3. Create child chunks (50-150 tokens) - Granular semantic units
    4. Maintain parent-child relationships for context retrieval
    """
    
    def __init__(
        self,
        parent_min_tokens: int = 500,
        parent_max_tokens: int = 1000,
        child_min_tokens: int = 50,
        child_max_tokens: int = 150,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base"  # GPT-4 tokenizer
    ):
        self.parent_min_tokens = parent_min_tokens
        self.parent_max_tokens = parent_max_tokens
        self.child_min_tokens = child_min_tokens
        self.child_max_tokens = child_max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        logger.info(f"Hierarchical chunker initialized with parent: {parent_min_tokens}-{parent_max_tokens}, child: {child_min_tokens}-{child_max_tokens}")
    
    async def chunk_document(
        self,
        document_id: str,
        text: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> HierarchicalChunks:
        """
        Create hierarchical chunks for a document
        
        Args:
            document_id: Unique document identifier
            text: Full document text
            document_metadata: Optional document metadata with sections
        
        Returns:
            Complete hierarchical chunking result
        """
        logger.info(f"Starting hierarchical chunking for document {document_id}")
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Extract document structure if available
        sections = self._extract_sections(cleaned_text, document_metadata)
        
        # Create parent chunks
        parent_chunks = await self._create_parent_chunks(
            document_id, sections, cleaned_text
        )
        
        # Create child chunks from parent chunks
        child_chunks = await self._create_child_chunks(parent_chunks)
        
        # Build relationships
        chunk_relationships = self._build_relationships(parent_chunks, child_chunks)
        
        # Calculate total tokens
        total_tokens = sum(chunk.metadata.token_count for chunk in parent_chunks + child_chunks)
        
        result = HierarchicalChunks(
            document_id=document_id,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            total_tokens=total_tokens,
            chunk_relationships=chunk_relationships
        )
        
        logger.info(f"Hierarchical chunking completed: {len(parent_chunks)} parents, {len(child_chunks)} children, {total_tokens} tokens")
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess document text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove page headers/footers (basic patterns)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space before capital letters
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)     # Add space after periods
        
        return text.strip()
    
    def _extract_sections(
        self, 
        text: str, 
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract document sections based on headers or metadata
        Returns list of sections with title, text, and position
        """
        sections = []
        
        # Try to use provided section metadata first
        if document_metadata and "sections" in document_metadata:
            # Use provided section structure
            current_pos = 0
            for i, section_title in enumerate(document_metadata["sections"]):
                # Find section text (simplified - in practice would use more sophisticated parsing)
                next_section = (document_metadata["sections"][i + 1] 
                              if i + 1 < len(document_metadata["sections"]) else None)
                
                # Find section boundaries
                start_idx = text.find(section_title, current_pos)
                if start_idx == -1:
                    continue
                
                if next_section:
                    end_idx = text.find(next_section, start_idx + len(section_title))
                    if end_idx == -1:
                        end_idx = len(text)
                else:
                    end_idx = len(text)
                
                section_text = text[start_idx:end_idx].strip()
                
                if len(section_text) > 100:  # Skip very short sections
                    sections.append({
                        "title": section_title,
                        "text": section_text,
                        "start_pos": start_idx,
                        "end_pos": end_idx,
                        "position": i
                    })
                
                current_pos = end_idx
        
        # Fallback: Use pattern-based section detection
        if not sections:
            sections = self._detect_sections_by_pattern(text)
        
        # If no sections found, treat entire document as one section
        if not sections:
            sections = [{
                "title": "Full Document",
                "text": text,
                "start_pos": 0,
                "end_pos": len(text),
                "position": 0
            }]
        
        return sections
    
    def _detect_sections_by_pattern(self, text: str) -> List[Dict[str, Any]]:
        """Detect sections using common academic paper patterns"""
        sections = []
        
        # Common section headers (case-insensitive)
        section_patterns = [
            r'\n\s*(Abstract|Introduction|Background|Related Work|Methodology|Methods|Results|Discussion|Conclusion|References|Acknowledgments)\s*\n',
            r'\n\s*\d+\.?\s+(Introduction|Background|Methodology|Results|Discussion|Conclusion)\s*\n',
            r'\n\s*[IVX]+\.?\s+[A-Z][A-Za-z\s]+\n'  # Roman numerals
        ]
        
        all_matches = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                all_matches.append({
                    "start": match.start(),
                    "end": match.end(),
                    "title": match.group(1) if match.groups() else match.group().strip(),
                    "match": match
                })
        
        # Sort by position and create sections
        all_matches.sort(key=lambda x: x["start"])
        
        for i, match in enumerate(all_matches):
            start_pos = match["start"]
            end_pos = all_matches[i + 1]["start"] if i + 1 < len(all_matches) else len(text)
            
            section_text = text[start_pos:end_pos].strip()
            
            if len(section_text) > 200:  # Minimum section length
                sections.append({
                    "title": match["title"].strip(),
                    "text": section_text,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "position": i
                })
        
        return sections
    
    async def _create_parent_chunks(
        self,
        document_id: str,
        sections: List[Dict[str, Any]],
        full_text: str
    ) -> List[DocumentChunk]:
        """Create parent chunks (500-1000 tokens) from sections"""
        parent_chunks = []
        chunk_position = 0
        
        for section in sections:
            section_text = section["text"]
            section_title = section["title"]
            
            # Split section into parent-sized chunks
            section_parents = await self._split_into_parent_chunks(
                document_id, section_text, section_title, chunk_position
            )
            
            parent_chunks.extend(section_parents)
            chunk_position += len(section_parents)
        
        return parent_chunks
    
    async def _split_into_parent_chunks(
        self,
        document_id: str,
        text: str,
        section_title: str,
        start_position: int
    ) -> List[DocumentChunk]:
        """Split text into parent-sized chunks with overlap"""
        chunks = []
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.parent_max_tokens:
            # Single chunk for short text
            chunk_id = str(uuid.uuid4())
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                parent_chunk_id=None,
                chunk_type="parent",
                position=start_position,
                section_title=section_title,
                token_count=len(tokens)
            )
            
            chunks.append(DocumentChunk(
                text=text,
                metadata=metadata
            ))
            return chunks
        
        # Split into multiple parent chunks with overlap
        chunk_start = 0
        position = start_position
        
        while chunk_start < len(tokens):
            # Determine chunk end
            chunk_end = min(chunk_start + self.parent_max_tokens, len(tokens))
            
            # Try to end at sentence boundary
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Find good breaking point
            if chunk_end < len(tokens):
                chunk_text = self._find_sentence_boundary(chunk_text)
                chunk_tokens = self.tokenizer.encode(chunk_text)
            
            # Ensure minimum chunk size
            if len(chunk_tokens) >= self.parent_min_tokens or chunk_end >= len(tokens):
                chunk_id = str(uuid.uuid4())
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    parent_chunk_id=None,
                    chunk_type="parent",
                    position=position,
                    section_title=section_title,
                    token_count=len(chunk_tokens)
                )
                
                chunks.append(DocumentChunk(
                    text=chunk_text.strip(),
                    metadata=metadata
                ))
                position += 1
            
            # Move to next chunk with overlap
            chunk_start = chunk_end - self.overlap_tokens
            if chunk_start >= len(tokens):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str) -> str:
        """Find good sentence boundary for chunk ending"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 1:
            return text
        
        # Remove last incomplete sentence
        complete_text = ' '.join(sentences[:-1])
        
        # Check if we're losing too much content
        if len(complete_text) < len(text) * 0.7:
            return text  # Keep original if we lose too much
        
        return complete_text
    
    async def _create_child_chunks(
        self, 
        parent_chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Create child chunks (50-150 tokens) from parent chunks"""
        child_chunks = []
        
        for parent_chunk in parent_chunks:
            parent_children = await self._split_parent_into_children(parent_chunk)
            child_chunks.extend(parent_children)
        
        return child_chunks
    
    async def _split_parent_into_children(
        self, 
        parent_chunk: DocumentChunk
    ) -> List[DocumentChunk]:
        """Split a parent chunk into child chunks"""
        children = []
        parent_text = parent_chunk.text
        parent_tokens = self.tokenizer.encode(parent_text)
        
        if len(parent_tokens) <= self.child_max_tokens:
            # Parent is small enough to be its own child
            child_id = str(uuid.uuid4())
            metadata = ChunkMetadata(
                chunk_id=child_id,
                document_id=parent_chunk.metadata.document_id,
                parent_chunk_id=parent_chunk.metadata.chunk_id,
                chunk_type="child",
                position=0,
                section_title=parent_chunk.metadata.section_title,
                token_count=len(parent_tokens)
            )
            
            children.append(DocumentChunk(
                text=parent_text,
                metadata=metadata
            ))
            return children
        
        # Split into multiple children
        chunk_start = 0
        child_position = 0
        
        while chunk_start < len(parent_tokens):
            chunk_end = min(chunk_start + self.child_max_tokens, len(parent_tokens))
            
            # Extract child chunk
            child_tokens = parent_tokens[chunk_start:chunk_end]
            child_text = self.tokenizer.decode(child_tokens)
            
            # Find sentence boundary for child
            if chunk_end < len(parent_tokens):
                child_text = self._find_sentence_boundary(child_text)
                child_tokens = self.tokenizer.encode(child_text)
            
            # Ensure minimum child size
            if len(child_tokens) >= self.child_min_tokens or chunk_end >= len(parent_tokens):
                child_id = str(uuid.uuid4())
                metadata = ChunkMetadata(
                    chunk_id=child_id,
                    document_id=parent_chunk.metadata.document_id,
                    parent_chunk_id=parent_chunk.metadata.chunk_id,
                    chunk_type="child",
                    position=child_position,
                    section_title=parent_chunk.metadata.section_title,
                    token_count=len(child_tokens)
                )
                
                children.append(DocumentChunk(
                    text=child_text.strip(),
                    metadata=metadata
                ))
                child_position += 1
            
            # Move to next child with small overlap
            overlap = min(self.overlap_tokens // 4, 10)  # Smaller overlap for children
            chunk_start = chunk_end - overlap
            if chunk_start >= len(parent_tokens):
                break
        
        return children
    
    def _build_relationships(
        self,
        parent_chunks: List[DocumentChunk],
        child_chunks: List[DocumentChunk]
    ) -> Dict[str, List[str]]:
        """Build parent-child relationship mapping"""
        relationships = {}
        
        for parent in parent_chunks:
            parent_id = parent.metadata.chunk_id
            relationships[parent_id] = []
        
        for child in child_chunks:
            parent_id = child.metadata.parent_chunk_id
            if parent_id and parent_id in relationships:
                relationships[parent_id].append(child.metadata.chunk_id)
        
        return relationships
    
    def get_chunking_stats(self, result: HierarchicalChunks) -> Dict[str, Any]:
        """Get statistics about the chunking result"""
        parent_tokens = [chunk.metadata.token_count for chunk in result.parent_chunks]
        child_tokens = [chunk.metadata.token_count for chunk in result.child_chunks]
        
        return {
            "document_id": result.document_id,
            "total_chunks": len(result.parent_chunks) + len(result.child_chunks),
            "parent_chunks": len(result.parent_chunks),
            "child_chunks": len(result.child_chunks),
            "total_tokens": result.total_tokens,
            "parent_stats": {
                "min_tokens": min(parent_tokens) if parent_tokens else 0,
                "max_tokens": max(parent_tokens) if parent_tokens else 0,
                "avg_tokens": sum(parent_tokens) / len(parent_tokens) if parent_tokens else 0
            },
            "child_stats": {
                "min_tokens": min(child_tokens) if child_tokens else 0,
                "max_tokens": max(child_tokens) if child_tokens else 0,
                "avg_tokens": sum(child_tokens) / len(child_tokens) if child_tokens else 0
            },
            "avg_children_per_parent": len(result.child_chunks) / len(result.parent_chunks) if result.parent_chunks else 0
        }
