"""
Document Chunking with Parent/Child Hierarchy
Creates optimal chunks for multi-vector embeddings
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re


@dataclass
class ChunkMetadata:
    """Chunk metadata"""
    chunk_id: str
    parent_id: str
    doc_id: str
    chunk_type: str  # 'parent' or 'child'
    position: int
    section_title: Optional[str] = None


@dataclass
class DocumentChunk:
    """Document chunk with hierarchy"""
    text: str
    metadata: ChunkMetadata
    token_count: int


class DocumentChunker:
    """Hierarchical document chunker for RAG optimization"""
    
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 250,
        overlap: int = 50
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap
    
    def chunk_document(self, doc_id: str, text: str) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Create parent/child hierarchy chunks
        Returns: (parent_chunks, child_chunks)
        """
        # Detect sections using headers
        sections = self._detect_sections(text)
        
        parent_chunks = []
        child_chunks = []
        
        for i, (section_title, section_text) in enumerate(sections):
            # Create parent chunks (full sections or large chunks)
            parent_chunk_texts = self._create_chunks(section_text, self.parent_chunk_size, self.overlap)
            
            for j, parent_text in enumerate(parent_chunk_texts):
                parent_id = f"{doc_id}_parent_{i}_{j}"
                
                # Create parent chunk
                parent_chunk = DocumentChunk(
                    text=parent_text,
                    metadata=ChunkMetadata(
                        chunk_id=parent_id,
                        parent_id=parent_id,  # Parent points to itself
                        doc_id=doc_id,
                        chunk_type="parent",
                        position=len(parent_chunks),
                        section_title=section_title
                    ),
                    token_count=self._estimate_tokens(parent_text)
                )
                parent_chunks.append(parent_chunk)
                
                # Create child chunks from parent
                child_texts = self._create_chunks(parent_text, self.child_chunk_size, self.overlap)
                
                for k, child_text in enumerate(child_texts):
                    child_id = f"{parent_id}_child_{k}"
                    
                    child_chunk = DocumentChunk(
                        text=child_text,
                        metadata=ChunkMetadata(
                            chunk_id=child_id,
                            parent_id=parent_id,
                            doc_id=doc_id,
                            chunk_type="child",
                            position=k,
                            section_title=section_title
                        ),
                        token_count=self._estimate_tokens(child_text)
                    )
                    child_chunks.append(child_chunk)
        
        return parent_chunks, child_chunks
    
    def _detect_sections(self, text: str) -> List[Tuple[str, str]]:
        """Detect document sections using headers"""
        # Simple regex-based section detection
        section_pattern = r'^(#{1,3}\s+.+|[A-Z][A-Z\s]{2,20}[A-Z])$'
        lines = text.split('\n')
        
        sections = []
        current_section = "Introduction"
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if re.match(section_pattern, line):
                if current_text:
                    sections.append((current_section, current_text.strip()))
                current_section = line.replace('#', '').strip()
                current_text = ""
            else:
                current_text += line + "\n"
        
        # Add final section
        if current_text:
            sections.append((current_section, current_text.strip()))
        
        return sections if sections else [("Content", text)]
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping chunks from text"""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            
            if end >= len(words):
                break
                
            start = end - overlap
        
        return chunks if chunks else [text]
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
