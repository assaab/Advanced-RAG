"""
Metadata Extraction from Academic Papers
Extracts structured metadata for better retrieval
"""
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
from data_pipeline.ingestion.pdf_loader import DocumentMetadata


@dataclass
class ExtractedMetadata:
    """Enhanced metadata from document analysis"""
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    sections: List[str]
    references_count: int
    figures_count: int
    tables_count: int


class MetadataExtractor:
    """Extract and enhance document metadata"""
    
    def __init__(self):
        self.title_patterns = [
            r'^(.+?)(?:\n|\r)',  # First line
            r'Title:\s*(.+?)(?:\n|\r)',
            r'title\s*[:=]\s*(.+?)(?:\n|\r)',
        ]
        
        self.author_patterns = [
            r'Authors?:\s*(.+?)(?:\n|\r)',
            r'By:\s*(.+?)(?:\n|\r)',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*)',
        ]
        
        self.abstract_patterns = [
            r'Abstract\s*:?\s*(.+?)(?:\n\n|\n[A-Z])',
            r'ABSTRACT\s*:?\s*(.+?)(?:\n\n|\n[A-Z])',
        ]
    
    async def extract_metadata(self, text: str, base_metadata: DocumentMetadata) -> ExtractedMetadata:
        """Extract enhanced metadata from document text"""
        
        # Extract title
        title = self._extract_title(text) or base_metadata.title
        
        # Extract authors
        authors = self._extract_authors(text)
        if not authors:
            authors = base_metadata.authors
        
        # Extract abstract
        abstract = self._extract_abstract(text) or base_metadata.abstract
        
        # Extract keywords
        keywords = self._extract_keywords(text)
        
        # Extract sections
        sections = self._extract_sections(text)
        
        # Count elements
        references_count = self._count_references(text)
        figures_count = self._count_figures(text)
        tables_count = self._count_tables(text)
        
        return ExtractedMetadata(
            title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            sections=sections,
            references_count=references_count,
            figures_count=figures_count,
            tables_count=tables_count
        )
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract document title"""
        for pattern in self.title_patterns:
            if match := re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                title = match.group(1).strip()
                if 10 <= len(title) <= 200:  # Reasonable title length
                    return title
        return None
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract authors"""
        for pattern in self.author_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                authors_text = match.group(1).strip()
                # Split by common delimiters
                authors = re.split(r',\s*|\s+and\s+|\s*&\s*', authors_text)
                return [author.strip() for author in authors if author.strip()]
        return []
    
    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract"""
        for pattern in self.abstract_patterns:
            if match := re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                abstract = match.group(1).strip()
                if 50 <= len(abstract) <= 2000:  # Reasonable abstract length
                    return abstract
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords"""
        # Look for keywords section
        keyword_pattern = r'Keywords?\s*:?\s*(.+?)(?:\n\n|\n[A-Z])'
        if match := re.search(keyword_pattern, text, re.IGNORECASE):
            keywords_text = match.group(1).strip()
            keywords = re.split(r',\s*|;\s*', keywords_text)
            return [kw.strip() for kw in keywords if kw.strip()]
        
        # Extract from common academic terms
        academic_terms = [
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
            'natural language processing', 'computer vision', 'reinforcement learning',
            'transformer', 'attention mechanism', 'embedding', 'classification', 'regression'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for term in academic_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords[:10]  # Limit to 10 keywords
    
    def _extract_sections(self, text: str) -> List[str]:
        """Extract section headers"""
        section_patterns = [
            r'^(#{1,3}\s+(.+))$',  # Markdown headers
            r'^([A-Z][A-Z\s]{2,30}[A-Z])$',  # ALL CAPS headers
            r'^\d+\.?\s+([A-Z][a-zA-Z\s]{5,50})$',  # Numbered sections
        ]
        
        sections = []
        for line in text.split('\n'):
            line = line.strip()
            for pattern in section_patterns:
                if match := re.match(pattern, line):
                    section = match.group(1).replace('#', '').strip()
                    if section not in sections and len(section) > 3:
                        sections.append(section)
        
        return sections[:20]  # Limit to 20 sections
    
    def _count_references(self, text: str) -> int:
        """Count references"""
        ref_patterns = [
            r'\[\d+\]',  # [1], [23]
            r'\(\d{4}\)',  # (2023)
            r'et al\.',  # et al.
        ]
        
        count = 0
        for pattern in ref_patterns:
            count += len(re.findall(pattern, text))
        
        return min(count, 500)  # Cap at reasonable number
    
    def _count_figures(self, text: str) -> int:
        """Count figures"""
        figure_patterns = [
            r'Figure\s+\d+',
            r'Fig\.\s+\d+',
            r'figure\s+\d+',
        ]
        
        count = 0
        for pattern in figure_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return count
    
    def _count_tables(self, text: str) -> int:
        """Count tables"""
        table_patterns = [
            r'Table\s+\d+',
            r'table\s+\d+',
        ]
        
        count = 0
        for pattern in table_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return count
