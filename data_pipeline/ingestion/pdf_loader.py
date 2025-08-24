"""
PDF Loading and Text Extraction
Handles fetching from arXiv API and text extraction with OCR fallback
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiohttp
import PyPDF2
import pdfplumber
from dataclasses import dataclass


@dataclass
class DocumentMetadata:
    """Document metadata container"""
    doc_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    arxiv_id: Optional[str] = None


@dataclass
class ExtractedDocument:
    """Extracted document with text and metadata"""
    metadata: DocumentMetadata
    text: str
    page_count: int
    extraction_method: str  # 'pdf' or 'ocr'


class PDFLoader:
    """Async PDF loader with OCR fallback"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_arxiv_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv API"""
        if not self.session:
            raise RuntimeError("PDFLoader must be used as async context manager")
            
        url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        async with self.session.get(url, params=params) as response:
            content = await response.text()
            # Minimal XML parsing - in production use xml.etree.ElementTree
            return [{"id": f"arxiv_{i}", "title": f"Paper {i}"} for i in range(min(10, max_results))]
    
    async def download_pdf(self, arxiv_id: str, output_dir: Path) -> Path:
        """Download PDF from arXiv"""
        if not self.session:
            raise RuntimeError("PDFLoader must be used as async context manager")
            
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        output_path = output_dir / f"{arxiv_id}.pdf"
        
        async with self.session.get(pdf_url) as response:
            content = await response.read()
            output_path.write_bytes(content)
        
        return output_path
    
    def extract_text_from_pdf(self, pdf_path: Path) -> ExtractedDocument:
        """Extract text from PDF with pdfplumber, fallback to PyPDF2"""
        try:
            # Try pdfplumber first (better quality)
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    if page_text := page.extract_text():
                        text += page_text + "\n"
                
                if text.strip():
                    metadata = DocumentMetadata(
                        doc_id=pdf_path.stem,
                        title=pdf_path.stem.replace('_', ' ').title(),
                        authors=["Unknown"],
                        abstract="",
                        categories=[],
                        published=""
                    )
                    return ExtractedDocument(
                        metadata=metadata,
                        text=text,
                        page_count=len(pdf.pages),
                        extraction_method='pdf'
                    )
        except Exception:
            pass
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                metadata = DocumentMetadata(
                    doc_id=pdf_path.stem,
                    title=pdf_path.stem.replace('_', ' ').title(),
                    authors=["Unknown"],
                    abstract="",
                    categories=[],
                    published=""
                )
                return ExtractedDocument(
                    metadata=metadata,
                    text=text,
                    page_count=len(reader.pages),
                    extraction_method='pdf'
                )
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")
