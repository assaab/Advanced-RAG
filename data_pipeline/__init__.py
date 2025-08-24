"""
Data Pipeline Module
Handles PDF ingestion, processing, and chunk creation with parent/child hierarchy.
"""

from .ingestion.pdf_loader import PDFLoader
from .processing.chunker import DocumentChunker
from .processing.metadata_extractor import MetadataExtractor

__all__ = ["PDFLoader", "DocumentChunker", "MetadataExtractor"]
