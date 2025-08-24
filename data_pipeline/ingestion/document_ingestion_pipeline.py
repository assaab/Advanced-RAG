"""
Complete Document Ingestion Pipeline
Orchestrates the entire process: PDF → Hierarchical Chunks → Multi-Vector Embeddings → Storage
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import uuid

from data_pipeline.ingestion.pdf_loader import PDFLoader, ExtractedDocument
from data_pipeline.chunking.hierarchical_chunker import HierarchicalChunker, HierarchicalChunks
from retrieval_pipeline.embeddings.multi_vector_embedder import MultiVectorEmbedder, EmbeddingResult
from storage.opensearch.client import OpenSearchClient
from storage.postgresql.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for document ingestion pipeline"""
    # Chunking settings
    parent_min_tokens: int = 500
    parent_max_tokens: int = 1000
    child_min_tokens: int = 50
    child_max_tokens: int = 150
    overlap_tokens: int = 50
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    colbert_dim: int = 128
    
    # Processing settings
    batch_size: int = 10
    max_concurrent: int = 3
    enable_ocr: bool = True
    
    # Storage settings
    opensearch_chunk_index: str = "document_chunks"
    opensearch_embedding_index: str = "chunk_embeddings"
    
    # Quality filters
    min_text_length: int = 100
    max_text_length: int = 1000000


@dataclass
class IngestionResult:
    """Result of document ingestion"""
    document_id: str
    status: str  # "success", "error", "partial"
    parent_chunks_count: int
    child_chunks_count: int
    embeddings_count: int
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0
    metadata: Dict[str, Any] = None


class DocumentIngestionPipeline:
    """
    Complete document ingestion pipeline for Advanced RAG
    
    Pipeline stages:
    1. Document loading (PDF extraction)
    2. Hierarchical chunking (parent/child structure)
    3. Multi-vector embedding (ColBERT style)
    4. Storage (OpenSearch + PostgreSQL)
    
    Supports:
    - Batch processing
    - Error handling and retry
    - Progress tracking
    - Quality filtering
    """
    
    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        db_manager: DatabaseManager,
        config: Optional[IngestionConfig] = None
    ):
        self.opensearch = opensearch_client
        self.db = db_manager
        self.config = config or IngestionConfig()
        
        # Initialize pipeline components
        self.pdf_loader = PDFLoader()
        self.chunker = HierarchicalChunker(
            parent_min_tokens=self.config.parent_min_tokens,
            parent_max_tokens=self.config.parent_max_tokens,
            child_min_tokens=self.config.child_min_tokens,
            child_max_tokens=self.config.child_max_tokens,
            overlap_tokens=self.config.overlap_tokens
        )
        self.embedder = MultiVectorEmbedder(
            model_name=self.config.embedding_model,
            colbert_dim=self.config.colbert_dim
        )
        
        logger.info(f"Document ingestion pipeline initialized")
    
    async def ingest_documents(
        self,
        sources: List[Union[Path, str, Dict[str, Any]]],
        progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """
        Ingest multiple documents
        
        Args:
            sources: List of document sources (file paths, URLs, or metadata dicts)
            progress_callback: Optional callback function for progress updates
        
        Returns:
            List of ingestion results
        """
        logger.info(f"Starting ingestion of {len(sources)} documents")
        
        # Process in batches to manage memory and resources
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        for i in range(0, len(sources), self.config.batch_size):
            batch = sources[i:i + self.config.batch_size]
            
            # Process batch concurrently
            async def process_document_with_semaphore(source):
                async with semaphore:
                    return await self.ingest_single_document(source)
            
            batch_tasks = [process_document_with_semaphore(source) for source in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Document {i + j} failed: {result}")
                    error_result = IngestionResult(
                        document_id=f"error_{i + j}",
                        status="error",
                        parent_chunks_count=0,
                        child_chunks_count=0,
                        embeddings_count=0,
                        error_message=str(result)
                    )
                    results.append(error_result)
                else:
                    results.append(result)
            
            # Progress update
            if progress_callback:
                progress_callback(len(results), len(sources))
            
            logger.info(f"Completed batch {i//self.config.batch_size + 1}: {len(results)}/{len(sources)} documents")
        
        # Summary
        successful = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "error")
        partial = sum(1 for r in results if r.status == "partial")
        
        logger.info(f"Ingestion completed: {successful} success, {partial} partial, {failed} failed")
        
        return results
    
    async def ingest_single_document(self, source: Union[Path, str, Dict[str, Any]]) -> IngestionResult:
        """
        Ingest a single document through the complete pipeline
        
        Args:
            source: Document source (file path, URL, or metadata dict)
        
        Returns:
            Ingestion result
        """
        import time
        start_time = time.time()
        
        document_id = str(uuid.uuid4())
        
        try:
            # Stage 1: Document Loading
            logger.info(f"Stage 1: Loading document {document_id}")
            extracted_doc = await self._load_document(source, document_id)
            
            # Stage 2: Quality Filtering
            logger.info(f"Stage 2: Quality filtering document {document_id}")
            if not self._passes_quality_filter(extracted_doc):
                return IngestionResult(
                    document_id=document_id,
                    status="error",
                    parent_chunks_count=0,
                    child_chunks_count=0,
                    embeddings_count=0,
                    error_message="Document failed quality filters",
                    processing_time_seconds=time.time() - start_time
                )
            
            # Stage 3: Hierarchical Chunking
            logger.info(f"Stage 3: Hierarchical chunking document {document_id}")
            hierarchical_chunks = await self._chunk_document(extracted_doc)
            
            # Stage 4: Multi-Vector Embedding
            logger.info(f"Stage 4: Embedding document {document_id}")
            embeddings = await self._embed_chunks(hierarchical_chunks)
            
            # Stage 5: Storage
            logger.info(f"Stage 5: Storing document {document_id}")
            await self._store_document_data(extracted_doc, hierarchical_chunks, embeddings)
            
            processing_time = time.time() - start_time
            
            result = IngestionResult(
                document_id=document_id,
                status="success",
                parent_chunks_count=len(hierarchical_chunks.parent_chunks),
                child_chunks_count=len(hierarchical_chunks.child_chunks),
                embeddings_count=len(embeddings),
                processing_time_seconds=processing_time,
                metadata={
                    "title": extracted_doc.metadata.title,
                    "page_count": extracted_doc.page_count,
                    "extraction_method": extracted_doc.extraction_method,
                    "total_tokens": hierarchical_chunks.total_tokens
                }
            )
            
            logger.info(f"Document {document_id} ingested successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document {document_id} ingestion failed: {e}")
            
            return IngestionResult(
                document_id=document_id,
                status="error",
                parent_chunks_count=0,
                child_chunks_count=0,
                embeddings_count=0,
                error_message=str(e),
                processing_time_seconds=processing_time
            )
    
    async def _load_document(self, source: Union[Path, str, Dict[str, Any]], document_id: str) -> ExtractedDocument:
        """Load document from various sources"""
        
        if isinstance(source, Path):
            # Local file
            if source.suffix.lower() == '.pdf':
                async with PDFLoader() as loader:
                    return loader.extract_text_from_pdf(source)
            else:
                # Plain text file
                text = source.read_text(encoding='utf-8')
                from data_pipeline.ingestion.pdf_loader import DocumentMetadata, ExtractedDocument
                metadata = DocumentMetadata(
                    doc_id=document_id,
                    title=source.stem,
                    authors=["Unknown"],
                    abstract="",
                    categories=[],
                    published=""
                )
                return ExtractedDocument(
                    metadata=metadata,
                    text=text,
                    page_count=1,
                    extraction_method='text'
                )
        
        elif isinstance(source, str):
            if source.startswith("http"):
                # URL - implement URL fetching
                raise NotImplementedError("URL fetching not yet implemented")
            elif source.startswith("arxiv:"):
                # arXiv ID - implement arXiv fetching
                arxiv_id = source.replace("arxiv:", "")
                async with PDFLoader() as loader:
                    # This would download from arXiv
                    raise NotImplementedError("arXiv fetching not yet implemented")
            else:
                # Treat as plain text
                from data_pipeline.ingestion.pdf_loader import DocumentMetadata, ExtractedDocument
                metadata = DocumentMetadata(
                    doc_id=document_id,
                    title="Text Document",
                    authors=["Unknown"],
                    abstract="",
                    categories=[],
                    published=""
                )
                return ExtractedDocument(
                    metadata=metadata,
                    text=source,
                    page_count=1,
                    extraction_method='text'
                )
        
        elif isinstance(source, dict):
            # Structured data
            from data_pipeline.ingestion.pdf_loader import DocumentMetadata, ExtractedDocument
            metadata = DocumentMetadata(
                doc_id=document_id,
                title=source.get("title", "Unknown"),
                authors=source.get("authors", ["Unknown"]),
                abstract=source.get("abstract", ""),
                categories=source.get("categories", []),
                published=source.get("published", ""),
                arxiv_id=source.get("arxiv_id")
            )
            return ExtractedDocument(
                metadata=metadata,
                text=source.get("text", ""),
                page_count=source.get("page_count", 1),
                extraction_method=source.get("extraction_method", "manual")
            )
        
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    def _passes_quality_filter(self, document: ExtractedDocument) -> bool:
        """Apply quality filters to document"""
        
        text_length = len(document.text)
        
        # Text length filters
        if text_length < self.config.min_text_length:
            logger.warning(f"Document {document.metadata.doc_id} too short: {text_length} chars")
            return False
        
        if text_length > self.config.max_text_length:
            logger.warning(f"Document {document.metadata.doc_id} too long: {text_length} chars")
            return False
        
        # Content quality checks
        word_count = len(document.text.split())
        if word_count < 50:  # Minimum word count
            logger.warning(f"Document {document.metadata.doc_id} too few words: {word_count}")
            return False
        
        # Check for mostly non-text content
        alpha_ratio = sum(c.isalpha() for c in document.text) / len(document.text)
        if alpha_ratio < 0.5:  # Less than 50% alphabetic characters
            logger.warning(f"Document {document.metadata.doc_id} low text quality: {alpha_ratio:.2f} alpha ratio")
            return False
        
        return True
    
    async def _chunk_document(self, document: ExtractedDocument) -> HierarchicalChunks:
        """Create hierarchical chunks from document"""
        
        # Prepare document metadata for chunker
        document_metadata = {
            "title": document.metadata.title,
            "authors": document.metadata.authors,
            "abstract": document.metadata.abstract
        }
        
        # Use the hierarchical chunker
        hierarchical_chunks = await self.chunker.chunk_document(
            document_id=document.metadata.doc_id,
            text=document.text,
            document_metadata=document_metadata
        )
        
        return hierarchical_chunks
    
    async def _embed_chunks(self, hierarchical_chunks: HierarchicalChunks) -> List[EmbeddingResult]:
        """Create multi-vector embeddings for child chunks"""
        
        # Only embed child chunks for search (parents are retrieved later)
        child_chunks_data = []
        for chunk in hierarchical_chunks.child_chunks:
            child_chunks_data.append({
                "id": chunk.metadata.chunk_id,
                "text": chunk.text,
                "parent_id": chunk.metadata.parent_chunk_id
            })
        
        if not child_chunks_data:
            return []
        
        # Create embeddings using multi-vector embedder
        texts = [chunk["text"] for chunk in child_chunks_data]
        chunk_ids = [chunk["id"] for chunk in child_chunks_data]
        
        embeddings = await self.embedder.embed_documents(texts, chunk_ids)
        
        return embeddings
    
    async def _store_document_data(
        self,
        document: ExtractedDocument,
        hierarchical_chunks: HierarchicalChunks,
        embeddings: List[EmbeddingResult]
    ):
        """Store document data in OpenSearch and PostgreSQL"""
        
        # Store in PostgreSQL (document metadata and chunks)
        await self._store_in_postgresql(document, hierarchical_chunks)
        
        # Store in OpenSearch (chunks and embeddings)
        await self._store_in_opensearch(hierarchical_chunks, embeddings)
    
    async def _store_in_postgresql(
        self,
        document: ExtractedDocument,
        hierarchical_chunks: HierarchicalChunks
    ):
        """Store document and chunk metadata in PostgreSQL"""
        
        async with self.db.get_session() as session:
            try:
                # Store document metadata
                doc_data = {
                    "document_id": document.metadata.doc_id,
                    "title": document.metadata.title,
                    "authors": document.metadata.authors,
                    "abstract": document.metadata.abstract,
                    "categories": document.metadata.categories,
                    "published": document.metadata.published,
                    "arxiv_id": document.metadata.arxiv_id,
                    "page_count": document.page_count,
                    "extraction_method": document.extraction_method,
                    "text_length": len(document.text),
                    "total_chunks": len(hierarchical_chunks.parent_chunks) + len(hierarchical_chunks.child_chunks)
                }
                
                # Insert document (implementation depends on your schema)
                # await session.execute(insert_document_query, doc_data)
                
                # Store parent chunks
                for chunk in hierarchical_chunks.parent_chunks:
                    chunk_data = {
                        "chunk_id": chunk.metadata.chunk_id,
                        "document_id": document.metadata.doc_id,
                        "parent_chunk_id": None,  # Parent points to itself
                        "chunk_type": "parent",
                        "text": chunk.text,
                        "token_count": chunk.metadata.token_count,
                        "position": chunk.metadata.position,
                        "section_title": chunk.metadata.section_title
                    }
                    # Insert chunk (implementation depends on your schema)
                    # await session.execute(insert_chunk_query, chunk_data)
                
                # Store child chunks
                for chunk in hierarchical_chunks.child_chunks:
                    chunk_data = {
                        "chunk_id": chunk.metadata.chunk_id,
                        "document_id": document.metadata.doc_id,
                        "parent_chunk_id": chunk.metadata.parent_chunk_id,
                        "chunk_type": "child",
                        "text": chunk.text,
                        "token_count": chunk.metadata.token_count,
                        "position": chunk.metadata.position,
                        "section_title": chunk.metadata.section_title
                    }
                    # Insert chunk (implementation depends on your schema)
                    # await session.execute(insert_chunk_query, chunk_data)
                
                # Store chunk relationships
                for parent_id, child_ids in hierarchical_chunks.chunk_relationships.items():
                    for child_id in child_ids:
                        relationship_data = {
                            "parent_chunk_id": parent_id,
                            "child_chunk_id": child_id
                        }
                        # Insert relationship (implementation depends on your schema)
                        # await session.execute(insert_relationship_query, relationship_data)
                
                await session.commit()
                logger.info(f"Stored document {document.metadata.doc_id} metadata in PostgreSQL")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store document {document.metadata.doc_id} in PostgreSQL: {e}")
                raise
    
    async def _store_in_opensearch(
        self,
        hierarchical_chunks: HierarchicalChunks,
        embeddings: List[EmbeddingResult]
    ):
        """Store chunks and embeddings in OpenSearch"""
        
        try:
            # Store parent chunks (for context retrieval)
            parent_docs = []
            for chunk in hierarchical_chunks.parent_chunks:
                doc = {
                    "chunk_id": chunk.metadata.chunk_id,
                    "document_id": chunk.metadata.document_id,
                    "chunk_type": "parent",
                    "text": chunk.text,
                    "token_count": chunk.metadata.token_count,
                    "position": chunk.metadata.position,
                    "section_title": chunk.metadata.section_title,
                    "parent_chunk_id": None
                }
                parent_docs.append(doc)
            
            if parent_docs:
                await self.opensearch.bulk_index_documents(
                    index=self.config.opensearch_chunk_index,
                    documents=parent_docs
                )
            
            # Store child chunks (for search)
            child_docs = []
            for chunk in hierarchical_chunks.child_chunks:
                doc = {
                    "chunk_id": chunk.metadata.chunk_id,
                    "document_id": chunk.metadata.document_id,
                    "chunk_type": "child",
                    "text": chunk.text,
                    "token_count": chunk.metadata.token_count,
                    "position": chunk.metadata.position,
                    "section_title": chunk.metadata.section_title,
                    "parent_chunk_id": chunk.metadata.parent_chunk_id
                }
                child_docs.append(doc)
            
            if child_docs:
                await self.opensearch.bulk_index_documents(
                    index=self.config.opensearch_chunk_index,
                    documents=child_docs
                )
            
            # Store embeddings (multi-vector)
            embedding_docs = []
            for embedding in embeddings:
                # Store each token embedding separately for late interaction
                for i, token_embedding in enumerate(embedding.token_embeddings):
                    doc = {
                        "chunk_id": embedding.chunk_id,
                        "vector_id": i,  # Token position
                        "embedding": token_embedding,
                        "model_name": embedding.model_name
                    }
                    embedding_docs.append(doc)
                
                # Also store pooled embedding for initial filtering
                pooled_doc = {
                    "chunk_id": embedding.chunk_id,
                    "vector_id": -1,  # Special ID for pooled embedding
                    "embedding": embedding.pooled_embedding,
                    "model_name": embedding.model_name,
                    "is_pooled": True
                }
                embedding_docs.append(pooled_doc)
            
            if embedding_docs:
                await self.opensearch.bulk_index_documents(
                    index=self.config.opensearch_embedding_index,
                    documents=embedding_docs
                )
            
            logger.info(f"Stored {len(hierarchical_chunks.parent_chunks)} parents, {len(hierarchical_chunks.child_chunks)} children, and {len(embedding_docs)} embeddings in OpenSearch")
            
        except Exception as e:
            logger.error(f"Failed to store in OpenSearch: {e}")
            raise
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested documents"""
        
        try:
            # Get document count from PostgreSQL
            doc_count = 0  # await self.db.get_document_count()
            
            # Get chunk counts from OpenSearch
            parent_count = 0  # await self.opensearch.get_document_count(self.config.opensearch_chunk_index, {"chunk_type": "parent"})
            child_count = 0   # await self.opensearch.get_document_count(self.config.opensearch_chunk_index, {"chunk_type": "child"})
            embedding_count = 0  # await self.opensearch.get_document_count(self.config.opensearch_embedding_index)
            
            return {
                "documents": doc_count,
                "parent_chunks": parent_count,
                "child_chunks": child_count,
                "embeddings": embedding_count,
                "embedding_model": self.config.embedding_model,
                "chunk_size_config": {
                    "parent_min": self.config.parent_min_tokens,
                    "parent_max": self.config.parent_max_tokens,
                    "child_min": self.config.child_min_tokens,
                    "child_max": self.config.child_max_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_failed_ingestion(self, document_id: str):
        """Clean up partial data from failed ingestion"""
        
        try:
            # Remove from PostgreSQL
            async with self.db.get_session() as session:
                # Delete document and related chunks
                # Implementation depends on your schema
                pass
            
            # Remove from OpenSearch
            # Delete chunks
            await self.opensearch.delete_by_query(
                index=self.config.opensearch_chunk_index,
                query={"term": {"document_id": document_id}}
            )
            
            # Delete embeddings  
            await self.opensearch.delete_by_query(
                index=self.config.opensearch_embedding_index,
                query={"term": {"document_id": document_id}}
            )
            
            logger.info(f"Cleaned up failed ingestion for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup document {document_id}: {e}")


class BatchIngestionManager:
    """Manager for large-scale batch ingestion operations"""
    
    def __init__(self, pipeline: DocumentIngestionPipeline):
        self.pipeline = pipeline
    
    async def ingest_directory(
        self,
        directory: Path,
        file_patterns: List[str] = ["*.pdf", "*.txt"],
        recursive: bool = True
    ) -> List[IngestionResult]:
        """Ingest all matching files in a directory"""
        
        files = []
        for pattern in file_patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
        
        logger.info(f"Found {len(files)} files to ingest from {directory}")
        
        return await self.pipeline.ingest_documents(files)
    
    async def ingest_from_manifest(self, manifest_path: Path) -> List[IngestionResult]:
        """Ingest documents from a manifest file (JSON/YAML list)"""
        
        import json
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        sources = []
        for entry in manifest:
            if isinstance(entry, str):
                sources.append(entry)
            elif isinstance(entry, dict):
                sources.append(entry)
            else:
                logger.warning(f"Skipping invalid manifest entry: {entry}")
        
        logger.info(f"Loaded {len(sources)} sources from manifest {manifest_path}")
        
        return await self.pipeline.ingest_documents(sources)


# Example usage and testing functions
async def example_usage():
    """Example usage of the document ingestion pipeline"""
    
    # Mock clients (replace with real implementations)
    class MockOpenSearchClient:
        async def bulk_index_documents(self, index: str, documents: List[Dict]): 
            pass
        async def delete_by_query(self, index: str, query: Dict): 
            pass
    
    class MockDatabaseManager:
        async def get_session(self):
            class MockSession:
                async def __aenter__(self): return self
                async def __aexit__(self, *args): pass
                async def commit(self): pass
                async def rollback(self): pass
            return MockSession()
    
    # Initialize pipeline
    opensearch = MockOpenSearchClient()
    db = MockDatabaseManager()
    
    config = IngestionConfig(
        parent_max_tokens=800,
        child_max_tokens=120,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    pipeline = DocumentIngestionPipeline(opensearch, db, config)
    
    # Ingest sample documents
    sources = [
        {"title": "Sample Paper", "text": "This is a sample research paper..." * 100},
        {"title": "Another Paper", "text": "This is another research paper..." * 150}
    ]
    
    results = await pipeline.ingest_documents(sources)
    
    for result in results:
        print(f"Document {result.document_id}: {result.status}")
        print(f"  Parents: {result.parent_chunks_count}, Children: {result.child_chunks_count}")
        print(f"  Processing time: {result.processing_time_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(example_usage())
