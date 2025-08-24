"""
Database Manager
Handles PostgreSQL connections and operations
"""
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import os
from storage.postgresql.models import Base, Document, DocumentChunk, QueryLog


class DatabaseManager:
    """Async PostgreSQL database manager"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_size=10,
            max_overflow=20
        )
        self.SessionLocal = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def _get_database_url(self) -> str:
        """Get database URL from environment variables"""
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        db_name = os.getenv("POSTGRES_DB", "advanced_rag")
        
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
    
    async def create_tables(self):
        """Create all database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    @asynccontextmanager
    async def get_session(self):
        """Get async database session"""
        async with self.SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception:
            return False
    
    # Document operations
    async def create_document(self, document: Document) -> Document:
        """Create a new document"""
        async with self.get_session() as session:
            session.add(document)
            await session.flush()
            await session.refresh(document)
            return document
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        async with self.get_session() as session:
            result = await session.get(Document, document_id)
            return result
    
    async def update_document_status(self, document_id: str, status: str):
        """Update document processing status"""
        async with self.get_session() as session:
            document = await session.get(Document, document_id)
            if document:
                document.processing_status = status
                await session.flush()
    
    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """List documents with pagination"""
        async with self.get_session() as session:
            result = await session.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                limit, offset
            )
            return result.all()
    
    # Chunk operations
    async def create_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Create multiple chunks"""
        async with self.get_session() as session:
            session.add_all(chunks)
            await session.flush()
            for chunk in chunks:
                await session.refresh(chunk)
            return chunks
    
    async def get_chunks_by_document(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        async with self.get_session() as session:
            result = await session.execute(
                "SELECT * FROM document_chunks WHERE document_id = $1 ORDER BY position",
                document_id
            )
            return result.all()
    
    async def get_parent_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get parent chunks for a document"""
        async with self.get_session() as session:
            result = await session.execute(
                "SELECT * FROM document_chunks WHERE document_id = $1 AND chunk_type = 'parent' ORDER BY position",
                document_id
            )
            return result.all()
    
    async def get_child_chunks_by_parent(self, parent_chunk_id: str) -> List[DocumentChunk]:
        """Get child chunks for a parent"""
        async with self.get_session() as session:
            result = await session.execute(
                "SELECT * FROM document_chunks WHERE parent_chunk_id = $1 ORDER BY position",
                parent_chunk_id
            )
            return result.all()
    
    async def update_chunk_embedding_status(self, chunk_id: str, status: str, model: str = None, dimensions: int = None):
        """Update chunk embedding status"""
        async with self.get_session() as session:
            chunk = await session.get(DocumentChunk, chunk_id)
            if chunk:
                chunk.embedding_status = status
                if model:
                    chunk.embedding_model = model
                if dimensions:
                    chunk.vector_dimensions = dimensions
                await session.flush()
    
    # Query log operations
    async def log_query(self, query_log: QueryLog) -> QueryLog:
        """Log a query for analytics"""
        async with self.get_session() as session:
            session.add(query_log)
            await session.flush()
            await session.refresh(query_log)
            return query_log
    
    async def get_query_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get query statistics"""
        async with self.get_session() as session:
            result = await session.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(total_time_ms) as avg_response_time,
                    AVG(user_rating) as avg_user_rating
                FROM query_logs 
                WHERE created_at >= NOW() - INTERVAL '%s days'
            """, days)
            row = result.fetchone()
            return {
                "total_queries": row[0] or 0,
                "avg_response_time": float(row[1]) if row[1] else 0,
                "avg_user_rating": float(row[2]) if row[2] else 0
            }
    
    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
