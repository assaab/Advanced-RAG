"""
PostgreSQL Data Models
SQLAlchemy models for document and chunk storage
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class Document(Base):
    """Document metadata storage"""
    __tablename__ = "documents"
    
    id = Column(String(255), primary_key=True)
    title = Column(Text, nullable=False)
    authors = Column(JSON)  # List of authors
    abstract = Column(Text)
    categories = Column(JSON)  # List of categories
    published = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Enhanced metadata
    keywords = Column(JSON)  # List of keywords
    sections = Column(JSON)  # List of section titles
    references_count = Column(Integer, default=0)
    figures_count = Column(Integer, default=0)
    tables_count = Column(Integer, default=0)
    
    # Processing status
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    text_extraction_method = Column(String(50))  # pdf, ocr
    
    # Optional fields
    arxiv_id = Column(String(50), unique=True)
    page_count = Column(Integer)
    file_path = Column(String(500))
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "categories": self.categories,
            "published": self.published.isoformat() if self.published else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "keywords": self.keywords,
            "processing_status": self.processing_status,
            "arxiv_id": self.arxiv_id,
            "page_count": self.page_count
        }


class DocumentChunk(Base):
    """Document chunk storage with parent/child hierarchy"""
    __tablename__ = "document_chunks"
    
    id = Column(String(255), primary_key=True)  # chunk_id
    document_id = Column(String(255), ForeignKey("documents.id"), nullable=False)
    parent_chunk_id = Column(String(255))  # References parent chunk
    
    # Chunk content
    text = Column(Text, nullable=False)
    chunk_type = Column(String(20), nullable=False)  # 'parent' or 'child'
    position = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    
    # Section information
    section_title = Column(String(500))
    
    # Embedding metadata
    embedding_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    embedding_model = Column(String(100))  # Name of embedding model used
    vector_dimensions = Column(Integer)  # Number of embedding dimensions
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "parent_chunk_id": self.parent_chunk_id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,  # Truncate for API
            "chunk_type": self.chunk_type,
            "position": self.position,
            "token_count": self.token_count,
            "section_title": self.section_title,
            "embedding_status": self.embedding_status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class QueryLog(Base):
    """Query logging for analytics and feedback"""
    __tablename__ = "query_logs"
    
    id = Column(String(255), primary_key=True)
    query_text = Column(Text, nullable=False)
    user_id = Column(String(255))
    
    # Results
    retrieved_chunks = Column(JSON)  # List of chunk IDs
    final_answer = Column(Text)
    confidence_score = Column(Integer)  # 1-5 scale
    
    # Performance metrics
    retrieval_time_ms = Column(Integer)
    generation_time_ms = Column(Integer)
    total_time_ms = Column(Integer)
    
    # User feedback
    user_rating = Column(Integer)  # 1-5 scale
    user_feedback = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query_text": self.query_text,
            "confidence_score": self.confidence_score,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
            "user_rating": self.user_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
