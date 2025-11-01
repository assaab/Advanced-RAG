"""
PostgreSQL Data Models
SQLAlchemy models for document and chunk storage
"""
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON, ForeignKey, Boolean
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
    """Query logging for analytics and feedback with hallucination detection"""
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
    
    # Hallucination detection (NEW - Week 3)
    hallucination_score = Column(Float)  # Risk score 0.0 to 1.0
    hallucination_risk_level = Column(String(20))  # low, medium, high, critical
    is_hallucination = Column(Boolean, default=False)  # Whether hallucination detected
    quality_score = Column(Float)  # Overall quality score 0.0 to 1.0
    validation_confidence = Column(Float)  # Confidence in validation 0.0 to 1.0
    sla_certificate = Column(JSON)  # Optional SLA certificate data
    validation_warnings = Column(JSON)  # List of validation warnings
    sources_used = Column(JSON)  # List of source documents used
    
    # LLM metadata (NEW - Week 3)
    llm_backend = Column(String(50))  # openai, ollama, anthropic
    llm_model = Column(String(100))  # Model name used
    prompt_tokens = Column(Integer)  # Tokens in prompt
    completion_tokens = Column(Integer)  # Tokens in completion
    total_tokens = Column(Integer)  # Total tokens used
    regeneration_attempts = Column(Integer, default=0)  # Number of regeneration attempts
    
    # User feedback
    user_rating = Column(Integer)  # 1-5 scale
    user_feedback = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "query_text": self.query_text,
            "user_id": self.user_id,
            "final_answer": self.final_answer,
            "confidence_score": self.confidence_score,
            
            # Performance metrics
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
            
            # Hallucination metrics
            "hallucination_score": self.hallucination_score,
            "hallucination_risk_level": self.hallucination_risk_level,
            "is_hallucination": self.is_hallucination,
            "quality_score": self.quality_score,
            "validation_confidence": self.validation_confidence,
            "validation_warnings": self.validation_warnings,
            "sources_used": self.sources_used,
            
            # LLM metadata
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "total_tokens": self.total_tokens,
            "regeneration_attempts": self.regeneration_attempts,
            
            # User feedback
            "user_rating": self.user_rating,
            "user_feedback": self.user_feedback,
            
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk assessment summary"""
        return {
            "hallucination_score": self.hallucination_score,
            "risk_level": self.hallucination_risk_level,
            "is_hallucination": self.is_hallucination,
            "quality_score": self.quality_score,
            "confidence": self.validation_confidence,
            "warnings_count": len(self.validation_warnings) if self.validation_warnings else 0
        }
    
    def is_high_quality(self) -> bool:
        """Check if query result is high quality"""
        if self.quality_score is None or self.hallucination_score is None:
            return False
        
        # High quality if good quality score and low hallucination risk
        return (
            self.quality_score >= 0.7 and
            self.hallucination_score <= 0.3 and
            self.hallucination_risk_level in ["low", "medium"]
        )
