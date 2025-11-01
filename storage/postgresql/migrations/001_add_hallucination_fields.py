"""
Database Migration: Add Hallucination Detection Fields to QueryLog
Migration ID: 001
Date: 2024
Description: Adds hallucination detection, validation, and LLM metadata fields to query_logs table
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
import os

logger = logging.getLogger(__name__)


class Migration001:
    """
    Migration to add hallucination detection fields to QueryLog
    
    This migration is idempotent and can be run multiple times safely
    """
    
    MIGRATION_ID = "001"
    MIGRATION_NAME = "add_hallucination_fields"
    DESCRIPTION = "Add hallucination detection and LLM metadata fields to query_logs"
    
    # SQL statements for migration
    UP_STATEMENTS = [
        # Hallucination detection fields
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS hallucination_score FLOAT;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS hallucination_risk_level VARCHAR(20);
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS is_hallucination BOOLEAN DEFAULT FALSE;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS quality_score FLOAT;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS validation_confidence FLOAT;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS sla_certificate JSON;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS validation_warnings JSON;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS sources_used JSON;
        """,
        
        # LLM metadata fields
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS llm_backend VARCHAR(50);
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS llm_model VARCHAR(100);
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS completion_tokens INTEGER;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS total_tokens INTEGER;
        """,
        """
        ALTER TABLE query_logs 
        ADD COLUMN IF NOT EXISTS regeneration_attempts INTEGER DEFAULT 0;
        """,
        
        # Create indexes for common queries
        """
        CREATE INDEX IF NOT EXISTS idx_query_logs_risk_level 
        ON query_logs(hallucination_risk_level);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_query_logs_quality_score 
        ON query_logs(quality_score);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_query_logs_hallucination 
        ON query_logs(is_hallucination);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_query_logs_backend 
        ON query_logs(llm_backend);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_query_logs_created_at 
        ON query_logs(created_at);
        """
    ]
    
    # SQL statements for rollback (optional)
    DOWN_STATEMENTS = [
        # Drop indexes
        "DROP INDEX IF EXISTS idx_query_logs_risk_level;",
        "DROP INDEX IF EXISTS idx_query_logs_quality_score;",
        "DROP INDEX IF EXISTS idx_query_logs_hallucination;",
        "DROP INDEX IF EXISTS idx_query_logs_backend;",
        "DROP INDEX IF EXISTS idx_query_logs_created_at;",
        
        # Drop columns (Note: BE CAREFUL with this in production)
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS hallucination_score;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS hallucination_risk_level;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS is_hallucination;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS quality_score;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS validation_confidence;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS sla_certificate;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS validation_warnings;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS sources_used;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS llm_backend;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS llm_model;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS prompt_tokens;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS completion_tokens;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS total_tokens;",
        "ALTER TABLE query_logs DROP COLUMN IF EXISTS regeneration_attempts;"
    ]
    
    def __init__(self, engine: Optional[AsyncEngine] = None):
        """
        Initialize migration
        
        Args:
            engine: SQLAlchemy async engine (creates from env if None)
        """
        self.engine = engine or self._create_engine()
    
    def _create_engine(self) -> AsyncEngine:
        """Create database engine from environment variables"""
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        db_name = os.getenv("POSTGRES_DB", "advanced_rag")
        
        database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
        
        return create_async_engine(
            database_url,
            echo=False,
            pool_size=5,
            max_overflow=10
        )
    
    async def up(self) -> None:
        """
        Apply migration (add new columns and indexes)
        
        This is idempotent - can be run multiple times safely
        """
        logger.info(f"Running migration {self.MIGRATION_ID}: {self.MIGRATION_NAME}")
        
        try:
            async with self.engine.begin() as conn:
                # Check if migration was already applied
                if await self._is_migration_applied(conn):
                    logger.info(f"Migration {self.MIGRATION_ID} already applied, skipping")
                    return
                
                # Execute migration statements
                for i, statement in enumerate(self.UP_STATEMENTS, 1):
                    try:
                        logger.debug(f"Executing statement {i}/{len(self.UP_STATEMENTS)}")
                        await conn.execute(text(statement))
                    except Exception as e:
                        logger.error(f"Error executing statement {i}: {e}")
                        logger.error(f"Statement: {statement}")
                        raise
                
                # Record migration
                await self._record_migration(conn)
                
            logger.info(f"Migration {self.MIGRATION_ID} completed successfully")
            
        except Exception as e:
            logger.error(f"Migration {self.MIGRATION_ID} failed: {e}")
            raise
    
    async def down(self) -> None:
        """
        Rollback migration (remove columns and indexes)
        
        WARNING: This will delete data! Use with caution in production.
        """
        logger.warning(f"Rolling back migration {self.MIGRATION_ID}: {self.MIGRATION_NAME}")
        logger.warning("This will delete data from query_logs table!")
        
        try:
            async with self.engine.begin() as conn:
                # Execute rollback statements
                for i, statement in enumerate(self.DOWN_STATEMENTS, 1):
                    try:
                        logger.debug(f"Executing rollback statement {i}/{len(self.DOWN_STATEMENTS)}")
                        await conn.execute(text(statement))
                    except Exception as e:
                        logger.error(f"Error executing rollback statement {i}: {e}")
                        logger.error(f"Statement: {statement}")
                        # Continue with other statements
                
                # Remove migration record
                await self._remove_migration_record(conn)
                
            logger.info(f"Migration {self.MIGRATION_ID} rolled back successfully")
            
        except Exception as e:
            logger.error(f"Migration rollback {self.MIGRATION_ID} failed: {e}")
            raise
    
    async def _is_migration_applied(self, conn) -> bool:
        """
        Check if migration was already applied
        
        Args:
            conn: Database connection
            
        Returns:
            True if migration was applied
        """
        # First, ensure migrations table exists
        await self._ensure_migrations_table(conn)
        
        # Check if this migration is recorded
        result = await conn.execute(
            text("SELECT COUNT(*) FROM migrations WHERE migration_id = :migration_id"),
            {"migration_id": self.MIGRATION_ID}
        )
        count = result.scalar()
        
        return count > 0
    
    async def _ensure_migrations_table(self, conn) -> None:
        """
        Ensure migrations tracking table exists
        
        Args:
            conn: Database connection
        """
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS migrations (
                id SERIAL PRIMARY KEY,
                migration_id VARCHAR(50) UNIQUE NOT NULL,
                migration_name VARCHAR(255) NOT NULL,
                description TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
    
    async def _record_migration(self, conn) -> None:
        """
        Record migration as applied
        
        Args:
            conn: Database connection
        """
        await conn.execute(
            text("""
                INSERT INTO migrations (migration_id, migration_name, description)
                VALUES (:migration_id, :migration_name, :description)
                ON CONFLICT (migration_id) DO NOTHING
            """),
            {
                "migration_id": self.MIGRATION_ID,
                "migration_name": self.MIGRATION_NAME,
                "description": self.DESCRIPTION
            }
        )
    
    async def _remove_migration_record(self, conn) -> None:
        """
        Remove migration record
        
        Args:
            conn: Database connection
        """
        await conn.execute(
            text("DELETE FROM migrations WHERE migration_id = :migration_id"),
            {"migration_id": self.MIGRATION_ID}
        )
    
    async def status(self) -> dict:
        """
        Get migration status
        
        Returns:
            Dict with migration status information
        """
        try:
            async with self.engine.begin() as conn:
                applied = await self._is_migration_applied(conn)
                
                return {
                    "migration_id": self.MIGRATION_ID,
                    "migration_name": self.MIGRATION_NAME,
                    "description": self.DESCRIPTION,
                    "applied": applied,
                    "statements_count": len(self.UP_STATEMENTS)
                }
        except Exception as e:
            return {
                "migration_id": self.MIGRATION_ID,
                "error": str(e)
            }


async def run_migration(engine: Optional[AsyncEngine] = None):
    """
    Run the migration
    
    Args:
        engine: Optional database engine
    """
    migration = Migration001(engine)
    await migration.up()


async def rollback_migration(engine: Optional[AsyncEngine] = None):
    """
    Rollback the migration
    
    Args:
        engine: Optional database engine
    """
    migration = Migration001(engine)
    await migration.down()


async def check_migration_status(engine: Optional[AsyncEngine] = None):
    """
    Check migration status
    
    Args:
        engine: Optional database engine
        
    Returns:
        Dict with status information
    """
    migration = Migration001(engine)
    return await migration.status()


# CLI interface for running migration
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def main():
        command = sys.argv[1] if len(sys.argv) > 1 else "up"
        
        if command == "up":
            print("Running migration...")
            await run_migration()
            print("Migration completed successfully!")
            
        elif command == "down":
            print("WARNING: This will rollback the migration and DELETE DATA!")
            response = input("Are you sure? Type 'yes' to confirm: ")
            
            if response.lower() == "yes":
                print("Rolling back migration...")
                await rollback_migration()
                print("Rollback completed successfully!")
            else:
                print("Rollback cancelled")
                
        elif command == "status":
            print("Checking migration status...")
            status = await check_migration_status()
            print(f"Migration {status['migration_id']}: {status['migration_name']}")
            print(f"Applied: {status.get('applied', 'Unknown')}")
            print(f"Description: {status.get('description', 'N/A')}")
            
        else:
            print(f"Unknown command: {command}")
            print("Usage: python 001_add_hallucination_fields.py [up|down|status]")
            sys.exit(1)
    
    asyncio.run(main())

