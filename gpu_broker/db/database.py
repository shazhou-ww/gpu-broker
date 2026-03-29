"""Database setup and initialization."""
import logging
import aiosqlite
from pathlib import Path

from gpu_broker.config import DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)


async def init_db():
    """Initialize the database with required tables."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DB_PATH) as db:
        # Create models table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('text-to-image', 'image-to-image', 'text-generation')),
                repo_id TEXT NOT NULL,
                local_path TEXT,
                status TEXT NOT NULL CHECK(status IN ('downloading', 'ready', 'error')),
                size_bytes INTEGER,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create tasks table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('text-to-image', 'image-to-image', 'text-generation')),
                status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
                priority INTEGER DEFAULT 0,
                params JSON NOT NULL,
                result JSON,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)
        
        # Create indexes for better query performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status 
            ON tasks(status, priority DESC, created_at)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_model_id 
            ON tasks(model_id)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_models_status 
            ON models(status)
        """)
        
        await db.commit()
        
    logger.info(f"Database initialized at {DB_PATH}")
