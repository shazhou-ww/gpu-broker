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
        # Create models table — updated schema with sha256
        await db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id            TEXT PRIMARY KEY,     -- SHA256 short ID (12 chars)
                sha256        TEXT NOT NULL DEFAULT '' UNIQUE, -- Full SHA256 hash
                name          TEXT NOT NULL,
                source        TEXT NOT NULL CHECK(source IN ('huggingface', 'civitai', 'local')),
                source_url    TEXT,
                path          TEXT NOT NULL,
                format        TEXT NOT NULL CHECK(format IN ('diffusers', 'safetensors')),
                size_bytes    INTEGER NOT NULL DEFAULT 0,
                type          TEXT NOT NULL DEFAULT 'checkpoint' CHECK(type IN ('checkpoint')),
                trigger_words TEXT,
                pulled_at     TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        
        # ── Migration: add sha256 column if missing (pre-existing DB) ──
        try:
            await db.execute("SELECT sha256 FROM models LIMIT 1")
        except Exception:
            logger.info("Migrating models table: adding sha256 column")
            await db.execute(
                "ALTER TABLE models ADD COLUMN sha256 TEXT NOT NULL DEFAULT ''"
            )
            # Backfill: use existing id as sha256 placeholder
            await db.execute(
                "UPDATE models SET sha256 = id WHERE sha256 = ''"
            )
            # Try to create unique index (may fail if duplicates exist)
            try:
                await db.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_models_sha256 ON models(sha256)"
                )
            except Exception as idx_err:
                logger.warning(f"Could not create unique sha256 index: {idx_err}")
        
        # Create tasks table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id            TEXT PRIMARY KEY,
                type          TEXT NOT NULL DEFAULT 'txt2img' CHECK(type IN ('txt2img')),
                model_id      TEXT NOT NULL,
                params        TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'pending'
                              CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
                result_path   TEXT,
                error         TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                started_at    TEXT,
                completed_at  TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)
        
        # Create indexes for better query performance
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_model ON tasks(model_id)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)
        """)
        
        await db.commit()
        
    logger.info(f"Database initialized at {DB_PATH}")
