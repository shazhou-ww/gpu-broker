"""Task scheduler for managing inference queue."""
import logging
import asyncio
import aiosqlite
import json
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Async task queue with persistent state."""
    
    def __init__(self, db_path: Path, engine, model_manager):
        """Initialize the task scheduler.
        
        Args:
            db_path: Path to SQLite database
            engine: InferenceEngine instance
            model_manager: ModelManager instance
        """
        self.db_path = db_path
        self._engine = engine
        self._model_manager = model_manager
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        logger.info("TaskScheduler initialized")
    
    async def start(self) -> None:
        """Start the background worker loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Task scheduler worker started")
    
    async def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping task scheduler...")
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task scheduler stopped")
    
    async def submit(self, task_type: str, model_id: str, params: dict) -> str:
        """Submit a new task. Returns task_id.
        
        Flow:
        1. 验证 model_id 存在（查 ModelManager）
        2. 生成 task_id (tsk_ + uuid hex[:12])
        3. 插入 DB (status=pending)
        4. 放入队列
        5. 返回 task_id
        
        Args:
            task_type: Type of task ('txt2img')
            model_id: Model to use
            params: Task parameters dictionary
        
        Returns:
            task_id: Unique task identifier
        
        Raises:
            ValueError: If model not found
        """
        # 1. Validate model exists
        model = self._model_manager.get(model_id)
        if not model:
            raise ValueError(f"Model '{model_id}' not found")
        
        # 2. Generate task_id
        task_id = f"tsk_{uuid.uuid4().hex[:12]}"
        
        # 3. Insert into DB
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO tasks (id, type, model_id, params, status, created_at)
                VALUES (?, ?, ?, ?, 'pending', datetime('now'))
            """, (task_id, task_type, model_id, json.dumps(params)))
            await db.commit()
        
        # 4. Put in queue
        await self._queue.put(task_id)
        
        logger.info(f"Task {task_id} submitted (type={task_type}, model={model_id})")
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[dict]:
        """Get task info from DB.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Task info dictionary or None if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT id, type, model_id, params, status, result_path, error,
                       created_at, started_at, completed_at
                FROM tasks
                WHERE id = ?
            """, (task_id,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return {
                    'id': row['id'],
                    'type': row['type'],
                    'model_id': row['model_id'],
                    'params': row['params'],
                    'status': row['status'],
                    'result_path': row['result_path'],
                    'error': row['error'],
                    'created_at': row['created_at'],
                    'started_at': row['started_at'],
                    'completed_at': row['completed_at']
                }
    
    async def list_tasks(
        self,
        status: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List tasks with filters. Returns (tasks, total_count).
        
        Args:
            status: Filter by status (optional)
            model_id: Filter by model_id (optional)
            limit: Maximum number of results
            offset: Offset for pagination
        
        Returns:
            Tuple of (tasks list, total_count)
        """
        # Build query
        where_clauses = []
        params_list = []
        
        if status:
            where_clauses.append("status = ?")
            params_list.append(status)
        
        if model_id:
            where_clauses.append("model_id = ?")
            params_list.append(model_id)
        
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get total count
            count_query = f"SELECT COUNT(*) as count FROM tasks {where_sql}"
            async with db.execute(count_query, params_list) as cursor:
                row = await cursor.fetchone()
                total_count = row['count']
            
            # Get tasks
            query = f"""
                SELECT id, type, model_id, params, status, result_path, error,
                       created_at, started_at, completed_at
                FROM tasks
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            async with db.execute(query, params_list + [limit, offset]) as cursor:
                rows = await cursor.fetchall()
                
                tasks = []
                for row in rows:
                    tasks.append({
                        'id': row['id'],
                        'type': row['type'],
                        'model_id': row['model_id'],
                        'params': row['params'],
                        'status': row['status'],
                        'result_path': row['result_path'],
                        'error': row['error'],
                        'created_at': row['created_at'],
                        'started_at': row['started_at'],
                        'completed_at': row['completed_at']
                    })
                
                return tasks, total_count
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            True if cancelled successfully, False if not found or not pending
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Check if task exists and is pending
            async with db.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return False
                
                if row[0] != 'pending':
                    return False
            
            # Update to cancelled
            await db.execute("""
                UPDATE tasks
                SET status = 'cancelled', completed_at = datetime('now')
                WHERE id = ?
            """, (task_id,))
            await db.commit()
            
            logger.info(f"Task {task_id} cancelled")
            return True
    
    async def _worker_loop(self) -> None:
        """Background worker:
        1. 从队列取任务
        2. 更新 status=running, started_at
        3. 确保模型已加载（如需要则 load）
        4. 调用 engine.txt2img()
        5. 保存结果路径到 DB, status=completed
        6. 异常时 status=failed, error 写入 DB
        """
        logger.info("Worker loop started")
        
        while self._running:
            try:
                # Get task from queue (with timeout to allow checking _running)
                try:
                    task_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                logger.info(f"Processing task {task_id}")
                
                # Get task details from DB
                task = await self.get_task(task_id)
                if not task:
                    logger.error(f"Task {task_id} not found in DB")
                    continue
                
                # Check if already cancelled
                if task['status'] == 'cancelled':
                    logger.info(f"Task {task_id} was cancelled, skipping")
                    continue
                
                # Update to running
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE tasks
                        SET status = 'running', started_at = datetime('now')
                        WHERE id = ?
                    """, (task_id,))
                    await db.commit()
                
                try:
                    # Get model info
                    model = self._model_manager.get(task['model_id'])
                    if not model:
                        raise ValueError(f"Model {task['model_id']} not found")
                    
                    # Load model if needed
                    if self._engine.loaded_model_id != task['model_id']:
                        # Unload current model if any
                        if self._engine.loaded_model_id:
                            self._engine.unload_model()
                        
                        # Load new model
                        self._engine.load_model(
                            task['model_id'],
                            model['path'],
                            model['format']
                        )
                    
                    # Parse params and run inference
                    params = json.loads(task['params'])
                    result_path = self._engine.txt2img(params)
                    
                    # Update to completed
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("""
                            UPDATE tasks
                            SET status = 'completed', result_path = ?, completed_at = datetime('now')
                            WHERE id = ?
                        """, (result_path, task_id))
                        await db.commit()
                    
                    logger.info(f"Task {task_id} completed: {result_path}")
                
                except Exception as e:
                    # Update to failed
                    error_msg = str(e)
                    logger.error(f"Task {task_id} failed: {error_msg}")
                    
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("""
                            UPDATE tasks
                            SET status = 'failed', error = ?, completed_at = datetime('now')
                            WHERE id = ?
                        """, (error_msg, task_id))
                        await db.commit()
            
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                # Continue processing other tasks
                continue
        
        logger.info("Worker loop exited")
    
    async def get_queue_stats(self) -> dict:
        """Get queue statistics.
        
        Returns:
            Dictionary with pending, running, completed_today counts
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Pending count
            async with db.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'") as cursor:
                row = await cursor.fetchone()
                pending = row[0] if row else 0
            
            # Running count
            async with db.execute("SELECT COUNT(*) FROM tasks WHERE status = 'running'") as cursor:
                row = await cursor.fetchone()
                running = row[0] if row else 0
            
            # Completed today
            async with db.execute("""
                SELECT COUNT(*) FROM tasks 
                WHERE status = 'completed' 
                AND DATE(completed_at) = DATE('now')
            """) as cursor:
                row = await cursor.fetchone()
                completed_today = row[0] if row else 0
            
            return {
                'pending': pending,
                'running': running,
                'completed_today': completed_today
            }
