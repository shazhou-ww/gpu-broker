"""Model manager for downloading and managing ML models."""
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Optional
import os
import requests
from datetime import datetime

# Conditional imports for optional GPU dependencies
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloads and storage."""
    
    def __init__(self, db_path: Path, models_dir: Path):
        """Initialize the model manager.
        
        Args:
            db_path: Path to SQLite database
            models_dir: Directory to store downloaded models
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelManager initialized with models_dir={models_dir}")
    
    def _get_db(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _calculate_size(self, path: Path) -> int:
        """Recursively calculate directory or file size.
        
        Args:
            path: Path to file or directory
            
        Returns:
            Total size in bytes
        """
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
            return total
        return 0
    
    def pull(self, source: str, repo_id: Optional[str] = None, 
             url: Optional[str] = None, filename: Optional[str] = None) -> dict:
        """Download a model from HuggingFace or Civitai.
        
        Args:
            source: 'huggingface' or 'civitai'
            repo_id: HuggingFace repository ID (e.g., 'runwayml/stable-diffusion-v1-5')
            url: Civitai download URL
            filename: Custom filename for Civitai models
            
        Returns:
            Model info dictionary
            
        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If download fails
        """
        if source == 'huggingface':
            if not repo_id:
                raise ValueError("repo_id is required for HuggingFace models")
            return self._pull_huggingface(repo_id)
        elif source == 'civitai':
            if not url:
                raise ValueError("url is required for Civitai models")
            return self._pull_civitai(url, filename)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _pull_huggingface(self, repo_id: str) -> dict:
        """Download a model from HuggingFace Hub.
        
        Args:
            repo_id: Repository ID (e.g., 'runwayml/stable-diffusion-v1-5')
            
        Returns:
            Model info dictionary
        """
        if not HF_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is not available. "
                "Install GPU dependencies with: pip install gpu-broker[gpu]"
            )
        
        model_id = repo_id.replace('/', '_')
        local_path = self.models_dir / model_id
        
        logger.info(f"Downloading HuggingFace model: {repo_id}")
        
        try:
            # Download the model using snapshot_download
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False
            )
            
            size_bytes = self._calculate_size(local_path)
            
            # Register in database
            model_info = {
                'id': model_id,
                'name': repo_id.split('/')[-1],
                'source': 'huggingface',
                'source_url': f"https://huggingface.co/{repo_id}",
                'path': str(local_path),
                'format': 'diffusers',
                'size_bytes': size_bytes,
                'type': 'checkpoint',
                'trigger_words': None,
                'pulled_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            with self._get_db() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO models 
                    (id, name, source, source_url, path, format, size_bytes, type, trigger_words, pulled_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_info['id'], model_info['name'], model_info['source'],
                    model_info['source_url'], model_info['path'], model_info['format'],
                    model_info['size_bytes'], model_info['type'], model_info['trigger_words'],
                    model_info['pulled_at'], model_info['updated_at']
                ))
                conn.commit()
            
            logger.info(f"Successfully downloaded {repo_id} ({size_bytes} bytes)")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {e}")
            raise RuntimeError(f"Download failed: {e}")
    
    def _pull_civitai(self, url: str, filename: Optional[str] = None) -> dict:
        """Download a model from Civitai.
        
        Args:
            url: Civitai download URL
            filename: Custom filename (optional, will be inferred from response)
            
        Returns:
            Model info dictionary
        """
        if not filename:
            # Try to extract filename from URL or Content-Disposition header
            filename = url.split('/')[-1].split('?')[0]
            if not filename.endswith('.safetensors'):
                filename = f"model_{uuid.uuid4().hex[:8]}.safetensors"
        
        if not filename.endswith('.safetensors'):
            filename = f"{filename}.safetensors"
        
        local_path = self.models_dir / filename
        
        logger.info(f"Downloading Civitai model from: {url}")
        
        try:
            # Download the file
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Try to get filename from Content-Disposition if not provided
            if 'Content-Disposition' in response.headers:
                content_disp = response.headers['Content-Disposition']
                if 'filename=' in content_disp:
                    suggested_name = content_disp.split('filename=')[-1].strip('"\'')
                    if suggested_name.endswith('.safetensors'):
                        filename = suggested_name
                        local_path = self.models_dir / filename
            
            # Write file in chunks
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size_bytes = local_path.stat().st_size
            model_id = local_path.stem
            
            # Register in database
            model_info = {
                'id': model_id,
                'name': filename,
                'source': 'civitai',
                'source_url': url,
                'path': str(local_path),
                'format': 'safetensors',
                'size_bytes': size_bytes,
                'type': 'checkpoint',
                'trigger_words': None,
                'pulled_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            with self._get_db() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO models 
                    (id, name, source, source_url, path, format, size_bytes, type, trigger_words, pulled_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_info['id'], model_info['name'], model_info['source'],
                    model_info['source_url'], model_info['path'], model_info['format'],
                    model_info['size_bytes'], model_info['type'], model_info['trigger_words'],
                    model_info['pulled_at'], model_info['updated_at']
                ))
                conn.commit()
            
            logger.info(f"Successfully downloaded {filename} ({size_bytes} bytes)")
            return model_info
            
        except requests.RequestException as e:
            logger.error(f"Failed to download from {url}: {e}")
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Download failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Download failed: {e}")
    
    def list(self) -> list[dict]:
        """List all downloaded models.
        
        Returns:
            List of model info dictionaries
        """
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT id, name, source, source_url, path, format, size_bytes, 
                       type, trigger_words, pulled_at
                FROM models
                ORDER BY pulled_at DESC
            """)
            
            models = []
            for row in cursor.fetchall():
                models.append({
                    'id': row['id'],
                    'name': row['name'],
                    'source': row['source'],
                    'source_url': row['source_url'],
                    'path': row['path'],
                    'format': row['format'],
                    'size_bytes': row['size_bytes'],
                    'type': row['type'],
                    'trigger_words': row['trigger_words'],
                    'pulled_at': row['pulled_at'],
                    'loaded': False  # Runtime status, always False for now
                })
            
            return models
    
    def get(self, model_id: str) -> Optional[dict]:
        """Get a single model's information.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model info dictionary or None if not found
        """
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT id, name, source, source_url, path, format, size_bytes,
                       type, trigger_words, pulled_at
                FROM models
                WHERE id = ?
            """, (model_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'id': row['id'],
                'name': row['name'],
                'source': row['source'],
                'source_url': row['source_url'],
                'path': row['path'],
                'format': row['format'],
                'size_bytes': row['size_bytes'],
                'type': row['type'],
                'trigger_words': row['trigger_words'],
                'pulled_at': row['pulled_at'],
                'loaded': False
            }
    
    def delete(self, model_id: str) -> bool:
        """Delete a model from storage.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful, False if model not found
        """
        with self._get_db() as conn:
            # Get model info first
            cursor = conn.execute("SELECT path FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            model_path = Path(row['path'])
            
            # Delete from database
            conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
            conn.commit()
            
            # Delete files
            try:
                if model_path.is_dir():
                    import shutil
                    shutil.rmtree(model_path)
                elif model_path.is_file():
                    model_path.unlink()
                logger.info(f"Deleted model {model_id} from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to delete model files at {model_path}: {e}")
            
            return True
