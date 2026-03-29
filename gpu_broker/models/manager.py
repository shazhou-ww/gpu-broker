"""Model manager for downloading and managing ML models."""
import hashlib
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import os
import requests
from datetime import datetime

# Conditional imports for optional GPU dependencies
try:
    from huggingface_hub import snapshot_download, hf_hub_download
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
    
    # ── SHA256 ID System ──────────────────────────────────────────────
    
    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 hash of a file or directory.
        
        For directories (diffusers format): hash the concatenation of
        relative paths and file sizes for all files in sorted order.
        For single files (safetensors): hash the file content directly.
        """
        if path.is_file():
            sha256 = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        elif path.is_dir():
            sha256 = hashlib.sha256()
            for file_path in sorted(path.rglob('*')):
                if file_path.is_file():
                    sha256.update(str(file_path.relative_to(path)).encode())
                    sha256.update(str(file_path.stat().st_size).encode())
            return sha256.hexdigest()
        return hashlib.sha256(b'empty').hexdigest()
    
    def _short_id(self, full_hash: str) -> str:
        """Get short ID (first 12 chars) from full hash."""
        return full_hash[:12]
    
    # ── URL Detection & Parsing ───────────────────────────────────────
    
    def _detect_source(self, url: str) -> str:
        """Auto-detect model source from URL.
        
        Returns:
            'huggingface' or 'civitai'
            
        Raises:
            ValueError: If URL hostname is not recognized
        """
        parsed = urlparse(url)
        hostname = parsed.hostname or ''
        
        if 'huggingface.co' in hostname or 'hf.co' in hostname:
            return 'huggingface'
        elif 'civitai.com' in hostname:
            return 'civitai'
        else:
            raise ValueError(f"Unknown model source: {hostname}. Supported: huggingface.co, civitai.com")
    
    def _parse_hf_url(self, url: str) -> tuple[str, Optional[str]]:
        """Parse HuggingFace URL to extract repo_id and optional filename.
        
        Examples:
            https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
              → ('stabilityai/stable-diffusion-xl-base-1.0', None)
            https://huggingface.co/stabilityai/sdxl/blob/main/sd_xl_base_1.0.safetensors
              → ('stabilityai/sdxl', 'sd_xl_base_1.0.safetensors')
        """
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            raise ValueError(f"Invalid HuggingFace URL: {url}")
        
        org, repo = path_parts[0], path_parts[1]
        repo_id = f"{org}/{repo}"
        
        # Check for specific file reference
        filename = None
        if len(path_parts) > 3 and path_parts[2] in ('blob', 'resolve'):
            # e.g., /org/repo/blob/main/model.safetensors
            filename = '/'.join(path_parts[4:]) if len(path_parts) > 4 else None
        
        return repo_id, filename
    
    def _parse_civitai_url(self, url: str) -> str:
        """Parse Civitai URL to get download URL.
        
        Examples:
            https://civitai.com/models/12345 → API download URL
            https://civitai.com/api/download/models/12345 → use directly
        """
        parsed = urlparse(url)
        
        if '/api/download/' in parsed.path:
            return url  # Already a direct download URL
        
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) >= 2 and path_parts[0] == 'models':
            model_id = path_parts[1].split('?')[0]
            return f"https://civitai.com/api/download/models/{model_id}"
        
        raise ValueError(f"Cannot parse Civitai URL: {url}")
    
    # ── Download (new unified entry point) ────────────────────────────
    
    def download(self, url: str) -> dict:
        """Download a model from URL, auto-detect source.
        
        Supported URLs:
            - https://huggingface.co/<org>/<repo>
            - https://huggingface.co/<org>/<repo>/blob/main/<file>
            - https://civitai.com/models/<id>
            - https://civitai.com/api/download/models/<id>
            
        Returns:
            Model info dictionary
        """
        source = self._detect_source(url)
        if source == 'huggingface':
            return self._download_huggingface(url)
        elif source == 'civitai':
            return self._download_civitai(url)
        else:
            raise ValueError(f"Unsupported URL: {url}. Supported: huggingface.co, civitai.com")
    
    # ── Legacy pull() — kept for backward compatibility ───────────────
    
    def pull(self, source: str, repo_id: Optional[str] = None,
             url: Optional[str] = None, filename: Optional[str] = None) -> dict:
        """Download a model (legacy interface).
        
        Prefer download(url) for new code.
        """
        if source == 'huggingface':
            if not repo_id:
                raise ValueError("repo_id is required for HuggingFace models")
            hf_url = f"https://huggingface.co/{repo_id}"
            return self._download_huggingface(hf_url)
        elif source == 'civitai':
            if not url:
                raise ValueError("url is required for Civitai models")
            return self._download_civitai(url, filename)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    # ── HuggingFace download ──────────────────────────────────────────
    
    def _download_huggingface(self, url: str) -> dict:
        """Download a model from HuggingFace Hub.
        
        Args:
            url: HuggingFace URL (repo or specific file)
            
        Returns:
            Model info dictionary
        """
        if not HF_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is not available. "
                "Install GPU dependencies with: pip install gpu-broker[gpu]"
            )
        
        repo_id, filename = self._parse_hf_url(url)
        model_dir_name = repo_id.replace('/', '_')
        local_path = self.models_dir / model_dir_name
        
        logger.info(f"Downloading HuggingFace model: {repo_id}" +
                     (f" file={filename}" if filename else ""))
        
        try:
            if filename:
                # Download single file
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(local_path),
                )
                # For single file, the actual file path
                actual_path = local_path / filename
                model_format = 'safetensors' if filename.endswith('.safetensors') else 'diffusers'
            else:
                # Download entire repo
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_path),
                    local_dir_use_symlinks=False,
                )
                actual_path = local_path
                model_format = 'diffusers'
            
            size_bytes = self._calculate_size(local_path)
            full_hash = self._compute_sha256(actual_path)
            short_id = self._short_id(full_hash)
            
            model_info = {
                'id': short_id,
                'sha256': full_hash,
                'name': repo_id.split('/')[-1],
                'source': 'huggingface',
                'source_url': url,
                'path': str(local_path),
                'format': model_format,
                'size_bytes': size_bytes,
                'type': 'checkpoint',
                'trigger_words': None,
                'pulled_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }
            
            self._upsert_model(model_info)
            
            logger.info(f"Successfully downloaded {repo_id} ({size_bytes} bytes) id={short_id}")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {e}")
            raise RuntimeError(f"Download failed: {e}")
    
    # ── Civitai download ──────────────────────────────────────────────
    
    def _download_civitai(self, url: str, filename: Optional[str] = None) -> dict:
        """Download a model from Civitai.
        
        Args:
            url: Civitai URL (page or direct download)
            filename: Custom filename (optional)
            
        Returns:
            Model info dictionary
        """
        download_url = self._parse_civitai_url(url)
        
        if not filename:
            filename = download_url.split('/')[-1].split('?')[0]
            if not filename.endswith('.safetensors'):
                filename = f"model_{uuid.uuid4().hex[:8]}.safetensors"
        
        if not filename.endswith('.safetensors'):
            filename = f"{filename}.safetensors"
        
        local_path = self.models_dir / filename
        
        logger.info(f"Downloading Civitai model from: {download_url}")
        
        try:
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Try to get filename from Content-Disposition
            if 'Content-Disposition' in response.headers:
                content_disp = response.headers['Content-Disposition']
                if 'filename=' in content_disp:
                    suggested_name = content_disp.split('filename=')[-1].strip('"\'')
                    if suggested_name.endswith('.safetensors'):
                        filename = suggested_name
                        local_path = self.models_dir / filename
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size_bytes = local_path.stat().st_size
            full_hash = self._compute_sha256(local_path)
            short_id = self._short_id(full_hash)
            
            model_info = {
                'id': short_id,
                'sha256': full_hash,
                'name': filename,
                'source': 'civitai',
                'source_url': url,
                'path': str(local_path),
                'format': 'safetensors',
                'size_bytes': size_bytes,
                'type': 'checkpoint',
                'trigger_words': None,
                'pulled_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }
            
            self._upsert_model(model_info)
            
            logger.info(f"Successfully downloaded {filename} ({size_bytes} bytes) id={short_id}")
            return model_info
            
        except requests.RequestException as e:
            logger.error(f"Failed to download from {download_url}: {e}")
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Download failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading {download_url}: {e}")
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Download failed: {e}")
    
    # ── DB helpers ────────────────────────────────────────────────────
    
    def _upsert_model(self, model_info: dict) -> None:
        """Insert or replace a model record in the database."""
        with self._get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models
                (id, sha256, name, source, source_url, path, format,
                 size_bytes, type, trigger_words, pulled_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_info['id'], model_info['sha256'],
                model_info['name'], model_info['source'],
                model_info['source_url'], model_info['path'],
                model_info['format'], model_info['size_bytes'],
                model_info['type'], model_info['trigger_words'],
                model_info['pulled_at'], model_info['updated_at'],
            ))
            conn.commit()
    
    # ── Model resolution (short ID / prefix / name) ──────────────────
    
    def resolve_id(self, query: str) -> Optional[dict]:
        """Resolve a model by short ID, full hash, or name prefix.
        
        Priority:
            1. Exact match on id (short hash)
            2. Prefix match on sha256 (full hash)
            3. Name contains match (case-insensitive)
            
        Returns:
            Model info dict or None
            
        Raises:
            ValueError: If query matches multiple models (ambiguous)
        """
        with self._get_db() as conn:
            # 1. Exact short ID match
            cursor = conn.execute("SELECT * FROM models WHERE id = ?", (query,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            
            # 2. SHA256 prefix match
            cursor = conn.execute(
                "SELECT * FROM models WHERE sha256 LIKE ?", (query + '%',)
            )
            rows = cursor.fetchall()
            if len(rows) == 1:
                return self._row_to_dict(rows[0])
            elif len(rows) > 1:
                raise ValueError(
                    f"Ambiguous ID prefix '{query}', matches {len(rows)} models"
                )
            
            # 3. Name fuzzy match (case-insensitive contains)
            cursor = conn.execute(
                "SELECT * FROM models WHERE LOWER(name) LIKE ?",
                (f'%{query.lower()}%',)
            )
            rows = cursor.fetchall()
            if len(rows) == 1:
                return self._row_to_dict(rows[0])
            elif len(rows) > 1:
                raise ValueError(
                    f"Ambiguous name '{query}', matches {len(rows)} models"
                )
            
            return None
    
    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict with standard model fields."""
        d = dict(row)
        # Ensure sha256 key exists (backward compat with old rows)
        if 'sha256' not in d:
            d['sha256'] = d.get('id', '')
        d['loaded'] = False  # Runtime status
        return d
    
    # ── List / Get / Delete ───────────────────────────────────────────
    
    def list(self) -> list[dict]:
        """List all downloaded models."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT id, sha256, name, source, source_url, path, format,
                       size_bytes, type, trigger_words, pulled_at
                FROM models
                ORDER BY pulled_at DESC
            """)
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get(self, model_id: str) -> Optional[dict]:
        """Get a single model's information.
        
        Uses resolve_id for flexible matching.
        """
        try:
            return self.resolve_id(model_id)
        except ValueError:
            # Ambiguous — fall back to exact match only
            with self._get_db() as conn:
                cursor = conn.execute(
                    "SELECT * FROM models WHERE id = ?", (model_id,)
                )
                row = cursor.fetchone()
                return self._row_to_dict(row) if row else None
    
    def delete(self, model_id: str) -> bool:
        """Delete a model from storage.
        
        Uses resolve_id for flexible matching.
        """
        model = self.get(model_id)
        if not model:
            return False
        
        actual_id = model['id']
        model_path = Path(model['path'])
        
        with self._get_db() as conn:
            conn.execute("DELETE FROM models WHERE id = ?", (actual_id,))
            conn.commit()
        
        try:
            if model_path.is_dir():
                import shutil
                shutil.rmtree(model_path)
            elif model_path.is_file():
                model_path.unlink()
            logger.info(f"Deleted model {actual_id} from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to delete model files at {model_path}: {e}")
        
        return True
