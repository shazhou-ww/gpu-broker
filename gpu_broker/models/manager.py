"""Model manager for downloading and managing ML models."""
import hashlib
import json
import logging
import shutil
import sqlite3
import time
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
    
    def download(self, url: str, model_type: str = 'checkpoint') -> dict:
        """Download a model from URL, auto-detect source.
        
        Supported URLs:
            - https://huggingface.co/<org>/<repo>
            - https://huggingface.co/<org>/<repo>/blob/main/<file>
            - https://civitai.com/models/<id>
            - https://civitai.com/api/download/models/<id>
        
        Args:
            url: Model URL
            model_type: 'checkpoint' or 'lora'
            
        Returns:
            Model info dictionary
        """
        if model_type not in ('checkpoint', 'lora'):
            raise ValueError(f"Invalid model type: {model_type}. Must be 'checkpoint' or 'lora'")
        
        source = self._detect_source(url)
        if source == 'huggingface':
            return self._download_huggingface(url, model_type=model_type)
        elif source == 'civitai':
            return self._download_civitai(url, model_type=model_type)
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
    
    def _download_huggingface(self, url: str, model_type: str = 'checkpoint') -> dict:
        """Download a model from HuggingFace Hub.
        
        Args:
            url: HuggingFace URL (repo or specific file)
            model_type: 'checkpoint' or 'lora'
            
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
                'type': model_type,
                'trigger_words': None,
                'pulled_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }
            
            self._upsert_model(model_info)
            
            logger.info(f"Successfully downloaded {repo_id} ({size_bytes} bytes) id={short_id} type={model_type}")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {e}")
            raise RuntimeError(f"Download failed: {e}")
    
    # ── Civitai download ──────────────────────────────────────────────
    
    def _download_civitai(self, url: str, filename: Optional[str] = None,
                          model_type: str = 'checkpoint') -> dict:
        """Download a model from Civitai.
        
        Args:
            url: Civitai URL (page or direct download)
            filename: Custom filename (optional)
            model_type: 'checkpoint' or 'lora'
            
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
                'type': model_type,
                'trigger_words': None,
                'pulled_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }
            
            self._upsert_model(model_info)
            
            logger.info(f"Successfully downloaded {filename} ({size_bytes} bytes) id={short_id} type={model_type}")
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
    
    # ── Local model registration ─────────────────────────────────────

    def add_local(self, path: str, name: Optional[str] = None,
                  lookup: bool = False, strategy: str = 'symlink',
                  model_type: str = 'checkpoint') -> dict:
        """Register a local model file/directory.

        Args:
            path: Path to model file (.safetensors/.ckpt) or directory (diffusers format)
            name: Custom name, auto-detect if None
            lookup: Whether to lookup metadata from Civitai by hash
            strategy: 'symlink' (default), 'copy', or 'move'
            model_type: 'checkpoint' or 'lora'

        Returns:
            dict with model info including short_id

        Raises:
            FileNotFoundError: path doesn't exist
            ValueError: unrecognized format or model already registered
        """
        if model_type not in ('checkpoint', 'lora'):
            raise ValueError(f"Invalid model type: {model_type}. Must be 'checkpoint' or 'lora'")
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Determine format
        if resolved.is_dir():
            if not (resolved / 'model_index.json').exists():
                raise ValueError(
                    f"Unrecognized model format: directory '{resolved}' "
                    "has no model_index.json (expected diffusers format)"
                )
            model_format = 'diffusers'
        elif resolved.is_file():
            ext = resolved.suffix.lower()
            if ext not in ('.safetensors', '.ckpt'):
                raise ValueError(
                    f"Unrecognized model format: '{resolved.name}'. "
                    "Supported extensions: .safetensors, .ckpt"
                )
            model_format = 'safetensors'
        else:
            raise ValueError(f"Unrecognized model format: {resolved}")

        # Compute SHA256
        try:
            full_hash = self._compute_sha256(resolved)
        except Exception as e:
            raise RuntimeError(f"Failed to compute SHA256: {e}")

        short_id = self._short_id(full_hash)

        # Check if already registered
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT id FROM models WHERE sha256 = ?", (full_hash,)
            )
            existing = cursor.fetchone()
            if existing:
                raise ValueError(
                    f"Model already registered as {existing['id']} "
                    f"(sha256={full_hash[:12]}...)"
                )

        # Determine name
        if name is None:
            if resolved.is_dir():
                name = resolved.name
            else:
                name = resolved.stem  # filename without extension

        # Civitai lookup (only for single files)
        source = 'local'
        source_url = None
        trigger_words = None
        if lookup and resolved.is_file():
            civitai_info = self._lookup_civitai(full_hash)
            if civitai_info:
                source = 'civitai'
                source_url = civitai_info.get('source_url')
                trigger_words = civitai_info.get('trigger_words')
                if not name or name == resolved.stem:
                    # Use Civitai name if user didn't specify one
                    civitai_name = civitai_info.get('name')
                    if civitai_name:
                        name = civitai_name

        # Execute file strategy
        if resolved.is_dir():
            target_name = short_id
        else:
            target_name = short_id + resolved.suffix
        target_path = self.models_dir / target_name

        try:
            if strategy == 'symlink':
                os.symlink(resolved, target_path)
            elif strategy == 'copy':
                if resolved.is_dir():
                    shutil.copytree(resolved, target_path)
                else:
                    shutil.copy2(resolved, target_path)
            elif strategy == 'move':
                shutil.move(str(resolved), str(target_path))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        except OSError as e:
            raise OSError(
                f"Failed to {strategy} '{resolved}' → '{target_path}': {e}"
            )

        size_bytes = self._calculate_size(
            target_path if strategy != 'symlink' else resolved
        )

        model_info = {
            'id': short_id,
            'sha256': full_hash,
            'name': name,
            'source': source,
            'source_url': source_url,
            'path': str(target_path),
            'format': model_format,
            'size_bytes': size_bytes,
            'type': model_type,
            'trigger_words': trigger_words,
            'pulled_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        }

        self._upsert_model(model_info)

        logger.info(
            f"Registered local model '{name}' id={short_id} "
            f"strategy={strategy} format={model_format} type={model_type}"
        )
        return model_info

    def _lookup_civitai(self, sha256: str) -> Optional[dict]:
        """Lookup model info from Civitai by SHA256 hash.

        Returns:
            dict with 'name', 'source_url', 'trigger_words' if found
            None if not found or API error
        """
        try:
            resp = requests.get(
                f"https://civitai.com/api/v1/model-versions/by-hash/{sha256}",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'name': data.get('model', {}).get('name', ''),
                    'source_url': f"https://civitai.com/models/{data.get('modelId', '')}",
                    'trigger_words': ', '.join(data.get('trainedWords', [])),
                }
            return None
        except Exception:
            return None

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

        # Deserialize tags: JSON string → list
        raw_tags = d.get('tags', '') or ''
        try:
            d['tags'] = json.loads(raw_tags) if raw_tags else []
        except Exception:
            d['tags'] = [t.strip() for t in raw_tags.split(',') if t.strip()] if raw_tags else []

        # Deserialize trigger_words: JSON string or comma-separated → list
        raw_tw = d.get('trigger_words', '') or ''
        try:
            parsed = json.loads(raw_tw) if raw_tw else []
            d['trigger_words'] = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            d['trigger_words'] = [t.strip() for t in raw_tw.split(',') if t.strip()] if raw_tw else []

        # nsfw: int → bool
        d['nsfw'] = bool(d.get('nsfw', 0))

        # Defaults for new fields if missing (old rows without migration)
        d.setdefault('description', '')
        d.setdefault('base_model', '')
        d.setdefault('recommended_cfg', 7.0)
        d.setdefault('recommended_steps', 20)
        d.setdefault('civitai_id', 0)

        return d
    
    # ── List / Get / Delete ───────────────────────────────────────────
    
    def list(self, model_type: Optional[str] = None, tag: Optional[str] = None,
             base_model: Optional[str] = None, nsfw: Optional[bool] = None,
             search: Optional[str] = None) -> list[dict]:
        """List all downloaded models.

        Args:
            model_type: Filter by type ('checkpoint' or 'lora'). None for all.
            tag: Comma-separated tags; each tag must match (AND logic).
            base_model: Filter by base model name.
            nsfw: True → only NSFW, False → only SFW, None → all.
            search: Search in name, description, and tags.
        """
        conditions = []
        params = []

        if model_type:
            conditions.append('type = ?')
            params.append(model_type)

        if tag:
            for t in [t.strip() for t in tag.split(',') if t.strip()]:
                conditions.append("tags LIKE ?")
                params.append(f'%{t}%')

        if base_model is not None:
            conditions.append('base_model = ?')
            params.append(base_model)

        if nsfw is True:
            conditions.append('nsfw = 1')
        elif nsfw is False:
            conditions.append('nsfw = 0')

        if search:
            conditions.append(
                "(name LIKE ? OR description LIKE ? OR tags LIKE ?)"
            )
            q = f'%{search}%'
            params.extend([q, q, q])

        where_clause = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''

        with self._get_db() as conn:
            cursor = conn.execute(f"""
                SELECT *
                FROM models
                {where_clause}
                ORDER BY pulled_at DESC
            """, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def enrich(self, cminfo_dir: str = '/mnt/e/ComfyUI/models/checkpoints',
               use_civitai: bool = False) -> dict:
        """Enrich model metadata from .cminfo.json files and optionally Civitai API.

        Args:
            cminfo_dir: Directory containing *.cminfo.json files.
            use_civitai: If True, query Civitai API for models without civitai_id.

        Returns:
            dict with keys: updated, skipped, not_found
        """
        stats = {'updated': 0, 'skipped': 0, 'not_found': 0}
        cminfo_path = Path(cminfo_dir)

        # ── Phase 1: scan .cminfo.json files ──────────────────────────────
        if cminfo_path.is_dir():
            for cminfo_file in cminfo_path.glob('*.cminfo.json'):
                try:
                    with open(cminfo_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read {cminfo_file}: {e}")
                    stats['skipped'] += 1
                    continue

                # Extract SHA256 from Hashes.SHA256 (may be uppercase)
                sha256 = (data.get('Hashes', {}) or {}).get('SHA256', '')
                if not sha256:
                    stats['skipped'] += 1
                    continue
                sha256 = sha256.lower()

                # Find matching model in DB
                with self._get_db() as conn:
                    cursor = conn.execute(
                        'SELECT id FROM models WHERE sha256 = ?', (sha256,)
                    )
                    row = cursor.fetchone()

                if not row:
                    logger.debug(f"No model found for sha256={sha256[:12]} ({cminfo_file.name})")
                    stats['not_found'] += 1
                    continue

                model_id = row['id']

                # Build update fields
                description = data.get('ModelName', '') or ''
                tags_list = data.get('Tags', []) or []
                base_model = data.get('BaseModel', '') or ''
                nsfw_val = 1 if data.get('Nsfw', False) else 0
                civitai_id = data.get('ModelId', 0) or 0
                trained_words = data.get('TrainedWords', []) or []

                with self._get_db() as conn:
                    conn.execute("""
                        UPDATE models
                        SET description = ?,
                            tags = ?,
                            base_model = ?,
                            nsfw = ?,
                            civitai_id = ?,
                            trigger_words = ?,
                            updated_at = ?
                        WHERE id = ?
                    """, (
                        description,
                        json.dumps(tags_list),
                        base_model,
                        nsfw_val,
                        civitai_id,
                        json.dumps(trained_words),
                        datetime.now().isoformat(),
                        model_id,
                    ))
                    conn.commit()

                logger.info(f"Enriched model {model_id} from {cminfo_file.name}")
                stats['updated'] += 1

        # ── Phase 2: Civitai API for unmatched models ─────────────────────
        if use_civitai:
            civitai_token = os.environ.get('CIVITAI_API_TOKEN', '')
            headers = {}
            if civitai_token:
                headers['Authorization'] = f'Bearer {civitai_token}'

            with self._get_db() as conn:
                cursor = conn.execute(
                    "SELECT id, sha256 FROM models WHERE civitai_id = 0 AND length(sha256) > 12"
                )
                candidates = cursor.fetchall()

            for row in candidates:
                model_id = row['id']
                sha256 = row['sha256']
                try:
                    resp = requests.get(
                        f'https://civitai.com/api/v1/model-versions/by-hash/{sha256}',
                        headers=headers,
                        timeout=10,
                    )
                    if resp.status_code != 200:
                        stats['skipped'] += 1
                        time.sleep(0.5)
                        continue

                    cv = resp.json()
                    description = (cv.get('model', {}) or {}).get('name', '') or ''
                    tags_list = (cv.get('model', {}) or {}).get('tags', []) or []
                    base_model = cv.get('baseModel', '') or ''
                    nsfw_val = 1 if (cv.get('model', {}) or {}).get('nsfw', False) else 0
                    civitai_id = cv.get('modelId', 0) or 0
                    trained_words = cv.get('trainedWords', []) or []

                    with self._get_db() as conn:
                        conn.execute("""
                            UPDATE models
                            SET description = ?,
                                tags = ?,
                                base_model = ?,
                                nsfw = ?,
                                civitai_id = ?,
                                trigger_words = ?,
                                updated_at = ?
                            WHERE id = ?
                        """, (
                            description,
                            json.dumps(tags_list),
                            base_model,
                            nsfw_val,
                            civitai_id,
                            json.dumps(trained_words),
                            datetime.now().isoformat(),
                            model_id,
                        ))
                        conn.commit()

                    logger.info(f"Enriched model {model_id} from Civitai (civitai_id={civitai_id})")
                    stats['updated'] += 1

                except Exception as e:
                    logger.warning(f"Civitai lookup failed for {model_id}: {e}")
                    stats['skipped'] += 1

                time.sleep(0.5)

        return stats

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
