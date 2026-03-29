"""Inference engine for running ML models."""
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Try to import diffusers (optional)
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.info("diffusers not available, running in mock mode")


class InferenceEngine:
    """Thin wrapper around diffusers pipelines."""
    
    def __init__(self, outputs_dir: Path, max_cached_models: int = 1):
        """Initialize the inference engine.
        
        Args:
            outputs_dir: Directory to save generated images
            max_cached_models: Maximum number of models to keep in VRAM cache (default 1)
        """
        self._pipeline = None
        self._current_model_id: Optional[str] = None
        self._current_model_path: Optional[str] = None
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # LRU model cache: model_id -> pipeline (or None in mock mode)
        self._model_cache: OrderedDict = OrderedDict()
        self.max_cached_models: int = max_cached_models
        
        logger.info(f"InferenceEngine initialized (mock={self.is_mock}, max_cached={max_cached_models})")
    
    @property
    def is_mock(self) -> bool:
        """Whether running in mock mode (no GPU)."""
        return not DIFFUSERS_AVAILABLE
    
    @property
    def loaded_model_id(self) -> Optional[str]:
        """Get currently loaded model ID."""
        return self._current_model_id
    
    def load_model(self, model_id: str, model_path: str, model_format: str) -> None:
        """Load a model into VRAM (or mock if no GPU). Uses LRU cache.
        
        Args:
            model_id: Model identifier
            model_path: Path to the model files
            model_format: 'diffusers' or 'safetensors'
        """
        # Check cache first
        if model_id in self._model_cache:
            logger.info(f"Cache hit for model {model_id}")
            self._model_cache.move_to_end(model_id)
            self._pipeline = self._model_cache[model_id]
            self._current_model_id = model_id
            self._current_model_path = model_path
            return
        
        if self.is_mock:
            logger.info(f"Mock mode: pretending to load {model_id}")
            self._pipeline = None
            self._current_model_id = model_id
            self._current_model_path = model_path
            # Store in cache (None for mock)
            self._model_cache[model_id] = None
            self._evict_if_needed()
            return
        
        # Real loading with diffusers
        try:
            logger.info(f"Loading model {model_id} from {model_path}")
            
            if model_format == 'diffusers':
                self._pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                )
            elif model_format == 'safetensors':
                # Try SDXL first (for XL models), fallback to SD 1.5
                try:
                    logger.info("Trying StableDiffusionXLPipeline...")
                    self._pipeline = StableDiffusionXLPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16
                    )
                    logger.info("Loaded as SDXL pipeline")
                except Exception as xl_err:
                    logger.info(f"SDXL load failed ({xl_err}), falling back to SD 1.5")
                    self._pipeline = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16
                    )
                    logger.info("Loaded as SD 1.5 pipeline")
            else:
                raise ValueError(f"Unsupported model format: {model_format}")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self._pipeline = self._pipeline.to("cuda")
                logger.info(f"Model loaded to GPU")
            else:
                logger.warning("No CUDA available, using CPU (slow)")
            
            self._current_model_id = model_id
            self._current_model_path = model_path
            
            # Store in cache and evict if needed
            self._model_cache[model_id] = self._pipeline
            self._evict_if_needed()
            
            logger.info(f"Successfully loaded {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def _evict_if_needed(self) -> None:
        """Evict oldest cached model if cache exceeds max size."""
        while len(self._model_cache) > self.max_cached_models:
            evicted_id, evicted_pipeline = self._model_cache.popitem(last=False)
            logger.info(f"Evicting model {evicted_id} from cache")
            if evicted_pipeline is not None:
                del evicted_pipeline
            if not self.is_mock and DIFFUSERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("VRAM cleared after eviction")
    
    def unload_model(self) -> None:
        """Unload current model (clear active reference, keep cache intact)."""
        if self._current_model_id is not None:
            logger.info(f"Unloading active model {self._current_model_id} (cache preserved)")
        
        self._pipeline = None
        self._current_model_id = None
        self._current_model_path = None
    
    def get_status(self) -> dict:
        """Get engine status including cache info.
        
        Returns:
            Dictionary with engine status for /v1/status endpoint
        """
        return {
            'loaded_model_id': self._current_model_id,
            'is_mock': self.is_mock,
            'cached_models': list(self._model_cache.keys()),
            'max_cached_models': self.max_cached_models,
        }
    
    def txt2img(self, params: dict) -> str:
        """Run txt2img inference.
        
        Args:
            params: Dictionary with keys:
                - prompt: str
                - negative_prompt: str (optional)
                - width: int
                - height: int
                - steps: int
                - cfg_scale: float
                - seed: int (optional)
        
        Returns:
            Path to output PNG file.
        
        Raises:
            RuntimeError: If no model is loaded
        """
        if self._current_model_id is None:
            raise RuntimeError("No model loaded")
        
        # Extract params with defaults
        prompt = params.get('prompt', '')
        negative_prompt = params.get('negative_prompt', '')
        width = params.get('width', 512)
        height = params.get('height', 512)
        steps = params.get('steps', 20)
        cfg_scale = params.get('cfg_scale', 7.0)
        seed = params.get('seed')
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.outputs_dir / f"{timestamp}_{self._current_model_id}.png"
        
        if self.is_mock:
            # Generate mock placeholder image
            logger.info(f"Mock mode: generating placeholder for '{prompt}'")
            self._generate_mock_image(
                output_path, prompt, width, height, steps, cfg_scale, seed
            )
        else:
            # Real inference
            logger.info(f"Running inference: {prompt[:50]}...")
            
            # Set random seed if provided
            if seed is not None:
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
                generator.manual_seed(seed)
            else:
                generator = None
            
            # Run inference
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            )
            
            # Save image
            result.images[0].save(output_path)
            logger.info(f"Image saved to {output_path}")
        
        return str(output_path)
    
    def _generate_mock_image(
        self,
        output_path: Path,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: Optional[int]
    ) -> None:
        """Generate a mock placeholder image with params text.
        
        Args:
            output_path: Where to save the image
            prompt: User prompt
            width: Image width
            height: Image height
            steps: Inference steps
            cfg_scale: CFG scale
            seed: Random seed
        """
        # Create image with light gray background
        img = Image.new('RGB', (width, height), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fallback to default
        try:
            # Try common font paths
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()
        
        # Draw mock mode label in top-right corner
        mock_label = "MOCK MODE - No GPU"
        draw.text((width - 200, 10), mock_label, fill=(255, 0, 0), font=font_small)
        
        # Draw title
        title = "Mock Generated Image"
        draw.text((20, 20), title, fill=(0, 0, 0), font=font_large)
        
        # Draw prompt (wrapped)
        prompt_label = f"Prompt: {prompt}"
        max_width = width - 40
        
        # Simple word wrapping
        words = prompt_label.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Rough width estimation (not perfect with default font)
            if len(test_line) * 8 < max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        y_offset = 60
        for line in lines[:5]:  # Max 5 lines
            draw.text((20, y_offset), line, fill=(50, 50, 50), font=font_small)
            y_offset += 20
        
        # Draw parameters
        y_offset += 20
        params_text = [
            f"Model: {self._current_model_id}",
            f"Size: {width}x{height}",
            f"Steps: {steps}",
            f"CFG Scale: {cfg_scale}",
            f"Seed: {seed if seed is not None else 'random'}"
        ]
        
        for line in params_text:
            draw.text((20, y_offset), line, fill=(100, 100, 100), font=font_tiny)
            y_offset += 16
        
        # Draw a simple gradient or pattern to make it look less boring
        for i in range(0, width, 40):
            draw.line([(i, height - 100), (i + 20, height)], fill=(200, 200, 200), width=2)
        
        # Save
        img.save(output_path)
        logger.info(f"Mock image saved to {output_path}")
