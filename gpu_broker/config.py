"""Configuration constants for GPU Broker."""
import os
from pathlib import Path

# Server configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7878

# Data directories
DATA_DIR = Path(os.getenv("GPU_BROKER_DATA_DIR", Path.home() / ".gpu-broker"))
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "gpu-broker.db"

# Logging
LOG_LEVEL = os.getenv("GPU_BROKER_LOG_LEVEL", "INFO")
