"""Configuration for GPU Broker."""
import os
import json
from pathlib import Path

# Defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7878
LOG_LEVEL = os.getenv("GPU_BROKER_LOG_LEVEL", "INFO")

# Data directories
DATA_DIR = Path(os.getenv("GPU_BROKER_DATA_DIR", Path.home() / ".gpu-broker"))
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "gpu-broker.db"
CONFIG_FILE = Path.home() / ".config" / "gpu-broker" / "config.json"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "host": DEFAULT_HOST,
    "port": DEFAULT_PORT,
    "output_dir": str(OUTPUTS_DIR),
    "default_steps": 20,
    "default_width": 1024,
    "default_height": 1024,
    "default_cfg_scale": 7.0,
    "hf_token": os.getenv("HF_TOKEN", ""),
    "civitai_key": os.getenv("CIVITAI_API_KEY", ""),
}


def load_config() -> dict:
    """Load config from file, merge with defaults. Env vars override."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            file_config = json.load(f)
        config.update(file_config)

    # Env vars override
    if os.getenv("HF_TOKEN"):
        config["hf_token"] = os.getenv("HF_TOKEN")
    if os.getenv("CIVITAI_API_KEY"):
        config["civitai_key"] = os.getenv("CIVITAI_API_KEY")

    return config


def save_config(config: dict) -> None:
    """Save config to file."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def set_config(key: str, value: str) -> dict:
    """Set a single config value with automatic type conversion."""
    config = load_config()
    # Type conversion based on known keys
    if key in ("port", "default_steps", "default_width", "default_height"):
        value = int(value)
    elif key in ("default_cfg_scale",):
        value = float(value)
    config[key] = value
    save_config(config)
    return config
