"""Daemon entry point for background execution."""
import argparse
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="GPU Broker daemon process")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7878)
    args = parser.parse_args()

    # Write PID file (overwrite what the parent wrote with actual daemon PID)
    data_dir = Path(os.getenv("GPU_BROKER_DATA_DIR", Path.home() / ".gpu-broker"))
    data_dir.mkdir(parents=True, exist_ok=True)
    pid_file = data_dir / "daemon.pid"
    pid_file.write_text(str(os.getpid()))

    try:
        uvicorn.run(
            "gpu_broker.api.app:create_app",
            host=args.host,
            port=args.port,
            factory=True,
            log_level="info",
        )
    finally:
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
