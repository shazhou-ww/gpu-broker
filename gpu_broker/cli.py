"""Command-line interface for GPU Broker.

All output is JSON for easy agent/script parsing.
"""
import json
import logging
import os
import signal
import subprocess
import sys
import time

import click
import httpx

from gpu_broker import __version__
from gpu_broker.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DATA_DIR,
    DB_PATH,
    MODELS_DIR,
    LOG_LEVEL,
    load_config,
    set_config,
)

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_MODEL_NOT_FOUND = 2
EXIT_DAEMON_NOT_RUNNING = 3
EXIT_TIMEOUT = 4
EXIT_VRAM_ERROR = 5

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def output_json(data):
    """Emit JSON to stdout."""
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))


def output_error(message: str, code: str = "error"):
    """Emit error JSON to stderr."""
    click.echo(
        json.dumps({"error": code, "message": message}, ensure_ascii=False),
        err=True,
    )


def _daemon_url(host: str = "localhost", port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def _is_process_running(pid: int) -> bool:
    """Check whether a PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_pid_file():
    """Return (pid: int | None, pid_file: Path)."""
    pid_file = DATA_DIR / "daemon.pid"
    if not pid_file.exists():
        return None, pid_file
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None, pid_file
    return pid, pid_file


def _load_port_from_config() -> int:
    """Best-effort: read port from config for status/stop commands."""
    try:
        cfg = load_config()
        return int(cfg.get("port", DEFAULT_PORT))
    except Exception:
        return DEFAULT_PORT


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version__)
def cli():
    """GPU Broker – GPU inference task broker."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# ============================= daemon =====================================

@cli.group()
def daemon():
    """Daemon lifecycle management."""
    pass


@daemon.command(name="start")
@click.option("--host", default=None, help="Host to bind to (default from config)")
@click.option("--port", default=None, type=int, help="Port to bind to (default from config)")
@click.option("--foreground", is_flag=True, help="Run in foreground (for debugging)")
def daemon_start(host, port, foreground):
    """Start the GPU Broker daemon."""
    cfg = load_config()
    host = host or cfg.get("host", DEFAULT_HOST)
    port = port or int(cfg.get("port", DEFAULT_PORT))

    if foreground:
        # Run directly (blocks) – useful for debugging
        import uvicorn

        uvicorn.run(
            "gpu_broker.api.app:create_app",
            host=host,
            port=port,
            factory=True,
            log_level=LOG_LEVEL.lower(),
        )
        return

    # --- background mode ---
    pid_file = DATA_DIR / "daemon.pid"
    if pid_file.exists():
        existing_pid = int(pid_file.read_text().strip())
        if _is_process_running(existing_pid):
            output_json({"error": "Daemon already running", "pid": existing_pid})
            sys.exit(EXIT_ERROR)
        # Stale PID file – remove
        pid_file.unlink(missing_ok=True)

    log_path = DATA_DIR / "daemon.log"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpu_broker.daemon_main",
            "--host",
            str(host),
            "--port",
            str(port),
        ],
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Write PID immediately (daemon_main will overwrite with its own once it
    # starts, but we need this so that *stop* can target the right process even
    # before uvicorn's event loop kicks in).
    pid_file.write_text(str(proc.pid))

    output_json(
        {"status": "started", "pid": proc.pid, "host": host, "port": port}
    )


@daemon.command(name="stop")
def daemon_stop():
    """Stop the running daemon."""
    pid, pid_file = _read_pid_file()
    if pid is None or not _is_process_running(pid):
        output_error("Daemon not running (no PID file or process dead)")
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)
        sys.exit(EXIT_DAEMON_NOT_RUNNING)

    os.kill(pid, signal.SIGTERM)
    pid_file.unlink(missing_ok=True)
    output_json({"status": "stopped", "pid": pid})


@daemon.command(name="status")
def daemon_status():
    """Show daemon status (PID + HTTP health-check)."""
    pid, pid_file = _read_pid_file()
    port = _load_port_from_config()

    if pid is None:
        output_json({"status": "stopped"})
        sys.exit(EXIT_DAEMON_NOT_RUNNING)

    if not _is_process_running(pid):
        output_json(
            {"status": "stopped", "note": "PID file exists but process is dead"}
        )
        pid_file.unlink(missing_ok=True)
        sys.exit(EXIT_DAEMON_NOT_RUNNING)

    # PID is alive – try HTTP health-check for richer info
    try:
        resp = httpx.get(f"http://localhost:{port}/v1/status", timeout=3)
        data = resp.json()
        data["pid"] = pid
        output_json(data)
    except Exception:
        output_json(
            {
                "status": "starting",
                "pid": pid,
                "note": "Process alive but HTTP not responding yet",
            }
        )


# ============================= model ======================================

@cli.group()
def model():
    """Model management commands."""
    pass


@model.command(name="download")
@click.argument("url")
@click.option("--name", default=None, help="Custom filename override")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def model_download(url: str, name: str, host: str, port: int):
    """Download a model from a URL (HuggingFace or Civitai auto-detected)."""
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/models/download"

    # New unified download endpoint: just send the URL
    data = {"url": url}

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(api_url, json=data)
            response.raise_for_status()
            output_json(response.json())
    except httpx.ConnectError:
        output_error(f"Cannot connect to daemon at {_daemon_url(host, port)}")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        output_error(detail)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="list")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def model_list(host: str, port: int):
    """List available models."""
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/models"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url)
            response.raise_for_status()
            output_json(response.json())
    except httpx.ConnectError:
        output_error(f"Cannot connect to daemon at {_daemon_url(host, port)}")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="remove")
@click.argument("model_id")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def model_remove(model_id: str, host: str, port: int):
    """Remove a model by ID (supports SHA256 short-ID and name fuzzy match)."""
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/models/{model_id}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.delete(api_url)
            if response.status_code == 404:
                output_error(f"Model '{model_id}' not found")
                sys.exit(EXIT_MODEL_NOT_FOUND)
            response.raise_for_status()
            output_json(response.json())
    except httpx.ConnectError:
        output_error(f"Cannot connect to daemon at {_daemon_url(host, port)}")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            output_error(f"Model '{model_id}' not found")
            sys.exit(EXIT_MODEL_NOT_FOUND)
        detail = e.response.json().get("detail", str(e))
        output_error(detail)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="info")
@click.argument("model_id")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def model_info(model_id: str, host: str, port: int):
    """Show model details."""
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/models/{model_id}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url)
            if response.status_code == 404:
                output_error(f"Model '{model_id}' not found")
                sys.exit(EXIT_MODEL_NOT_FOUND)
            response.raise_for_status()
            output_json(response.json())
    except httpx.ConnectError:
        output_error(f"Cannot connect to daemon at {_daemon_url(host, port)}")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            output_error(f"Model '{model_id}' not found")
            sys.exit(EXIT_MODEL_NOT_FOUND)
        detail = e.response.json().get("detail", str(e))
        output_error(detail)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="add")
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", default=None, help="Custom model name")
@click.option("--lookup", is_flag=True, help="Lookup model info from Civitai by hash")
@click.option("--copy", "strategy", flag_value="copy", help="Copy to models directory")
@click.option("--move", "strategy", flag_value="move", help="Move to models directory")
def model_add(path, name, lookup, strategy):
    """Register a local model file or directory.

    \b
    Supports:
      - .safetensors / .ckpt files
      - Diffusers directories (with model_index.json)

    \b
    File strategy (default: symlink):
      --copy   Copy file/directory to models directory
      --move   Move file/directory to models directory

    \b
    Examples:
      gpu-broker model add /path/to/model.safetensors
      gpu-broker model add /path/to/model.safetensors --name "my-model"
      gpu-broker model add /path/to/model.safetensors --lookup
      gpu-broker model add /path/to/model.safetensors --copy
    """
    from gpu_broker.models.manager import ModelManager

    if strategy is None:
        strategy = "symlink"

    manager = ModelManager(DB_PATH, MODELS_DIR)

    try:
        result = manager.add_local(
            path=path,
            name=name,
            lookup=lookup,
            strategy=strategy,
        )
        output_json(result)
    except FileNotFoundError as e:
        output_error(str(e), code="path_not_found")
        sys.exit(EXIT_ERROR)
    except ValueError as e:
        msg = str(e)
        if "already registered" in msg.lower():
            output_error(msg, code="already_registered")
        else:
            output_error(msg, code="invalid_format")
        sys.exit(EXIT_ERROR)
    except OSError as e:
        output_error(str(e), code="file_operation_failed")
        sys.exit(EXIT_ERROR)
    except RuntimeError as e:
        output_error(str(e), code="sha256_failed")
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


# ============================= task =======================================

@cli.group()
def task():
    """Task management commands."""
    pass


@task.command(name="submit")
@click.argument("json_input", required=False)
@click.option("--wait", is_flag=True, help="Block until task completes")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def task_submit(json_input, wait, host, port):
    """Submit a task with a JSON payload.

    Accepts JSON as a positional argument or via stdin.

    \b
    Examples:
      gpu-broker task submit '{"type":"txt2img","model_id":"abc","params":{"prompt":"a cat"}}'
      echo '{"type":"txt2img",...}' | gpu-broker task submit
      gpu-broker task submit --wait '{"type":"txt2img",...}'
    """
    port = port or _load_port_from_config()

    # Resolve JSON input
    if json_input is None:
        if sys.stdin.isatty():
            output_error("No JSON input provided. Pass as argument or pipe via stdin.")
            sys.exit(EXIT_ERROR)
        json_input = sys.stdin.read()

    try:
        data = json.loads(json_input)
    except json.JSONDecodeError as e:
        output_error(f"Invalid JSON: {e}")
        sys.exit(EXIT_ERROR)

    api_url = f"{_daemon_url(host, port)}/v1/tasks"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(api_url, json=data)
            response.raise_for_status()
            result = response.json()
            task_id = result.get("task_id")

            if not wait:
                output_json(result)
                return

            # Polling loop
            status_url = f"{api_url}/{task_id}"
            while True:
                time.sleep(2)
                status_resp = client.get(status_url)
                status_resp.raise_for_status()
                task_info = status_resp.json()
                st = task_info.get("status")

                if st in ("completed", "failed", "cancelled"):
                    output_json(task_info)
                    if st == "failed":
                        sys.exit(EXIT_ERROR)
                    return

    except httpx.ConnectError:
        output_error(f"Cannot connect to daemon at {_daemon_url(host, port)}")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        output_error(detail)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@task.command(name="status")
@click.argument("task_id")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def task_status(task_id: str, host: str, port: int):
    """Get status of a task."""
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/tasks/{task_id}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url)
            if response.status_code == 404:
                output_error(f"Task '{task_id}' not found")
                sys.exit(EXIT_ERROR)
            response.raise_for_status()
            output_json(response.json())
    except httpx.ConnectError:
        output_error(f"Cannot connect to daemon at {_daemon_url(host, port)}")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@task.command(name="list")
@click.option("--status", "filter_status", default=None, help="Filter by status")
@click.option("--model", "model_id", default=None, help="Filter by model ID")
@click.option("--limit", default=20, type=int, help="Max results")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def task_list(filter_status, model_id, limit, host, port):
    """List tasks with optional filters."""
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/tasks"
    params = {"limit": limit}
    if filter_status:
        params["status"] = filter_status
    if model_id:
        params["model_id"] = model_id

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url, params=params)
            response.raise_for_status()
            output_json(response.json())
    except httpx.ConnectError:
        output_error(f"Cannot connect to daemon at {_daemon_url(host, port)}")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@task.command(name="cancel")
@click.argument("task_id")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def task_cancel(task_id: str, host: str, port: int):
    """Cancel a pending task."""
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/tasks/{task_id}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.delete(api_url)
            if response.status_code == 404:
                output_error(f"Task '{task_id}' not found or not cancellable")
                sys.exit(EXIT_ERROR)
            response.raise_for_status()
            output_json(response.json())
    except httpx.ConnectError:
        output_error(f"Cannot connect to daemon at {_daemon_url(host, port)}")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        output_error(detail)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


# ============================= config =====================================

@cli.group(name="config")
def config_group():
    """Configuration management."""
    pass


@config_group.command(name="show")
def config_show():
    """Display current configuration."""
    cfg = load_config()
    # Mask sensitive values
    display = dict(cfg)
    for key in ("hf_token", "civitai_key"):
        val = display.get(key, "")
        if val:
            display[key] = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"
    output_json(display)


@config_group.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value.

    \b
    Examples:
      gpu-broker config set port 8080
      gpu-broker config set output_dir /data/outputs
      gpu-broker config set hf_token hf_xxx
    """
    try:
        updated = set_config(key, value)
        output_json({"status": "ok", "key": key, "value": updated[key]})
    except (ValueError, TypeError) as e:
        output_error(f"Invalid value for '{key}': {e}")
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
