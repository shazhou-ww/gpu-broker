"""Command-line interface for GPU Broker.

Output defaults to YAML for human readability; use --json for JSON output.
"""
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

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
from gpu_broker.models.manager import ModelManager
from gpu_broker.templates.manager import TemplateManager

TEMPLATES_DIR = DATA_DIR / "templates"

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

def output_result(data, output_json: bool = False):
    """Emit data to stdout as YAML (default) or JSON."""
    if output_json:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        click.echo(yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False), nl=False)


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


def _parse_batch_lora(lora_list):
    """Parse batch file LoRA entries into API format.

    Supports:
      - Map: {"add-detail-xl": 0.5}
      - String: "add-detail-xl:0.5"
    Returns: [{"model_id": "name", "weight": float}]
    """
    result = []
    for item in lora_list:
        if isinstance(item, dict):
            for name, weight in item.items():
                result.append({"model_id": str(name), "weight": float(weight)})
        elif isinstance(item, str):
            if ":" in item:
                parts = item.rsplit(":", 1)
                try:
                    result.append({"model_id": parts[0], "weight": float(parts[1])})
                except ValueError:
                    result.append({"model_id": item, "weight": 0.8})
            else:
                result.append({"model_id": item, "weight": 0.8})
    return result


def _slugify_prompt(prompt, max_len=30):
    """Generate a filename-safe slug from a prompt string."""
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", prompt).strip("_")[:max_len]
    if not slug:
        slug = hashlib.md5(prompt.encode()).hexdigest()[:8]
    return slug + ".png"


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version__)
@click.option("--json", "output_json", is_flag=True, default=False, help="Output in JSON format")
@click.pass_context
def cli(ctx, output_json):
    """GPU Broker – GPU inference task broker."""
    ctx.ensure_object(dict)
    ctx.obj["output_json"] = output_json
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# ============================= generate (txt2img shortcut) =================

def _parse_lora(lora_strings: tuple) -> list:
    """Parse --lora "name:weight" strings into API-compatible list.

    Each string may be "name_or_id:weight" or just "name_or_id" (weight
    defaults to 0.8).
    """
    result = []
    for raw in lora_strings:
        if ":" in raw:
            parts = raw.rsplit(":", 1)
            name_or_id = parts[0]
            try:
                weight = float(parts[1])
            except ValueError:
                # Whole string is the name (contains colon in name?)
                name_or_id = raw
                weight = 0.8
        else:
            name_or_id = raw
            weight = 0.8
        result.append({"model_id": name_or_id, "weight": weight})
    return result


@cli.command(name="generate")
@click.option("--model", "-m", required=True, help="Model name or ID")
@click.option("--prompt", "-p", required=True, help="Positive prompt text")
@click.option("--negative", "-n", default="", help="Negative prompt (default: empty)")
@click.option("--width", "-W", default=1024, type=int, help="Image width (default: 1024)")
@click.option("--height", "-H", default=1024, type=int, help="Image height (default: 1024)")
@click.option("--steps", default=20, type=int, help="Sampling steps (default: 20)")
@click.option("--cfg", default=7.0, type=float, help="CFG scale (default: 7.0)")
@click.option("--seed", default=None, type=int, help="Random seed (default: random)")
@click.option("--lora", multiple=True, help='LoRA as "name_or_id:weight", repeatable')
@click.option("--output", "-o", default=None, type=click.Path(), help="Save image to this path")
@click.option("--wait", is_flag=True, help="Block until task completes")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def generate(model, prompt, negative, width, height, steps, cfg, seed,
             lora, output, wait, host, port):
    """Generate an image via txt2img (user-friendly shortcut).

    \b
    Examples:
      gpu-broker generate -m SDXLRonghua_v45 -p "a cat on the moon"
      gpu-broker generate -m mymodel -p "landscape" --steps 30 --cfg 3.5 --wait
      gpu-broker generate -m mymodel -p "girl" --lora "princess_xl:0.8" --wait -o out.png
    """
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    port = port or _load_port_from_config()

    # Build LoRA list
    lora_list = _parse_lora(lora)

    # Construct payload matching TaskSubmitRequest schema
    payload = {
        "type": "txt2img",
        "model": model,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative,
        },
        "params": {
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg,
        },
    }
    if seed is not None:
        payload["params"]["seed"] = seed
    if lora_list:
        payload["params"]["lora"] = lora_list

    api_url = f"{_daemon_url(host, port)}/v1/tasks"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            task_id = result.get("task_id")

            if not wait:
                output_result(result, use_json)
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
                    # Download image if --output specified and task completed
                    if st == "completed" and output:
                        try:
                            img_resp = client.get(f"{api_url}/{task_id}/image", timeout=60.0)
                            img_resp.raise_for_status()
                            from pathlib import Path as _Path
                            out_path = _Path(output)
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_bytes(img_resp.content)
                            task_info["output_path"] = str(out_path.resolve())
                        except Exception as e:
                            task_info["output_error"] = str(e)

                    output_result(task_info, use_json)
                    if st == "failed":
                        sys.exit(EXIT_ERROR)
                    return

    except httpx.ConnectError:
        output_error(
            "Daemon not running. Start it with: gpu-broker daemon start",
            code="daemon_not_running",
        )
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        output_error(detail)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


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
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
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
            output_result({"error": "Daemon already running", "pid": existing_pid}, use_json)
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

    output_result(
        {"status": "started", "pid": proc.pid, "host": host, "port": port}, use_json
    )


@daemon.command(name="stop")
def daemon_stop():
    """Stop the running daemon."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    pid, pid_file = _read_pid_file()
    if pid is None or not _is_process_running(pid):
        output_error("Daemon not running (no PID file or process dead)")
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)
        sys.exit(EXIT_DAEMON_NOT_RUNNING)

    os.kill(pid, signal.SIGTERM)
    pid_file.unlink(missing_ok=True)
    output_result({"status": "stopped", "pid": pid}, use_json)


@daemon.command(name="status")
def daemon_status():
    """Show daemon status (PID + HTTP health-check)."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    pid, pid_file = _read_pid_file()
    port = _load_port_from_config()

    if pid is None:
        output_result({"status": "stopped"}, use_json)
        sys.exit(EXIT_DAEMON_NOT_RUNNING)

    if not _is_process_running(pid):
        output_result(
            {"status": "stopped", "note": "PID file exists but process is dead"}, use_json
        )
        pid_file.unlink(missing_ok=True)
        sys.exit(EXIT_DAEMON_NOT_RUNNING)

    # PID is alive – try HTTP health-check for richer info
    try:
        resp = httpx.get(f"http://localhost:{port}/v1/status", timeout=3)
        data = resp.json()
        data["pid"] = pid
        output_result(data, use_json)
    except Exception:
        output_result(
            {
                "status": "starting",
                "pid": pid,
                "note": "Process alive but HTTP not responding yet",
            },
            use_json,
        )


# ============================= model ======================================

@cli.group()
def model():
    """Model management commands."""
    pass


@model.command(name="download")
@click.argument("url")
@click.option("--name", default=None, help="Custom filename override")
@click.option("--type", "model_type", default="checkpoint",
              type=click.Choice(["checkpoint", "lora"]),
              help="Model type (default: checkpoint)")
def model_download(url: str, name: str, model_type: str):
    """Download a model from a URL (HuggingFace or Civitai auto-detected)."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    manager = ModelManager(DB_PATH, MODELS_DIR)

    try:
        result = manager.download(url, model_type=model_type)
        output_result(result, use_json)
    except ValueError as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)
    except RuntimeError as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="list")
@click.option("--type", "model_type", default=None,
              type=click.Choice(["checkpoint", "lora"]),
              help="Filter by model type")
@click.option('--tag', default=None, help='Filter by tag (comma-sep for AND logic)')
@click.option('--base', 'base_model', default=None, help='Filter by base model name')
@click.option('--nsfw', 'nsfw_flag', is_flag=True, default=False, help='Only NSFW models')
@click.option('--sfw', 'sfw_flag', is_flag=True, default=False, help='Only SFW models')
@click.option('--search', default=None, help='Search in name/description/tags')
def model_list(model_type: str, tag: str, base_model: str, nsfw_flag: bool,
               sfw_flag: bool, search: str):
    """List available models."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    manager = ModelManager(DB_PATH, MODELS_DIR)

    nsfw_val = True if nsfw_flag else (False if sfw_flag else None)

    try:
        models = manager.list(model_type=model_type, tag=tag, base_model=base_model,
                              nsfw=nsfw_val, search=search)
        output_result({"models": models, "count": len(models)}, use_json)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="enrich")
@click.option('--civitai', is_flag=True, help='Fetch from Civitai API for unmatched')
@click.option('--dir', 'cminfo_dir', default='/mnt/e/ComfyUI/models/checkpoints',
              help='Directory containing .cminfo.json files')
def model_enrich(civitai: bool, cminfo_dir: str):
    """Enrich model metadata from .cminfo.json files (and optionally Civitai API)."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    manager = ModelManager(DB_PATH, MODELS_DIR)
    try:
        result = manager.enrich(cminfo_dir=cminfo_dir, use_civitai=civitai)
        output_result(result, use_json)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="remove")
@click.argument("model_id")
def model_remove(model_id: str):
    """Remove a model by ID (supports SHA256 short-ID and name fuzzy match)."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    manager = ModelManager(DB_PATH, MODELS_DIR)

    try:
        success = manager.delete(model_id)
        if not success:
            output_error(f"Model '{model_id}' not found")
            sys.exit(EXIT_MODEL_NOT_FOUND)
        output_result({"status": "deleted", "model_id": model_id}, use_json)
    except ValueError as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="info")
@click.argument("model_id")
def model_info(model_id: str):
    """Show model details."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    manager = ModelManager(DB_PATH, MODELS_DIR)

    try:
        info = manager.get(model_id)
        if not info:
            output_error(f"Model '{model_id}' not found")
            sys.exit(EXIT_MODEL_NOT_FOUND)
        output_result(info, use_json)
    except ValueError as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


@model.command(name="add")
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", default=None, help="Custom model name")
@click.option("--type", "model_type", default="checkpoint",
              type=click.Choice(["checkpoint", "lora"]),
              help="Model type (default: checkpoint)")
@click.option("--lookup", is_flag=True, help="Lookup model info from Civitai by hash")
@click.option("--copy", "strategy", flag_value="copy", help="Copy to models directory")
@click.option("--move", "strategy", flag_value="move", help="Move to models directory")
def model_add(path, name, model_type, lookup, strategy):
    """Register a local model file or directory.

    \b
    Supports:
      - .safetensors / .ckpt files
      - Diffusers directories (with model_index.json)

    \b
    Model types:
      --type checkpoint  (default) Base model checkpoint
      --type lora        LoRA adapter weights

    \b
    File strategy (default: symlink):
      --copy   Copy file/directory to models directory
      --move   Move file/directory to models directory

    \b
    Examples:
      gpu-broker model add /path/to/model.safetensors
      gpu-broker model add /path/to/model.safetensors --name "my-model"
      gpu-broker model add /path/to/lora.safetensors --type lora
      gpu-broker model add /path/to/model.safetensors --lookup
      gpu-broker model add /path/to/model.safetensors --copy
    """
    from gpu_broker.models.manager import ModelManager

    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False

    if strategy is None:
        strategy = "symlink"

    manager = ModelManager(DB_PATH, MODELS_DIR)

    try:
        result = manager.add_local(
            path=path,
            name=name,
            lookup=lookup,
            strategy=strategy,
            model_type=model_type,
        )
        output_result(result, use_json)
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
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
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
                output_result(result, use_json)
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
                    output_result(task_info, use_json)
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
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/tasks/{task_id}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url)
            if response.status_code == 404:
                output_error(f"Task '{task_id}' not found")
                sys.exit(EXIT_ERROR)
            response.raise_for_status()
            output_result(response.json(), use_json)
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
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
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
            output_result(response.json(), use_json)
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
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    port = port or _load_port_from_config()
    api_url = f"{_daemon_url(host, port)}/v1/tasks/{task_id}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.delete(api_url)
            if response.status_code == 404:
                output_error(f"Task '{task_id}' not found or not cancellable")
                sys.exit(EXIT_ERROR)
            response.raise_for_status()
            output_result(response.json(), use_json)
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
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    cfg = load_config()
    # Mask sensitive values
    display = dict(cfg)
    for key in ("hf_token", "civitai_key"):
        val = display.get(key, "")
        if val:
            display[key] = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"
    output_result(display, use_json)


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
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    try:
        updated = set_config(key, value)
        output_result({"status": "ok", "key": key, "value": updated[key]}, use_json)
    except (ValueError, TypeError) as e:
        output_error(f"Invalid value for '{key}': {e}")
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


# ============================= template ===================================

@cli.group()
def template():
    """Template management commands."""
    pass


@template.command(name="list")
@click.option("--tag", default=None, help="Filter by tag")
@click.option("--search", default=None, help="Search in name/description")
def template_list(tag, search):
    """List available templates."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    mgr = TemplateManager(TEMPLATES_DIR)
    templates = mgr.list(tag=tag, search=search)
    output_result(templates, use_json)


@template.command(name="show")
@click.argument("name")
def template_show(name):
    """Show template details."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    mgr = TemplateManager(TEMPLATES_DIR)
    tmpl = mgr.get(name)
    if not tmpl:
        output_error(f"Template '{name}' not found", code="template_not_found")
        sys.exit(EXIT_ERROR)
    output_result(tmpl, use_json)


@template.command(name="create")
@click.argument("name")
@click.option("--file", "file_path", type=click.Path(exists=True), help="YAML file path")
@click.option("--stdin", "use_stdin", is_flag=True, help="Read YAML from stdin")
def template_create(name, file_path, use_stdin):
    """Create or update a template."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    if file_path:
        content = Path(file_path).read_text()
    elif use_stdin:
        content = sys.stdin.read()
    else:
        output_error("Specify --file or --stdin")
        sys.exit(EXIT_ERROR)
    mgr = TemplateManager(TEMPLATES_DIR)
    result = mgr.create(name, content)
    output_result(result, use_json)


@template.command(name="delete")
@click.argument("name")
def template_delete(name):
    """Delete a template."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    mgr = TemplateManager(TEMPLATES_DIR)
    if not mgr.delete(name):
        output_error(f"Template '{name}' not found", code="template_not_found")
        sys.exit(EXIT_ERROR)
    output_result({"status": "deleted", "name": name}, use_json)


@template.command(name="test")
@click.argument("name")
@click.option("--prompt", default=None, help="Prompt variable")
@click.option("--var", multiple=True, help="Variable as key=value (repeatable)")
@click.option("--width", default=None, type=int, help="Width override")
@click.option("--height", default=None, type=int, help="Height override")
@click.option("--seed", default=None, type=int, help="Seed override")
def template_test(name, prompt, var, width, height, seed):
    """Test-render a template (dry run)."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    variables = {}
    if prompt:
        variables["prompt"] = prompt
    if width:
        variables["width"] = width
    if height:
        variables["height"] = height
    if seed is not None:
        variables["seed"] = seed
    for v in var:
        k, _, val = v.partition("=")
        variables[k] = val

    mgr = TemplateManager(TEMPLATES_DIR)
    is_valid, missing = mgr.validate(name, variables)
    if not is_valid:
        output_error(f"Missing required variables: {', '.join(missing)}", code="missing_variables")
        sys.exit(EXIT_ERROR)

    result = mgr.render(name, variables)
    output_result(result, use_json)


# ============================= run ========================================

@cli.command(name="run")
@click.argument("template_name")
@click.option("--prompt", default=None, help="Prompt variable")
@click.option("--var", multiple=True, help="Variable as key=value (repeatable)")
@click.option("--width", default=None, type=int)
@click.option("--height", default=None, type=int)
@click.option("--seed", default=None, type=int)
@click.option("--wait", is_flag=True, help="Wait for completion")
@click.option("--output", "-o", default=None, type=click.Path())
@click.option("--host", default="localhost")
@click.option("--port", default=None, type=int)
def run_template(template_name, prompt, var, width, height, seed, wait, output, host, port):
    """Run a template: render and submit to daemon."""
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    port = port or _load_port_from_config()

    # Build variables
    variables = {}
    if prompt:
        variables["prompt"] = prompt
    if width:
        variables["width"] = width
    if height:
        variables["height"] = height
    if seed is not None:
        variables["seed"] = seed
    for v in var:
        k, _, val = v.partition("=")
        variables[k] = val

    mgr = TemplateManager(TEMPLATES_DIR)

    # Validate
    is_valid, missing = mgr.validate(template_name, variables)
    if not is_valid:
        output_error(f"Missing required variables: {', '.join(missing)}", code="missing_variables")
        sys.exit(EXIT_ERROR)

    # Render
    try:
        task_json = mgr.render(template_name, variables)
    except ValueError as e:
        output_error(str(e), code="template_not_found")
        sys.exit(EXIT_ERROR)
    except RuntimeError as e:
        output_error(str(e), code="render_error")
        sys.exit(EXIT_ERROR)

    # Submit to daemon
    api_url = f"{_daemon_url(host, port)}/v1/tasks"
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(api_url, json=task_json)
            response.raise_for_status()
            result = response.json()
            task_id = result.get("task_id")

            if not wait:
                output_result(result, use_json)
                return

            status_url = f"{api_url}/{task_id}"
            while True:
                time.sleep(2)
                status_resp = client.get(status_url)
                status_resp.raise_for_status()
                task_info = status_resp.json()
                st = task_info.get("status")
                if st in ("completed", "failed", "cancelled"):
                    if st == "completed" and output:
                        try:
                            img_resp = client.get(f"{api_url}/{task_id}/image", timeout=60.0)
                            img_resp.raise_for_status()
                            out_path = Path(output)
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_bytes(img_resp.content)
                            task_info["output_path"] = str(out_path.resolve())
                        except Exception as e:
                            task_info["output_error"] = str(e)
                    output_result(task_info, use_json)
                    if st == "failed":
                        sys.exit(EXIT_ERROR)
                    return
    except httpx.ConnectError:
        output_error("Daemon not running. Start with: gpu-broker daemon start", code="daemon_not_running")
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        output_error(detail)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


# ============================= batch ======================================

@cli.command(name="batch")
@click.argument("batch_file", type=click.Path(exists=True))
@click.option("--wait", is_flag=True, help="Wait for all tasks to complete")
@click.option("--host", default="localhost", help="Daemon host")
@click.option("--port", default=None, type=int, help="Daemon port")
def batch(batch_file, wait, host, port):
    """Submit batch tasks from a YAML file.

    \b
    The YAML file can specify model directly or reference a template:

    Direct model:
      model: SDXLRonghua_v45
      defaults:
        steps: 25
        output_dir: ~/pictures/series
      tasks:
        - prompt: 月下汉服少女
          output: moonlight.png

    Template reference:
      template: guofeng-portrait
      defaults:
        output_dir: ~/pictures/batch
      tasks:
        - prompt: 月下汉服少女

    \b
    Examples:
      gpu-broker batch my-batch.yaml
      gpu-broker batch my-batch.yaml --wait
      gpu-broker --json batch my-batch.yaml --wait
    """
    ctx = click.get_current_context()
    use_json = ctx.obj.get("output_json", False) if ctx.obj else False
    port = port or _load_port_from_config()

    # Parse batch file
    with open(batch_file, "r", encoding="utf-8") as f:
        batch_data = yaml.safe_load(f)

    if not batch_data or "tasks" not in batch_data:
        output_error("Batch file must contain a tasks list", code="invalid_batch")
        sys.exit(EXIT_ERROR)

    tasks = batch_data["tasks"]
    if not tasks:
        output_error("No tasks in batch file", code="empty_batch")
        sys.exit(EXIT_ERROR)

    # Resolve model and defaults
    defaults = batch_data.get("defaults", {})
    model_name = batch_data.get("model")
    template_name = batch_data.get("template")

    # If template specified, load it to get model and defaults
    if template_name:
        mgr = TemplateManager(TEMPLATES_DIR)
        tmpl = mgr.get(template_name)
        if not tmpl:
            output_error(f"Template {template_name} not found", code="template_not_found")
            sys.exit(EXIT_ERROR)
        # Extract model from template if not specified
        if not model_name:
            model_name = tmpl.get("model") or tmpl.get("defaults", {}).get("model")
        # Merge template defaults under batch defaults
        tmpl_defaults = tmpl.get("defaults", {})
        merged_defaults = {**tmpl_defaults, **defaults}
        defaults = merged_defaults

    if not model_name:
        output_error("No model specified (set model in batch file or template)", code="no_model")
        sys.exit(EXIT_ERROR)

    # Resolve output_dir
    output_dir = defaults.pop("output_dir", None)
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Submit tasks
    api_url = f"{_daemon_url(host, port)}/v1/tasks"
    submitted = []  # [(task_id, output_path, prompt)]

    try:
        with httpx.Client(timeout=30.0) as client:
            for i, task_def in enumerate(tasks):
                prompt = task_def.get("prompt", "")
                if not prompt:
                    output_error(f"Task {i+1} missing prompt", code="missing_prompt")
                    sys.exit(EXIT_ERROR)

                # Merge defaults with task overrides
                width = task_def.get("width", defaults.get("width", 1024))
                height = task_def.get("height", defaults.get("height", 1024))
                steps = task_def.get("steps", defaults.get("steps", 20))
                cfg = task_def.get("cfg", defaults.get("cfg", 7.0))
                seed = task_def.get("seed", defaults.get("seed"))
                negative = task_def.get("negative", defaults.get("negative", ""))

                # Build payload
                payload = {
                    "type": "txt2img",
                    "model": model_name,
                    "input": {
                        "prompt": prompt,
                        "negative_prompt": negative,
                    },
                    "params": {
                        "width": int(width),
                        "height": int(height),
                        "steps": int(steps),
                        "cfg_scale": float(cfg),
                    },
                }
                if seed is not None:
                    payload["params"]["seed"] = int(seed)

                # LoRA handling
                lora = task_def.get("lora", defaults.get("lora"))
                if lora:
                    payload["params"]["lora"] = _parse_batch_lora(lora)

                # Determine output path
                task_output = task_def.get("output")
                output_path = None
                if output_dir:
                    if task_output:
                        output_path = os.path.join(output_dir, task_output)
                    else:
                        output_path = os.path.join(output_dir, _slugify_prompt(prompt))
                elif task_output:
                    output_path = os.path.expanduser(task_output)

                # Submit
                try:
                    response = client.post(api_url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    task_id = result.get("task_id")
                    submitted.append({
                        "task_id": task_id,
                        "prompt": prompt[:50],
                        "output_path": output_path,
                        "status": "submitted",
                    })
                except httpx.HTTPStatusError as e:
                    try:
                        detail = e.response.json().get("detail", str(e))
                    except Exception:
                        detail = str(e)
                    submitted.append({
                        "task_id": None,
                        "prompt": prompt[:50],
                        "output_path": output_path,
                        "status": "submit_failed",
                        "error": detail,
                    })

            if not wait:
                # Just output submission results
                summary = {
                    "submitted": sum(1 for s in submitted if s["status"] == "submitted"),
                    "failed": sum(1 for s in submitted if s["status"] == "submit_failed"),
                    "total": len(submitted),
                    "results": submitted,
                }
                output_result(summary, use_json)
                return

            # Wait mode: poll all tasks until done
            pending = {s["task_id"]: s for s in submitted if s["task_id"]}

            while pending:
                time.sleep(2)
                done_ids = []
                for task_id, info in pending.items():
                    try:
                        resp = client.get(f"{api_url}/{task_id}")
                        resp.raise_for_status()
                        task_data = resp.json()
                        st = task_data.get("status")
                        if st in ("completed", "failed", "cancelled"):
                            info["status"] = st

                            # Download image if completed and output_path set
                            if st == "completed" and info["output_path"]:
                                try:
                                    img_resp = client.get(f"{api_url}/{task_id}/image", timeout=60.0)
                                    img_resp.raise_for_status()
                                    out_path = Path(info["output_path"])
                                    out_path.parent.mkdir(parents=True, exist_ok=True)
                                    out_path.write_bytes(img_resp.content)
                                    info["output"] = str(out_path)
                                except Exception as e:
                                    info["output_error"] = str(e)

                            done_ids.append(task_id)
                    except Exception:
                        pass  # Retry on next poll

                for tid in done_ids:
                    del pending[tid]

            # Build final summary
            completed = sum(1 for s in submitted if s.get("status") == "completed")
            failed = sum(1 for s in submitted if s.get("status") in ("failed", "submit_failed"))

            # Clean up results for output
            results = []
            for s in submitted:
                entry = {
                    "prompt": s["prompt"],
                    "status": s["status"],
                }
                if s.get("task_id"):
                    entry["task_id"] = s["task_id"]
                if s.get("output"):
                    entry["output"] = s["output"]
                elif s.get("output_path"):
                    entry["output"] = s["output_path"]
                if s.get("error"):
                    entry["error"] = s["error"]
                if s.get("output_error"):
                    entry["output_error"] = s["output_error"]
                results.append(entry)

            summary = {
                "completed": completed,
                "failed": failed,
                "total": len(submitted),
                "results": results,
            }
            output_result(summary, use_json)

    except httpx.ConnectError:
        output_error(
            "Daemon not running. Start it with: gpu-broker daemon start",
            code="daemon_not_running",
        )
        sys.exit(EXIT_DAEMON_NOT_RUNNING)
    except Exception as e:
        output_error(str(e))
        sys.exit(EXIT_ERROR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
