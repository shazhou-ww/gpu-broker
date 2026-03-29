# gpu-broker Design Document

**Date:** 2026-03-28
**Status:** Draft
**Author:** RAKU Team (auto-generated)

---

## 1. Overview

### 1.1 What is gpu-broker?

gpu-broker is a local GPU management daemon that acts as a "talent agent" for your GPU — it manages models, schedules inference tasks, and exposes a unified HTTP API for image generation (and future ML workloads). Think of it as a local, open-source, uncensored alternative to cloud image APIs.

### 1.2 Why?

| Problem | gpu-broker Solution |
|---------|-------------------|
| Cloud image APIs are expensive | Run locally on your own GPU for free |
| Cloud APIs censor prompts/outputs | No censorship — your hardware, your rules |
| Cloud APIs add latency | Local inference, no network round-trip |
| Image tools are scattered | One daemon, one API, one CLI |
| ML setup is painful | `gpu-broker model pull` and you're done |

### 1.3 Design Principles

1. **Agent-first**: Designed to be called by AI agents (OpenClaw, etc.), not just humans
2. **Simple over flexible**: MVP does one thing well (txt2img) before expanding
3. **Models stay loaded**: VRAM is precious — keep hot models in memory, don't reload per request
4. **CLI = client + server**: One binary, dual personality
5. **Open source**: For all OpenClaw users, no vendor lock-in

---

## 2. Architecture

### 2.1 High-Level Architecture (ASCII)

```

                         gpu-broker System Architecture

    +------------------+     +------------------+     +------------------+
    |   OpenClaw Agent |     |   CLI Client     |     |   curl / httpie  |
    |   (or any agent) |     |   (gpu-broker)   |     |   (direct HTTP)  |
    +--------+---------+     +--------+---------+     +--------+---------+
             |                        |                        |
             +------------------------+------------------------+
                                      |
                              HTTP (localhost:7878)
                                      |
                         +------------v-------------+
                         |     FastAPI Daemon        |
                         |     (gpu-broker serve)    |
                         |                           |
                         |  +---------------------+  |
                         |  |   Router Layer       |  |
                         |  |  /v1/tasks           |  |
                         |  |  /v1/models          |  |
                         |  |  /v1/status          |  |
                         |  +----------+----------+  |
                         |             |              |
                         |  +----------v----------+  |
                         |  |  Task Scheduler      |  |
                         |  |  (Queue + Workers)   |  |
                         |  +----------+----------+  |
                         |             |              |
                         |  +----------v----------+  |
                         |  |  Model Manager       |  |
                         |  |  (Load/Unload/VRAM)  |  |
                         |  +----------+----------+  |
                         |             |              |
                         +-------------|-------------+
                                       |
                    +------------------+------------------+
                    |                  |                  |
            +-------v------+  +-------v------+  +-------v------+
            |   diffusers  |  |   SQLite DB  |  |  Model Store |
            |   (PyTorch)  |  |  (tasks.db)  |  | (~/.gpu-brkr |
            |              |  |              |  |   /models/)   |
            +--------------+  +--------------+  +--------------+
                    |
            +-------v------+
            |   NVIDIA GPU |
            |   (CUDA)     |
            +--------------+

```

### 2.2 Component Summary

| Component        | Responsibility                              | Key Tech          |
|------------------|---------------------------------------------|--------------------|
| API Layer        | HTTP endpoints, request validation, routing | FastAPI, Pydantic  |
| Task Scheduler   | Queue management, worker dispatch, status   | asyncio, SQLite    |
| Model Manager    | Download, load/unload, VRAM tracking        | huggingface_hub    |
| Inference Engine | Run diffusers pipelines on GPU              | diffusers, PyTorch |
| Model Store      | On-disk model files + metadata              | File system, JSON  |
| Task DB          | Persistent task history and state            | SQLite             |

---

## 3. Module Breakdown

### 3.1 Model Manager (`gpu_broker/models/`)

**Responsibility:** Everything related to model lifecycle — discovering, downloading, loading into VRAM, unloading, and tracking metadata.

#### Key Classes / Functions

| Class / Function              | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `ModelManager`                | Singleton. Owns the model registry, coordinates load/unload, tracks VRAM.   |
| `ModelManager.pull(source)`   | Downloads a model from HuggingFace or Civitai into the local store.         |
| `ModelManager.load(model_id)` | Loads a model into VRAM via diffusers. Returns a ready-to-use pipeline.     |
| `ModelManager.unload(model_id)` | Removes model from VRAM, frees memory. Keeps files on disk.              |
| `ModelManager.list()`         | Returns all locally available models with metadata.                         |
| `ModelManager.delete(model_id)` | Deletes model files from disk. Refuses if model is currently loaded.      |
| `ModelManager.get_loaded()`   | Returns the currently loaded model (MVP: single model at a time).           |
| `ModelRecord`                 | Dataclass: id, name, source, source_url, path, format, size_bytes, type, trigger_words, pulled_at. |
| `download_from_hf(repo_id)`  | Pulls a diffusers-format model from HuggingFace Hub.                        |
| `download_from_civitai(url)` | Pulls a .safetensors checkpoint from Civitai (single-file format).          |
| `detect_model_format(path)`  | Inspects a path and returns `"diffusers"` or `"safetensors"`.               |

#### Storage Layout

```
~/.gpu-broker/
├── models/
│   ├── registry.json            # Model metadata index
│   ├── stable-diffusion-v1-5/   # Diffusers-format model (directory)
│   │   ├── model_index.json
│   │   ├── unet/
│   │   ├── vae/
│   │   └── ...
│   └── dreamshaper_8.safetensors  # Single-file checkpoint
└── data/
    └── tasks.db                 # SQLite database
```

#### Dependencies

- `huggingface_hub` — download from HF
- `requests` — download from Civitai
- `diffusers` — load pipelines from disk
- `torch` — VRAM queries (`torch.cuda.mem_get_info`)

#### MVP Constraints

- Only one model loaded in VRAM at a time (swap on demand)
- Only checkpoint models (no LoRA, no ControlNet in MVP)
- No automatic VRAM eviction — explicit load/unload only

---

### 3.2 Task Scheduler (`gpu_broker/scheduler/`)

**Responsibility:** Accepts inference requests, queues them, dispatches to the Inference Engine, tracks status, and persists results to SQLite.

#### Key Classes / Functions

| Class / Function                    | Description                                                              |
|-------------------------------------|--------------------------------------------------------------------------|
| `TaskScheduler`                     | Singleton. Owns the task queue and background worker loop.               |
| `TaskScheduler.submit(task)`        | Validates a task request, persists to DB as `pending`, enqueues it.      |
| `TaskScheduler.get_status(task_id)` | Returns current status + result path if completed.                       |
| `TaskScheduler.list(filters)`       | Lists tasks with optional status/model filters.                          |
| `TaskScheduler.cancel(task_id)`     | Cancels a pending task. Cannot cancel a running task in MVP.             |
| `_worker_loop()`                    | Async background loop: dequeue → ensure model loaded → call engine → save result. |
| `Task`                              | Dataclass: id, type, model_id, params, status, created_at, started_at, completed_at, result_path, error. |

#### Task Lifecycle

```
  submit()          _worker_loop()        engine.run()         save result
     │                    │                    │                    │
     v                    v                    v                    v
 ┌────────┐  dequeue  ┌─────────┐  dispatch ┌─────────┐  done  ┌───────────┐
 │ pending ├─────────>│ running  ├─────────>│ running  ├──────>│ completed │
 └────────┘           └─────────┘           └─────────┘        └───────────┘
     │                                           │
     │  cancel()                                 │  exception
     v                                           v
 ┌───────────┐                              ┌────────┐
 │ cancelled │                              │ failed │
 └───────────┘                              └────────┘
```

#### Dependencies

- `ModelManager` — to ensure the required model is loaded before dispatch
- `InferenceEngine` — to execute the actual inference
- `sqlite3` — task persistence
- `asyncio` — background worker, queue

---

### 3.3 Inference Engine (`gpu_broker/engine/`)

**Responsibility:** Thin wrapper around diffusers pipelines. Takes a loaded pipeline + task params, runs inference, saves the output image, returns the file path.

#### Key Classes / Functions

| Class / Function                          | Description                                                           |
|-------------------------------------------|-----------------------------------------------------------------------|
| `InferenceEngine`                         | Stateless. Each method takes a pipeline and params, returns a result. |
| `InferenceEngine.txt2img(pipeline, params)` | Runs text-to-image generation. Returns path to saved PNG.           |
| `Txt2ImgParams`                           | Dataclass: prompt, negative_prompt, width, height, steps, cfg_scale, seed. |
| `save_output(image, task_id)`             | Saves a PIL Image to `~/.gpu-broker/outputs/{task_id}.png`.           |

#### Defaults

| Parameter         | Default | Constraints                |
|-------------------|---------|----------------------------|
| `width`           | 512     | Multiple of 8, 128–2048    |
| `height`          | 512     | Multiple of 8, 128–2048    |
| `steps`           | 20      | 1–150                      |
| `cfg_scale`       | 7.0     | 1.0–30.0                   |
| `seed`            | random  | 0–2^32                     |
| `negative_prompt` | `""`    | Any string                 |

#### Dependencies

- `diffusers` — pipeline execution
- `torch` — tensor operations, GPU device management
- `PIL` / `Pillow` — image saving

---

### 3.4 API Layer (`gpu_broker/api/`)

**Responsibility:** FastAPI application with route handlers. Validates requests via Pydantic, calls into Scheduler and ModelManager, returns JSON responses.

#### Key Classes / Functions

| Class / Function         | Description                                               |
|--------------------------|-----------------------------------------------------------|
| `create_app()`           | Factory that builds the FastAPI app, wires up dependencies.|
| `models_router`          | APIRouter for `/v1/models/*` endpoints.                   |
| `tasks_router`           | APIRouter for `/v1/tasks/*` endpoints.                    |
| `status_router`          | APIRouter for `/v1/status` endpoint.                      |
| `lifespan(app)`          | Async context manager — starts scheduler worker on boot, cleans up on shutdown. |

#### Dependencies

- `fastapi`, `uvicorn` — HTTP server
- `pydantic` — request/response validation
- `TaskScheduler` — injected via app state
- `ModelManager` — injected via app state

---

## 4. API Design

All endpoints are prefixed with `/v1`. The daemon listens on `localhost:7878` by default.

### 4.1 Models

#### `GET /v1/models` — List local models

**Response `200 OK`:**

```json
{
  "models": [
    {
      "id": "stable-diffusion-v1-5",
      "name": "stable-diffusion-v1-5",
      "source": "huggingface",
      "source_url": "runwayml/stable-diffusion-v1-5",
      "format": "diffusers",
      "size_bytes": 4265380864,
      "type": "checkpoint",
      "trigger_words": null,
      "pulled_at": "2026-03-28T10:15:00Z",
      "loaded": true
    }
  ],
  "count": 1
}
```

#### `POST /v1/models/pull` — Download a model

**Request:**

```json
{
  "source": "huggingface",
  "repo_id": "runwayml/stable-diffusion-v1-5"
}
```

Or for Civitai:

```json
{
  "source": "civitai",
  "url": "https://civitai.com/api/download/models/128713",
  "filename": "dreamshaper_8.safetensors"
}
```

**Response `202 Accepted`:**

```json
{
  "model_id": "dreamshaper_8",
  "status": "downloading",
  "message": "Download started"
}
```

**Response `409 Conflict`** (model already exists):

```json
{
  "error": "model_exists",
  "message": "Model 'dreamshaper_8' already exists locally",
  "model_id": "dreamshaper_8"
}
```

#### `DELETE /v1/models/{model_id}` — Delete a model

**Response `200 OK`:**

```json
{
  "model_id": "dreamshaper_8",
  "deleted": true
}
```

**Response `409 Conflict`** (model currently loaded):

```json
{
  "error": "model_loaded",
  "message": "Cannot delete 'dreamshaper_8' while it is loaded. Unload it first."
}
```

**Response `404 Not Found`:**

```json
{
  "error": "model_not_found",
  "message": "No model with id 'nonexistent'"
}
```

---

### 4.2 Tasks

#### `POST /v1/tasks` — Submit a new task

**Request (txt2img):**

```json
{
  "type": "txt2img",
  "model_id": "stable-diffusion-v1-5",
  "params": {
    "prompt": "a cat sitting on a rainbow, digital art",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg_scale": 7.0,
    "seed": 42
  }
}
```

**Response `202 Accepted`:**

```json
{
  "task_id": "tsk_a1b2c3d4e5f6",
  "type": "txt2img",
  "model_id": "stable-diffusion-v1-5",
  "status": "pending",
  "created_at": "2026-03-28T11:30:00Z"
}
```

**Response `400 Bad Request`** (validation error):

```json
{
  "error": "validation_error",
  "message": "width must be a multiple of 8",
  "field": "params.width"
}
```

**Response `404 Not Found`** (model not available):

```json
{
  "error": "model_not_found",
  "message": "Model 'nonexistent' is not available locally. Pull it first."
}
```

#### `GET /v1/tasks/{task_id}` — Get task status

**Response `200 OK` (completed):**

```json
{
  "task_id": "tsk_a1b2c3d4e5f6",
  "type": "txt2img",
  "model_id": "stable-diffusion-v1-5",
  "status": "completed",
  "params": {
    "prompt": "a cat sitting on a rainbow, digital art",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg_scale": 7.0,
    "seed": 42
  },
  "created_at": "2026-03-28T11:30:00Z",
  "started_at": "2026-03-28T11:30:01Z",
  "completed_at": "2026-03-28T11:30:08Z",
  "result": {
    "image_path": "/outputs/tsk_a1b2c3d4e5f6.png",
    "image_url": "/v1/tasks/tsk_a1b2c3d4e5f6/image"
  }
}
```

**Response `200 OK` (failed):**

```json
{
  "task_id": "tsk_a1b2c3d4e5f6",
  "type": "txt2img",
  "model_id": "stable-diffusion-v1-5",
  "status": "failed",
  "params": { "..." : "..." },
  "created_at": "2026-03-28T11:30:00Z",
  "started_at": "2026-03-28T11:30:01Z",
  "completed_at": "2026-03-28T11:30:02Z",
  "error": "CUDA out of memory. Tried to allocate 2.00 GiB"
}
```

#### `GET /v1/tasks` — List tasks

**Query Parameters:**

| Parameter  | Type   | Default   | Description                                    |
|------------|--------|-----------|------------------------------------------------|
| `status`   | string | (all)     | Filter: `pending`, `running`, `completed`, `failed`, `cancelled` |
| `model_id` | string | (all)     | Filter by model used                           |
| `limit`    | int    | 20        | Max results (1–100)                            |
| `offset`   | int    | 0         | Pagination offset                              |

**Response `200 OK`:**

```json
{
  "tasks": [
    {
      "task_id": "tsk_a1b2c3d4e5f6",
      "type": "txt2img",
      "model_id": "stable-diffusion-v1-5",
      "status": "completed",
      "created_at": "2026-03-28T11:30:00Z",
      "completed_at": "2026-03-28T11:30:08Z"
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

#### `GET /v1/tasks/{task_id}/image` — Download result image

**Response `200 OK`:** Binary PNG data with `Content-Type: image/png`.

**Response `404 Not Found`:** Task doesn't exist or hasn't completed.

---

### 4.3 Server Status

#### `GET /v1/status` — Server health and GPU info

**Response `200 OK`:**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "gpu": {
    "name": "NVIDIA RTX 4090",
    "vram_total_mb": 24576,
    "vram_used_mb": 4200,
    "vram_free_mb": 20376,
    "cuda_version": "12.1"
  },
  "loaded_model": {
    "id": "stable-diffusion-v1-5",
    "loaded_at": "2026-03-28T10:30:00Z"
  },
  "queue": {
    "pending": 2,
    "running": 1,
    "completed_today": 47
  }
}
```

---

## 5. Data Models

All persistent state lives in a single SQLite database at `~/.gpu-broker/data/tasks.db`.

### 5.1 `models` Table

Tracks every model that has been pulled to the local store.

```sql
CREATE TABLE models (
    id            TEXT PRIMARY KEY,                -- e.g. "stable-diffusion-v1-5"
    name          TEXT NOT NULL,                    -- display name
    source        TEXT NOT NULL CHECK(source IN ('huggingface', 'civitai', 'local')),
    source_url    TEXT,                             -- HF repo id or Civitai download URL
    path          TEXT NOT NULL,                    -- absolute path to model dir or file
    format        TEXT NOT NULL CHECK(format IN ('diffusers', 'safetensors')),
    size_bytes    INTEGER NOT NULL DEFAULT 0,       -- total size on disk
    type          TEXT NOT NULL DEFAULT 'checkpoint' CHECK(type IN ('checkpoint')),
    trigger_words TEXT,                             -- JSON array string, nullable
    pulled_at     TEXT NOT NULL DEFAULT (datetime('now')),  -- ISO 8601
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))   -- ISO 8601
);
```

### 5.2 `tasks` Table

Tracks every inference task submitted to the scheduler.

```sql
CREATE TABLE tasks (
    id            TEXT PRIMARY KEY,                -- e.g. "tsk_a1b2c3d4e5f6"
    type          TEXT NOT NULL DEFAULT 'txt2img' CHECK(type IN ('txt2img')),
    model_id      TEXT NOT NULL,                    -- references models.id
    params        TEXT NOT NULL,                    -- JSON blob of task parameters
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    result_path   TEXT,                             -- path to output file, set on completion
    error         TEXT,                             -- error message, set on failure
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    started_at    TEXT,                             -- set when worker picks up the task
    completed_at  TEXT,                             -- set on completion or failure
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_model ON tasks(model_id);
CREATE INDEX idx_tasks_created ON tasks(created_at DESC);
```

### 5.3 Schema Notes

- **Why JSON for `params` and `trigger_words`?** Keeps the schema stable as we add new task types (img2img, inpaint) and model metadata. SQLite's `json_extract()` can still query inside these fields.
- **No `users` table.** MVP is single-user, local-only. Auth is out of scope.
- **Soft deletes?** No. When a model is deleted, its row is removed from `models` and files are deleted from disk. Tasks that reference a deleted model keep their `model_id` for history (no cascading delete).
- **Migration strategy.** MVP ships with `CREATE TABLE IF NOT EXISTS`. Future versions will use numbered migration scripts in `gpu_broker/migrations/`.

---

## 6. CLI Interface

The `gpu-broker` CLI has dual personality: it is both the server daemon and the client that talks to it.

### 6.1 Command Tree

```
gpu-broker
├── serve                  # Start the daemon (foreground or background)
├── model
│   ├── pull               # Download a model
│   ├── list (ls)          # List local models
│   └── rm                 # Delete a model
├── task
│   ├── submit             # Submit a new inference task
│   ├── status             # Check task status
│   └── list (ls)          # List tasks
└── status                 # Show server health + GPU info
```

### 6.2 Commands

#### `gpu-broker serve`

Start the HTTP daemon.

```bash
# Start in foreground (default)
gpu-broker serve

# Custom host/port
gpu-broker serve --host 0.0.0.0 --port 8080

# Daemonize (detach from terminal)
gpu-broker serve --daemon

# With specific log level
gpu-broker serve --log-level debug
```

| Flag          | Default        | Description                    |
|---------------|----------------|--------------------------------|
| `--host`      | `127.0.0.1`   | Bind address                   |
| `--port`      | `7878`         | Bind port                      |
| `--daemon`    | `false`        | Run in background              |
| `--log-level` | `info`         | `debug`, `info`, `warning`, `error` |

---

#### `gpu-broker model pull`

Download a model from HuggingFace or Civitai.

```bash
# From HuggingFace (diffusers format)
gpu-broker model pull runwayml/stable-diffusion-v1-5

# From Civitai (safetensors)
gpu-broker model pull --source civitai --url https://civitai.com/api/download/models/128713 --name dreamshaper_8

# Pull with progress bar
gpu-broker model pull stabilityai/stable-diffusion-xl-base-1.0
# Downloading stabilityai/stable-diffusion-xl-base-1.0 ...
# ████████████████████████████░░░░  89%  4.2 GB / 4.7 GB  ETA 0:32
```

| Flag       | Default       | Description                          |
|------------|---------------|--------------------------------------|
| `--source` | `huggingface` | `huggingface` or `civitai`           |
| `--url`    | (required for civitai) | Civitai download URL        |
| `--name`   | (auto)        | Override the model ID/name           |

---

#### `gpu-broker model list` / `gpu-broker model ls`

List all locally available models.

```bash
gpu-broker model list

# Output:
# ID                          FORMAT       SIZE      SOURCE       PULLED
# stable-diffusion-v1-5       diffusers    4.0 GB    huggingface  2026-03-28 10:15
# dreamshaper_8               safetensors  2.1 GB    civitai      2026-03-28 10:42
#
# 2 models, 6.1 GB total

# JSON output for scripting
gpu-broker model list --json
```

---

#### `gpu-broker model rm`

Delete a model from disk.

```bash
gpu-broker model rm dreamshaper_8

# Output:
# Deleted dreamshaper_8 (2.1 GB freed)

# Force delete even if loaded (unloads first)
gpu-broker model rm --force stable-diffusion-v1-5
```

---

#### `gpu-broker task submit`

Submit an inference task via CLI (calls the HTTP API under the hood).

```bash
# Basic txt2img
gpu-broker task submit --prompt "a cat sitting on a rainbow, digital art"

# Full options
gpu-broker task submit \
  --model stable-diffusion-v1-5 \
  --prompt "cyberpunk cityscape at night" \
  --negative-prompt "blurry, watermark" \
  --width 768 \
  --height 512 \
  --steps 30 \
  --cfg-scale 7.5 \
  --seed 42

# Output:
# Task submitted: tsk_a1b2c3d4e5f6
# Model: stable-diffusion-v1-5
# Status: pending (position 3 in queue)

# Submit and wait for result
gpu-broker task submit --prompt "sunset over mountains" --wait
# Task submitted: tsk_x9y8z7w6v5u4
# Waiting... ████████████████████ done (6.2s)
# Result saved to: ~/.gpu-broker/outputs/tsk_x9y8z7w6v5u4.png
```

| Flag                | Default                  | Description                      |
|---------------------|--------------------------|----------------------------------|
| `--model`           | (currently loaded model) | Model ID to use                  |
| `--prompt`          | (required)               | Text prompt                      |
| `--negative-prompt` | `""`                     | Negative prompt                  |
| `--width`           | `512`                    | Image width (multiple of 8)      |
| `--height`          | `512`                    | Image height (multiple of 8)     |
| `--steps`           | `20`                     | Inference steps                  |
| `--cfg-scale`       | `7.0`                    | CFG scale                        |
| `--seed`            | (random)                 | Random seed for reproducibility  |
| `--wait`            | `false`                  | Block until task completes       |
| `--output`, `-o`    | (default path)           | Copy result image to this path   |

---

#### `gpu-broker task status`

Check the status of a submitted task.

```bash
gpu-broker task status tsk_a1b2c3d4e5f6

# Output (pending):
# Task:    tsk_a1b2c3d4e5f6
# Status:  pending (position 2 in queue)
# Model:   stable-diffusion-v1-5
# Created: 2026-03-28 11:30:00

# Output (completed):
# Task:      tsk_a1b2c3d4e5f6
# Status:    completed
# Model:     stable-diffusion-v1-5
# Created:   2026-03-28 11:30:00
# Duration:  6.2s
# Result:    ~/.gpu-broker/outputs/tsk_a1b2c3d4e5f6.png
```

---

#### `gpu-broker task list` / `gpu-broker task ls`

List tasks with optional filters.

```bash
# List recent tasks
gpu-broker task list

# Output:
# ID                    TYPE     MODEL                    STATUS      CREATED
# tsk_a1b2c3d4e5f6      txt2img  stable-diffusion-v1-5    completed   11:30:00
# tsk_f6e5d4c3b2a1      txt2img  stable-diffusion-v1-5    running     11:31:15
# tsk_z9y8x7w6v5u4      txt2img  dreamshaper_8            pending     11:31:20

# Filter by status
gpu-broker task list --status completed

# Limit results
gpu-broker task list --limit 5
```

---

#### `gpu-broker status`

Show server health, GPU info, and queue state.

```bash
gpu-broker status

# Output:
# gpu-broker v0.1.0 — running (uptime: 1h 23m)
#
# GPU:     NVIDIA RTX 4090
# VRAM:    4.2 GB / 24.0 GB used
# CUDA:    12.1
#
# Model:   stable-diffusion-v1-5 (loaded 1h ago)
#
# Queue:   2 pending, 1 running, 47 completed today

# JSON output
gpu-broker status --json
```

---

## 7. MVP Milestones

### Milestone 1: Project Skeleton

**Goal:** Runnable Python package with CLI entry point and empty FastAPI server.

**Deliverables:**

| #  | Deliverable                           | Details                                                    |
|----|---------------------------------------|------------------------------------------------------------|
| 1  | Repository + pyproject.toml           | Package name `gpu-broker`, Python ≥3.10, dependencies declared |
| 2  | CLI entry point                       | `gpu-broker --help` works, shows command tree               |
| 3  | `gpu-broker serve`                    | Starts uvicorn, serves empty FastAPI app on `:7878`         |
| 4  | `GET /v1/status`                      | Returns `{"status": "ok", "version": "0.1.0"}`             |
| 5  | Project structure                     | Directories for `api/`, `models/`, `scheduler/`, `engine/`  |
| 6  | Basic logging                         | Structured logging with configurable level                  |

**Done when:** `pip install -e .` → `gpu-broker serve` → `curl localhost:7878/v1/status` returns OK.

---

### Milestone 2: Model Management

**Goal:** Download, store, list, and delete models via CLI and API.

**Deliverables:**

| #  | Deliverable                           | Details                                                    |
|----|---------------------------------------|------------------------------------------------------------|
| 1  | `ModelManager` core                   | Pull from HuggingFace (diffusers format)                   |
| 2  | Civitai download support              | Pull .safetensors by URL                                   |
| 3  | SQLite `models` table                 | Schema as defined in Section 5.1                           |
| 4  | `gpu-broker model pull <repo>`        | CLI command with progress bar                              |
| 5  | `gpu-broker model list`               | CLI table output                                           |
| 6  | `gpu-broker model rm`                 | Delete files + DB row                                      |
| 7  | `GET /v1/models`                      | API endpoint returning model list                          |
| 8  | `POST /v1/models/pull`               | API endpoint to trigger download                           |
| 9  | `DELETE /v1/models/{id}`             | API endpoint to delete a model                             |

**Done when:** Can pull Stable Diffusion v1.5 from HF, see it in `model list`, delete it with `model rm`, and do all the same via HTTP API.

---

### Milestone 3: Inference Engine

**Goal:** Load a model into VRAM and generate an image from a text prompt (no queue yet, synchronous).

**Deliverables:**

| #  | Deliverable                           | Details                                                    |
|----|---------------------------------------|------------------------------------------------------------|
| 1  | `InferenceEngine.txt2img()`           | Takes pipeline + params, returns PIL Image                 |
| 2  | Model loading                         | `ModelManager.load()` creates a diffusers pipeline on CUDA  |
| 3  | Safetensors support                   | Load single-file checkpoints via `StableDiffusionPipeline.from_single_file()` |
| 4  | Output saving                         | Save PNG to `~/.gpu-broker/outputs/`                       |
| 5  | Synchronous test endpoint             | Temporary `POST /v1/test/txt2img` for validation            |

**Done when:** `curl -X POST localhost:7878/v1/test/txt2img -d '{"prompt":"a cat"}'` returns a valid PNG path, and the image looks correct.

---

### Milestone 4: Task Scheduler + Full API

**Goal:** Async task queue with persistent state. Tasks are submitted, queued, executed, and results are retrievable.

**Deliverables:**

| #  | Deliverable                           | Details                                                    |
|----|---------------------------------------|------------------------------------------------------------|
| 1  | SQLite `tasks` table                  | Schema as defined in Section 5.2                           |
| 2  | `TaskScheduler` with async worker     | Background loop: dequeue → load model → infer → save       |
| 3  | `POST /v1/tasks`                      | Submit a txt2img task, returns task ID                      |
| 4  | `GET /v1/tasks/{id}`                  | Get task status and result                                 |
| 5  | `GET /v1/tasks`                       | List tasks with filters                                    |
| 6  | `GET /v1/tasks/{id}/image`            | Serve result image                                         |
| 7  | `gpu-broker task submit`              | CLI command with `--wait` option                           |
| 8  | `gpu-broker task status`              | CLI status check                                           |
| 9  | `gpu-broker task list`                | CLI task listing                                           |
| 10 | Remove temporary test endpoint        | Replace with real task-based flow                           |

**Done when:** Submit 3 tasks via CLI rapidly, they queue up and execute sequentially, all results are retrievable via both CLI and API.

---

### Milestone 5: Polish + End-to-End Validation

**Goal:** Production-quality MVP ready for OpenClaw users. Error handling, logging, documentation.

**Deliverables:**

| #  | Deliverable                           | Details                                                    |
|----|---------------------------------------|------------------------------------------------------------|
| 1  | Error handling                        | Graceful handling of CUDA OOM, missing models, bad params   |
| 2  | Model auto-loading                    | If a task references an unloaded model, swap automatically  |
| 3  | `gpu-broker status`                   | Full status with GPU info, queue state, loaded model        |
| 4  | `GET /v1/status` complete             | All fields populated (GPU, queue, model, uptime)            |
| 5  | README.md                             | Installation, quickstart, API reference                    |
| 6  | Agent integration test                | OpenClaw agent successfully generates an image via API      |
| 7  | Daemon mode                           | `gpu-broker serve --daemon` with PID file                  |

**Done when:** A fresh user can `pip install gpu-broker` → `gpu-broker model pull runwayml/stable-diffusion-v1-5` → `gpu-broker serve --daemon` → `gpu-broker task submit --prompt "hello world" --wait` and get a generated image. An OpenClaw agent can do the same via HTTP.

