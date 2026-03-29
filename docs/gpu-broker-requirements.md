# gpu-broker Requirements (Collected 2026-03-28)

## Project
- Name: gpu-broker
- Concept: GPU agent/broker - manages GPU like a talent agent manages a star
- License: Open source, for all OpenClaw users

## Pain Points (Priority Order)
1. Cloud image APIs too expensive/slow/censored, need local alternative
2. Image tools too scattered, need unified entry point
3. Lower ML barrier for agents/users without ML background

## Target Environment
- Core: Linux + NVIDIA GPU
- Best-effort: Windows CUDA, Mac MPS

## MVP Scope
1. Model Management: Download from HuggingFace + Civitai, unified directory, basic metadata (name, size, type, source, trigger words if available). Support both .safetensors single-file and diffusers directory format. MVP only manages checkpoints.
2. Task Scheduling: HTTP daemon (FastAPI), models stay in VRAM, SQLite persistence for task history. This is the core of MVP validation.
3. txt2img: First task type to validate the pipeline end-to-end.

## Tech Stack
- Python + FastAPI + diffusers + SQLite
- CLI: dual role (server management: start/stop/status + client: submit tasks via HTTP API)
