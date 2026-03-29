# GPU Broker — MVP 需求文档

> **版本**: 1.0  
> **状态**: 已批准  
> **最后更新**: 2026-03-29  
> **负责人**: Scott Wei

---

## 1. 概述

| 字段 | 说明 |
|------|------|
| **项目** | gpu-broker — 本地 GPU 管理守护进程 |
| **概念** | 像经纪人管理明星一样管理 GPU |
| **状态** | 已批准 |
| **目标** | 让 AI Agent 能够通过 CLI 在本地 GPU 上执行推理任务，跑通模型管理和任务调度的完整流程 |

## 2. 利益相关者

| 角色 | 负责人/团队 |
|------|------------|
| 业务负责人 | Scott Wei |
| 最终用户 | AI Agent（OpenClaw 通过 `exec` 调用 CLI） |
| 技术实现 | KUMA Team |
| 技术设计 | RAKU Team |

## 3. 业务背景

### 3.1 问题陈述

1. **成本与速度** — 云端图片生成 API 太贵、太慢、有审查限制
2. **工具碎片化** — 各种本地图片生成工具太分散，每个都要单独学
3. **Agent 门槛高** — AI Agent 使用本地 GPU 的门槛太高

### 3.2 业务价值

为 OpenClaw 生态提供统一的本地 GPU 推理能力，降低 Agent 调用门槛。

### 3.3 成功指标

- Agent 能通过一条 CLI 命令完成「提交任务 → 等待 → 拿到图片路径」的完整流程
- 模型管理（下载 / 列表 / 删除）全通过 CLI 完成，无需手动操作文件

## 4. 部署环境

| 项目 | 说明 |
|------|------|
| 硬件 | RAKU Home PC, NVIDIA RTX 4070Ti, 12 GB VRAM |
| OS | Linux（核心支持）; Windows CUDA / Mac MPS best-effort |
| 约束 | 单模型运行（12 GB VRAM 限制） |

## 5. 架构

```
┌─────────────────────────────────────────┐
│  AI Agent (OpenClaw)                    │
│  exec("gpu-broker task submit ...")     │
└──────────────┬──────────────────────────┘
               │ HTTP (localhost)
┌──────────────▼──────────────────────────┐
│  CLI Client (Click)                     │
│  gpu-broker <command> <args>            │
└──────────────┬──────────────────────────┘
               │ HTTP API
┌──────────────▼──────────────────────────┐
│  Daemon (FastAPI)                       │
│  ┌────────────┐ ┌──────────┐ ┌───────┐ │
│  │ Model Mgr  │ │ Task Q   │ │ Infr  │ │
│  │ (download, │ │ (asyncio │ │ (diff │ │
│  │  list,     │ │  .Queue) │ │ users)│ │
│  │  remove)   │ │          │ │       │ │
│  └────────────┘ └──────────┘ └───────┘ │
│  ┌────────────────────────────────────┐ │
│  │ SQLite (models + tasks metadata)  │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### 为什么 Daemon 架构

避免每次调用冷启动加载模型 — checkpoint reload 需要几十秒，Daemon 常驻进程保持模型热加载。

## 6. CLI 命令体系

```
gpu-broker
├── daemon
│   ├── start              # 启动后台 daemon
│   ├── stop               # 停止 daemon
│   └── status             # 查看状态（运行中/端口/已加载模型/队列深度）
├── model
│   ├── download <url>     # 自动识别 HuggingFace / Civitai
│   ├── list               # 列出已下载模型（ID + name + size）
│   ├── remove <id>        # 删除模型
│   └── info <id>          # 模型详情（完整 hash/路径/大小/下载时间）
├── task
│   ├── submit [--wait] <json>  # 提交任务（JSON 参数或 stdin）
│   ├── status <task_id>        # 查看任务状态
│   ├── list [--status ...]     # 列出任务
│   └── cancel <task_id>        # 取消排队中的任务
└── config
    ├── show               # 显示当前配置
    └── set <key> <value>  # 设置配置项
```

### 6.1 模型 ID 体系

- 使用 **SHA256 前 12 位**作为短 ID（类似 Docker image ID）
- 完整 hash 存储在数据库，CLI 显示短 ID
- 支持 ID 和 name 模糊匹配
- `model download` 后返回分配的 ID

### 6.2 task submit 格式

```json
{
  "type": "txt2img",
  "model": "a1b2c3d4e5f6",
  "prompt": "a cat sitting on a roof",
  "negative_prompt": "blurry, low quality",
  "params": {
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "seed": 42
  },
  "output_dir": "/path/to/output"
}
```

- 支持参数直接传或 stdin 读入（`echo '...' | gpu-broker task submit`）
- `--wait`：同步阻塞直到任务完成再返回结果
- 不加 `--wait`：立即返回 `task_id`，异步执行

### 6.3 输出格式

- 所有 CLI 输出默认 **JSON**（Agent 友好）
- 生成的图片保存到本地路径，返回路径

### 6.4 Exit Codes

| Code | 含义 |
|------|------|
| 0 | 成功 |
| 1 | 一般错误 |
| 2 | 模型未找到 |
| 3 | Daemon 未运行 |
| 4 | 超时 |
| 5 | VRAM 不足 |

### 6.5 API Key 管理

- **优先级**: 环境变量 > config 文件
- **环境变量**: `HF_TOKEN`, `CIVITAI_API_KEY`
- **Config**: `~/.config/gpu-broker/config.json`
- HuggingFace 公开模型不需要 token，gated 模型需要

## 7. 功能性需求

### FR-1: 模型管理

| ID | 需求 |
|----|------|
| FR-1.1 | 通过 URL 下载模型，自动识别 HuggingFace / Civitai |
| FR-1.2 | 下载后分配 SHA256 短 ID，存入本地数据库 |
| FR-1.3 | 列出所有已下载模型（ID、名称、大小、来源） |
| FR-1.4 | 按 ID 删除模型（同时清理文件和数据库记录） |
| FR-1.5 | 按 ID 查看模型详情 |

### FR-2: 任务调度

| ID | 需求 |
|----|------|
| FR-2.1 | 提交 txt2img 任务（JSON 格式） |
| FR-2.2 | 任务排队（`asyncio.Queue`），单模型串行执行 |
| FR-2.3 | 支持同步等待（`--wait`）和异步提交两种模式 |
| FR-2.4 | 查询任务状态（queued / running / completed / failed） |
| FR-2.5 | 取消排队中的任务 |
| FR-2.6 | 列出所有任务，支持按状态过滤 |

### FR-3: Daemon 管理

| ID | 需求 |
|----|------|
| FR-3.1 | 启动 / 停止后台 daemon |
| FR-3.2 | 查看 daemon 状态（运行状态、端口、已加载模型、队列深度） |

### FR-4: 配置管理

| ID | 需求 |
|----|------|
| FR-4.1 | 查看当前配置 |
| FR-4.2 | 设置配置项（`output_dir`, `default_steps` 等） |

## 8. 非功能性需求

### NFR-1: 错误处理

- 明确的 exit code（见 [6.4 Exit Codes](#64-exit-codes)）
- JSON 格式错误信息，包含 `error_code` + `message`
- VRAM 不足 / 模型加载失败 / 推理超时均有明确错误

### NFR-2: 超时

- 默认 **120 秒**（4070Ti 跑 SDXL txt2img 约 10-30 秒，留足余量）

### NFR-3: 并发

- 单模型场景，同时只处理一个任务
- 多任务排队等待，不拒绝

### NFR-4: 存储

| 数据 | 存储方式 |
|------|----------|
| 模型元数据 | SQLite |
| 任务记录 | SQLite（持久化队列） |
| 生成图片 | 本地文件系统 |

## 9. 验收标准

### AC-1: 模型下载

```
Given  daemon 已启动
When   执行 gpu-broker model download https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
Then   模型下载到本地，返回 JSON 包含 model_id, name, size
```

### AC-2: 任务提交（同步）

```
Given  daemon 已启动且模型已加载
When   执行 gpu-broker task submit --wait '{"type":"txt2img","model":"<id>","prompt":"a cat"}'
Then   阻塞等待完成，返回 JSON 包含 output 路径，图片文件存在
```

### AC-3: 任务提交（异步）

```
Given  daemon 已启动
When   执行 gpu-broker task submit '{"type":"txt2img",...}'
Then   立即返回 task_id

When   执行 gpu-broker task status <task_id>
Then   返回当前状态
```

### AC-4: 错误处理

```
Given  daemon 未运行
When   执行任何 task 命令
Then   exit code 3，JSON 错误信息包含 "daemon not running"
```

## 10. MVP 范围

### 包含 ✅

- txt2img 文生图
- 模型管理（HuggingFace + Civitai 下载）
- 任务调度（排队 + 执行）
- CLI + Daemon 架构
- JSON 输出
- SQLite 持久化

### 不包含（post-MVP）❌

- LoRA 管理和加载
- img2img（图生图）
- inpaint（局部重绘）
- ControlNet
- base64 图片输出
- 多模型并发
- Web UI
- 远程调用（仅本地）

## 11. 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| Daemon | FastAPI（HTTP server） |
| CLI | Click |
| 推理引擎 | diffusers |
| 数据库 | aiosqlite（异步） |
| 图片处理 | Pillow |

## 12. 里程碑

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| M1 项目骨架 | CLI + 空 FastAPI server | ✅ 完成 |
| M2 模型管理 | download / list / remove / info | ✅ 完成（需按新设计重构） |
| M3 推理引擎 | 加载模型到 VRAM，txt2img | ✅ 完成（需按新设计重构） |
| M4 任务调度 | 异步队列 + SQLite 持久化 | ✅ 完成（需按新设计重构） |
| M5 打磨 | 端到端验证 + 文档 + 错误处理 | 待做 |

> **注**: M1-M4 已有代码骨架（mock 模式），但 CLI 命令体系需按本文档重构。

## 13. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 12 GB VRAM 跑 SDXL 可能紧张 | 推理失败或 OOM | 支持较小模型（SD 1.5），后续加模型量化 |
| RAKU PC 网络不稳定 | 模型下载中断 | 下载支持断点续传 |
| diffusers 版本兼容性 | 推理结果不一致 | 锁定版本，CI 测试 |

---

*文档结束*
