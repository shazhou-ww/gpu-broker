# gpu-broker

Local GPU management daemon that acts as a "talent agent" for your GPU — manages models, schedules inference tasks, and exposes a unified HTTP API.

## Status

✅ **M1+M2: API 骨架 + 模型管理** — 完成
✅ **M3+M4: 推理引擎 + 任务调度** — 完成（Mock 模式）

**当前进度：** 完整 txt2img pipeline 已实现并验证通过（无 GPU 依赖的 mock 模式）。

### 功能清单
- [x] HTTP API 服务器（FastAPI）
- [x] 模型管理（HuggingFace / Civitai 下载）
- [x] 推理引擎（支持 diffusers/safetensors 格式）
- [x] 异步任务调度器
- [x] 任务队列和持久化
- [x] CLI 工具（model/task 管理）
- [x] Mock 模式（无 GPU 环境验证）
- [ ] 生产环境 GPU 测试
- [ ] 更多任务类型（img2img, inpaint 等）

### 快速开始
```bash
# 安装
pip install -e .

# 启动服务
gpu-broker serve

# 查看模型
gpu-broker model list

# 提交任务
gpu-broker task submit --model <model-id> --prompt "a cat" --wait
```

详见 [验收测试文档](ACCEPTANCE_M3M4.md)。

## Docs

- [Requirements](docs/gpu-broker-requirements.md)
- [Design Document](docs/2026-03-28-gpu-broker-design.md)

## Team

- RAKU (design) → KUMA (implementation)
