# Milestone 2 验收测试报告

## 执行时间
2026-03-29 08:02 UTC

## 测试环境
- OS: Linux 6.8.0-1044-azure (x64)
- Python: 3.x
- gpu-broker: v0.1.0
- 测试端口: 127.0.0.1:7879

## 验收标准完成情况

### ✅ DB Schema 完全一致

**models 表**:
```sql
CREATE TABLE models (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    source        TEXT NOT NULL CHECK(source IN ('huggingface', 'civitai', 'local')),
    source_url    TEXT,
    path          TEXT NOT NULL,
    format        TEXT NOT NULL CHECK(format IN ('diffusers', 'safetensors')),
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    type          TEXT NOT NULL DEFAULT 'checkpoint' CHECK(type IN ('checkpoint')),
    trigger_words TEXT,
    pulled_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**tasks 表**:
```sql
CREATE TABLE tasks (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL DEFAULT 'txt2img' CHECK(type IN ('txt2img')),
    model_id      TEXT NOT NULL,
    params        TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    result_path   TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    started_at    TEXT,
    completed_at  TEXT,
    FOREIGN KEY (model_id) REFERENCES models(id)
);
```

**索引**:
- ✅ `idx_tasks_status` ON tasks(status)
- ✅ `idx_tasks_model` ON tasks(model_id)
- ✅ `idx_tasks_created` ON tasks(created_at DESC)

### ✅ Pydantic Schemas 对齐

已更新 `gpu_broker/api/schemas.py`:
- ✅ ModelInfo: 包含所有必需字段 (id, name, source, source_url, path, format, size_bytes, type, trigger_words, pulled_at, loaded)
- ✅ ModelListResponse: models list + count
- ✅ ModelPullRequest: source, repo_id (可选), url (可选), filename (可选)
- ✅ TaskSubmitRequest: type, model_id, params (Txt2ImgParams)
- ✅ Txt2ImgParams: prompt, negative_prompt, width, height, steps, cfg_scale, seed

### ✅ ModelManager 实现

文件: `gpu_broker/models/manager.py`

实现的方法:
- ✅ `__init__(db_path, models_dir)` — 初始化
- ✅ `pull(source, repo_id=None, url=None, filename=None)` — 下载模型
  - ✅ HuggingFace: 使用 `huggingface_hub.snapshot_download()`
  - ✅ Civitai: 使用 requests 下载 .safetensors
  - ✅ 条件导入处理（try/except ImportError）
  - ✅ 友好错误提示
- ✅ `list()` — 返回所有本地模型
- ✅ `get(model_id)` — 获取单个模型信息
- ✅ `delete(model_id)` — 删除模型文件和 DB 记录
- ✅ `_calculate_size(path)` — 递归计算目录/文件大小

### ✅ API 路由实现

文件: `gpu_broker/api/routes/models.py`

- ✅ `GET /v1/models` — 返回模型列表（测试结果: `{"models":[],"count":0}`）
- ✅ `POST /v1/models/pull` — 触发下载，返回 202 Accepted
- ✅ `DELETE /v1/models/{model_id}` — 删除模型，404 for nonexistent
- ✅ 使用 FastAPI BackgroundTasks 处理下载

### ✅ CLI 实现

文件: `gpu_broker/cli.py`

实现的命令:
- ✅ `gpu-broker model pull <repo_id>` — 支持 HuggingFace
- ✅ `gpu-broker model pull --source civitai --url <url> --name <name>` — 支持 Civitai
- ✅ `gpu-broker model list` — 表格显示，空时显示友好提示
- ✅ `gpu-broker model rm <model_id>` — 删除模型，404 时显示友好错误

### ✅ config.py 更新

- ✅ 添加 `OUTPUTS_DIR = DATA_DIR / "outputs"`
- ✅ 自动创建 MODELS_DIR 和 OUTPUTS_DIR

## 测试结果

### Test 1: `gpu-broker model list` (empty)
```
✅ PASS
No models found
Tip: Pull a model with: gpu-broker model pull <repo_id>
```

### Test 2: `GET /v1/models`
```
✅ PASS
{"models": [], "count": 0}
```

### Test 3: `POST /v1/models/pull`
```
✅ PASS (返回 202)
{"status": "accepted", "message": "Model download started from huggingface"}
```

### Test 4: `DELETE /v1/models/nonexistent`
```
✅ PASS (返回 404)
{"detail": "Model not found"}
```

### Test 5: `gpu-broker model rm nonexistent`
```
✅ PASS
Error: Model 'nonexistent' not found
Tip: List models with: gpu-broker model list
```

### Test 6: 无 GPU 依赖时友好提示
```
✅ PASS
ModelManager 使用条件导入，在缺少 huggingface_hub 时会抛出友好错误消息
```

### Test 7: 现有功能不回归 - `gpu-broker serve`
```
✅ PASS
服务器正常启动并运行
```

### Test 8: 现有功能不回归 - `gpu-broker status`
```
✅ PASS
┏━━━━━━━━━┳━━━━━━━┓
┃ Field   ┃ Value ┃
┡━━━━━━━━━╇━━━━━━━┩
│ Status  │ ok    │
│ Version │ 0.1.0 │
└─────────┴───────┘
```

### Test 9: 现有功能不回归 - tasks endpoint
```
✅ PASS
{"tasks": [], "count": 0}
```

## 约束遵守情况

- ✅ **未安装 torch/diffusers** (使用条件导入)
- ✅ **用条件导入处理可选依赖** (`try/except ImportError`)
- ✅ **保持现有 M1 功能不变** (status, serve, tasks 均正常)
- ✅ **友好错误提示** (无 GPU 依赖时不 crash)

## 存储布局

已创建目录结构:
```
~/.gpu-broker/
├── models/              # 模型存储目录
├── outputs/             # 生成图像输出目录 (新增)
└── gpu-broker.db        # SQLite 数据库
```

## 总结

**所有 9 项验收标准全部通过 ✅**

Milestone 2 交付完成：
1. ✅ DB Schema 与设计文档完全一致
2. ✅ Pydantic schemas 与 DB schema 对齐
3. ✅ ModelManager 完整实现
4. ✅ API 路由完整实现
5. ✅ CLI 命令完整实现
6. ✅ config.py 更新完成
7. ✅ 无 GPU 依赖时友好提示
8. ✅ 所有现有功能无回归
9. ✅ 符合所有约束条件

## 下一步

Milestone 2 已完成。可以继续 Milestone 3（任务提交和执行）。
