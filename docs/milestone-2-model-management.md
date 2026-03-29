# gpu-broker Milestone 2 - 模型管理功能

## 新增功能

### CLI 命令

#### 列出模型
```bash
gpu-broker model list
gpu-broker model ls  # 别名
```

#### 下载 HuggingFace 模型
```bash
gpu-broker model pull runwayml/stable-diffusion-v1-5
gpu-broker model pull stabilityai/stable-diffusion-xl-base-1.0
```

#### 下载 Civitai 模型
```bash
gpu-broker model pull --source civitai \
  --url "https://civitai.com/api/download/models/XXXXX" \
  --name "dreamshaper_8"
```

#### 删除模型
```bash
gpu-broker model rm <model_id>
```

### HTTP API

#### 列出所有模型
```bash
GET /v1/models
```

响应示例:
```json
{
  "models": [
    {
      "id": "stable-diffusion-v1-5",
      "name": "stable-diffusion-v1-5",
      "source": "huggingface",
      "source_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5",
      "path": "/home/user/.gpu-broker/models/stable-diffusion-v1-5",
      "format": "diffusers",
      "size_bytes": 4265820160,
      "type": "checkpoint",
      "trigger_words": null,
      "pulled_at": "2026-03-29T08:00:00",
      "loaded": false
    }
  ],
  "count": 1
}
```

#### 获取单个模型信息
```bash
GET /v1/models/{model_id}
```

#### 下载模型 (异步)
```bash
POST /v1/models/pull
Content-Type: application/json

{
  "source": "huggingface",
  "repo_id": "runwayml/stable-diffusion-v1-5"
}
```

或 Civitai:
```json
{
  "source": "civitai",
  "url": "https://civitai.com/api/download/models/XXXXX",
  "filename": "dreamshaper_8"
}
```

响应 (202 Accepted):
```json
{
  "status": "accepted",
  "message": "Model download started from huggingface"
}
```

#### 删除模型
```bash
DELETE /v1/models/{model_id}
```

响应:
```json
{
  "status": "deleted",
  "model_id": "stable-diffusion-v1-5"
}
```

## 数据库 Schema

### models 表
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

### tasks 表
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

## 存储布局

```
~/.gpu-broker/
├── models/
│   ├── stable-diffusion-v1-5/   # Diffusers format (directory)
│   └── dreamshaper_8.safetensors # Single-file checkpoint
├── outputs/                      # Generated images (M3+)
└── gpu-broker.db                 # SQLite database
```

## 错误处理

### 缺少 GPU 依赖
如果 `huggingface_hub` 不可用，下载会失败并给出友好提示：
```
RuntimeError: huggingface_hub is not available. 
Install GPU dependencies with: pip install gpu-broker[gpu]
```

### 模型不存在
```bash
$ gpu-broker model rm nonexistent
Error: Model 'nonexistent' not found
Tip: List models with: gpu-broker model list
```

### 服务器未运行
```bash
$ gpu-broker model list
Error: Cannot connect to http://localhost:7878/v1/models
Is the server running? Try: gpu-broker serve
```

## Python API 示例

```python
from gpu_broker.models.manager import ModelManager
from gpu_broker.config import DB_PATH, MODELS_DIR

# 初始化管理器
manager = ModelManager(DB_PATH, MODELS_DIR)

# 下载 HuggingFace 模型
model_info = manager.pull(
    source='huggingface',
    repo_id='runwayml/stable-diffusion-v1-5'
)

# 下载 Civitai 模型
model_info = manager.pull(
    source='civitai',
    url='https://civitai.com/api/download/models/XXXXX',
    filename='dreamshaper_8'
)

# 列出所有模型
models = manager.list()

# 获取单个模型
model = manager.get('stable-diffusion-v1-5')

# 删除模型
success = manager.delete('stable-diffusion-v1-5')
```

## 已知限制 (M2)

- 下载是后台异步的，没有进度反馈（M3+ 可能加入）
- 模型加载状态 (`loaded`) 始终为 `false`（需要 Engine 实现）
- 只支持 text-to-image checkpoint 类型（M3+ 扩展）
- 没有模型验证或 hash 检查

## 下一步 (M3)

- 任务提交和执行
- 实际的 Diffusers pipeline 集成
- 图像生成和输出管理
- 任务队列和调度
