# 多模态客服智能体

基于赛题要求设计的多模态客服智能体系统，具备多模态感知、精准理解、RAG知识增强、多轮对话与幻觉抑制能力。

## 功能特性

### 核心能力

- **多模态理解**: 支持文本和图片输入，准确识别用户意图
- **RAG知识库**: 基于20万字说明书构建，支持文本和图片检索
- **多轮对话**: 管理会话历史，支持上下文理解
- **幻觉抑制**: 通过思维链推理和上下文验证减少幻觉

### 技术亮点

- 思维链(Chain-of-Thought)问题拆解
- 多模态语义检索与重排序
- 基于上下文的答案验证
- 多问题逐一回答机制

## 项目结构

```
multimodal-agent/
├── src/
│   ├── modules/
│   │   ├── multimodal_understanding.py  # 多模态理解
│   │   ├── rag_engine.py               # RAG引擎
│   │   ├── conversation_manager.py       # 对话管理
│   │   ├── hallucination_controller.py  # 幻觉抑制
│   │   └── response_generator.py        # 回答生成
│   ├── utils/
│   │   ├── image_utils.py              # 图片处理
│   │   └── text_utils.py              # 文本处理
│   ├── api.py                         # API接口
│   └── main.py                        # 主入口
├── scripts/
│   └── build_knowledge_base.py        # 知识库构建脚本
├── 手册/                               # 说明书数据
├── docs/                               # 技术文档
├── tests/                              # 测试代码
└── config/                             # 配置文件
```

## 快速开始

### 环境要求

- Python 3.10+
- 16GB+ RAM (CPU模式)

### 安装依赖

默认先安装轻量核心依赖，避免在 Cursor 终端中因 `torch/transformers` 导致内存峰值过高。

```bash
cd D:\multimodal-agent
pip install -r requirements.txt
```

如果后续需要更强的多模态视觉能力或 `sentence-transformers` 语义向量，再额外安装：

```bash
pip install -r requirements-vision.txt
```

### 配置

创建 `.env` 文件：

```env
# API认证配置 (赛题要求)
API_TOKEN=sk_customer_20260304

# LLM配置
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4-vision-preview

# Embedding配置
EMBEDDING_BACKEND=hashing
EMBEDDING_MODEL=sentence-transformers/multi-qa-MiniLM-L6-cos-v1

# 低内存推荐
ENABLE_VISION_MODEL=false
EAGER_INIT_RAG=false
EAGER_INIT_MULTIMODAL=false
EAGER_INIT_RESPONSE_GENERATOR=false
```

### 构建知识库

```bash
# 语义向量方式构建
python -m scripts.build_knowledge_base --mode build --backend sentence_transformer

# 或添加赛题示例数据 (用于测试)
python -m scripts.build_knowledge_base --mode sample

# 查看统计信息
python -m scripts.build_knowledge_base --mode stats
```

### 运行

#### 启动API服务

```bash
python -m src.main api
# 或
python src/api.py
```

说明：

- 默认采用按需加载，启动时不会立刻加载全部重型模型。
- 如果只做知识库构建或接口联调，建议先保持 `EMBEDDING_BACKEND=hashing` 和 `ENABLE_VISION_MODEL=false`。

服务将在 `http://localhost:8000` 启动，API文档：`http://localhost:8000/docs`

## 赛题接口规范

本项目严格遵循赛题接口规范。

### 认证要求

所有请求必须在HTTP头中携带认证令牌：

```
Authorization: Bearer sk_customer_20260304
```

### POST /chat

**请求示例:**

```json
{
    "question": "我的DCB107电钻指示灯闪烁代表什么？",
    "images": ["data:image/png;base64,..."],
    "session_id": "kf_session_889900",
    "stream": false
}
```

**响应示例:**

```json
{
    "code": 0,
    "msg": "success",
    "data": {
        "answer": "DCB107电池组指示灯闪烁表示：\n1. 充电中...",
        "session_id": "kf_session_889900",
        "timestamp": 1741008000
    }
}
```

### 其他接口

- `GET /health` - 健康检查
- `POST /session/create` - 创建会话
- `POST /session/{session_id}` - 获取会话
- `DELETE /session/{session_id}` - 删除会话
- `POST /knowledge/add` - 添加知识
- `POST /knowledge/retrieve` - 检索知识
- `POST /knowledge/build` - 构建索引

### 调用示例

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    headers={
        "Authorization": "Bearer sk_customer_20260304",
        "Content-Type": "application/json"
    },
    json={
        "question": "我的电钻指示灯闪烁代表什么？",
        "images": []
    }
)

print(response.json())
```


## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行API测试
pytest tests/test_api.py -v

# 运行模块测试
pytest tests/test_modules.py -v
```

## 技术栈

- **FastAPI** - Web框架
- **LangChain** - LLM应用框架
- **FAISS** - 向量数据库
- **Sentence-Transformers** - 文本嵌入
- **CLIP** - 多模态模型
- **Pydantic** - 数据验证

## 许可证

MIT License
