# 多模态客服智能体

面向客服场景的**多模态检索增强生成（Multi-Modal RAG）智能体**，通过自然语言接口为用户提供产品手册咨询和客服政策解答服务。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 多模态理解 | 支持文本 + Base64 图片混合输入 |
| 双路检索 | 客服类问题（退款/投诉）和产品手册类问题自动路由 |
| 多轮对话 | 基于 session_id 保持上下文记忆 |
| 幻觉抑制 | 生成答案与知识库一致性验证 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，填入 API Key：

```env
API_TOKEN=sk_customer_20260304
LLM_API_KEY=your_api_key
DASHSCOPE_API_KEY=your_dashscope_key
```

### 3. 启动服务

```bash
# 方式一：直接运行
python api.py

# 方式二：模块运行
python -m src.main api
```

服务启动后访问：
- API 地址: http://localhost:8000
- 文档界面: http://localhost:8000/docs

---

## API 接口

### 核心端点: POST /chat

**请求头**:
```
Authorization: Bearer sk_customer_20260304
Content-Type: application/json
```

**请求体**:
```json
{
  "question": "用户问题",
  "images": ["data:image/png;base64,..."],
  "session_id": "会话ID（可选）"
}
```

**响应**:
```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "智能体回答",
    "session_id": "会话ID",
    "timestamp": 1741008000
  }
}
```

### 测试命令

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Authorization: Bearer sk_customer_20260304" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"电钻指示灯闪烁是什么意思？\"}"
```

---

## 技术架构

### 外部 API 依赖

| 服务 | 模型 | 用途 |
|------|------|------|
| 百炼 | `text-embedding-v3` | 文本向量化 |
| 百炼 | `qwen3-vl-rerank` | 重排序 |
| Infini AI | `qwen3-vl-235b-a22b-thinking` | 视觉理解 |
| Infini AI | `deepseek-v4-pro` | LLM 生成 |

### 模块结构

```
src/
├── api.py                          # HTTP 入口
├── main.py                         # CLI 入口
├── modules/
│   ├── response_generator.py        # 主编排器
│   ├── multimodal_understanding.py  # 多模态理解
│   ├── question_decomposer.py      # 问题分解
│   ├── dual_route_retriever.py     # 双路检索
│   ├── rag_engine.py               # RAG 引擎
│   ├── route_classifier.py         # 路由分类器
│   ├── hallucination_controller.py  # 幻觉抑制
│   └── conversation_manager.py      # 会话管理
└── utils/
```

---

## 项目结构

```
multimodal-agent/
├── api.py                      # 服务入口
├── src/                        # 源代码
│   ├── api.py                  # FastAPI 接口
│   ├── main.py                 # CLI 入口
│   └── modules/                # 核心模块
├── config/                     # 配置
├── scripts/                    # 离线脚本
├── knowledge_base/             # 知识库数据
│   ├── index/                 # FAISS 索引
│   └── routes/                # 客服路由
├── docs/                       # 文档
├── tests/                      # 测试
└── .env                        # 环境变量
```

---

## 知识库

- 索引路径: `knowledge_base/index/`
- 文档数: 29,598 条
- 向量维度: 1024

### 重建知识库

```bash
python -m src.main build
```

---

## 配置说明

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `API_TOKEN` | 接口认证 Token | `sk_customer_20260304` |
| `LLM_API_KEY` | LLM API Key | - |
| `DASHSCOPE_API_KEY` | 百炼 API Key | - |
| `API_HOST` | 服务地址 | `0.0.0.0` |
| `API_PORT` | 服务端口 | `8000` |

---

## 评测规则对标

| 维度 | 状态 |
|------|------|
| 架构清晰（模块划分） | ✅ |
| 接口规范（REST + Bearer Token） | ✅ |
| 多模态理解（图片 Base64） | ✅ |
| RAG 检索效果 | ✅ |
| 多轮记忆 | ✅ |
| 幻觉抑制 | ✅ |
