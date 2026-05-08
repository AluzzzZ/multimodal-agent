# 多模态客服智能体

面向比赛场景的多模态客服智能体原型，当前代码围绕“多模态理解 -> 问题分解 -> 双路检索 -> 回答生成 -> 幻觉验证/修正”主链路实现，并提供离线知识库构建、检索评测、批量答题和路由分类器训练脚本。

## 当前实现概览

### 已实现

- `POST /chat` 标准 RESTful 接口，Bearer Token 认证
- 文本 + Base64 图片输入
- 多轮会话管理
- 多模态理解结果结构化输出（图片标签、候选产品、证据类型）
- 结构化问题分解（规则优先，LLM 按需补充）
- `service / manual / mixed` 双路检索
- 手册知识库构建（手册正文 + `<PIC>` 绑定图片 ID）
- 小模型幻觉验证与修正
- 公开题批量生成、检索评测、路由评测与性能压测脚本

### 需要注意的当前边界

- `/chat` 当前已实现 `question`、`images`、`session_id` 三个核心字段，`stream` 还未接入代码层。
- 当前知识库主形态是“文本 chunk + 图片 ID 绑定”，图片索引能力在底层留有接口，但离线构建脚本默认只写入文本知识。
- 评测报告与技术文档应基于最新脚本实跑结果回填，仓库中的历史评测目录不能直接视为最终提交结果。

## 项目结构

```text
D:\multimodal-agent
├── src
│   ├── api.py                          # FastAPI 接口层
│   ├── main.py                         # 启动入口（api/build/interactive/test）
│   ├── modules
│   │   ├── multimodal_understanding.py # 多模态理解
│   │   ├── question_decomposer.py      # 结构化问题分解
│   │   ├── route_classifier.py         # ONNX 路由分类器
│   │   ├── dual_route_retriever.py     # 双路检索编排
│   │   ├── rag_engine.py               # 向量检索与知识库底座
│   │   ├── hallucination_controller.py # 幻觉验证/修正
│   │   ├── response_generator.py       # 在线主编排器
│   │   └── conversation_manager.py     # 会话管理
│   └── utils
│       ├── text_utils.py
│       ├── image_utils.py
│       └── llm_corrector.py
├── scripts                             # 离线构建、评测、训练脚本
├── knowledge_base                      # 知识库索引、路由知识、评测产物
├── 手册                                # 赛题原始手册文件
├── docs                                # 技术文档、架构图、验证报告
├── tests                               # API 与模块测试
├── config                              # 全局配置
├── .env.example
├── PROJECT_PLAN.md
└── 架构说明稿.md
```

## 在线主链路

```text
/chat
  -> api.py
  -> ConversationManager
  -> ResponseGenerator.generate()
      -> MultimodalUnderstanding.analyze()
      -> QuestionDecomposer.decompose()
      -> DualRouteRetriever.retrieve()
          -> RouteClassifier.predict()
          -> service 检索 / manual 检索 / mixed 合并
          -> RAGEngine.retrieve()
      -> 逐子问题回答与聚合
      -> LLMHallucinationVerifier.verify()
      -> LLMHallucinationVerifier.refine()（按需）
  -> 标准响应
```

## 环境要求

- Python `3.10+`
- Windows / Linux 均可
- 推荐内存：
  - 仅构建轻量索引：`8GB+`
  - 使用 `sentence-transformer + reranker + 多模态理解`：`16GB+`

## 安装

### 1. 创建虚拟环境

```powershell
cd D:\multimodal-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

## 配置

复制环境变量模板并按需修改：

```powershell
Copy-Item .env.example .env
```

### 推荐比赛配置

- 使用 Infini OpenAI 兼容接口时：
  - `LLM_PROVIDER=openai`
  - `LLM_BASE_URL=https://cloud.infini-ai.com/maas/v1`
  - `LLM_MODEL=deepseek-v4-pro`
- 构建知识库时如果终端内存紧张：
  - `EMBEDDING_BACKEND=hashing`
  - `EMBEDDING_DEVICE=cpu`
- 所有模型缓存统一放到 `D:` 盘：
  - `HF_HOME=D:\model_cache\huggingface`
  - `SENTENCE_TRANSFORMERS_HOME=D:\model_cache\sentence_transformers`
  - `TORCH_HOME=D:\model_cache\torch`

完整示例见 `.env.example`。

## 构建知识库

### 1. 构建客服路由知识库

```powershell
python -m scripts.build_dual_route_kb
```

### 2. 构建手册知识库

低内存优先：

```powershell
python -m scripts.build_knowledge_base --mode build --backend hashing --force
```

高质量语义检索：

```powershell
python -m scripts.build_knowledge_base --mode build --backend sentence_transformer --force
```

### 3. 查看知识库统计

```powershell
python -m scripts.build_knowledge_base --mode stats
```

## 启动服务

```powershell
python -m src.main api
```

启动后可访问：

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- 健康检查: `http://127.0.0.1:8000/health`

## 接口说明

### `POST /chat`

#### 请求头

```http
Authorization: Bearer sk_customer_20260304
Content-Type: application/json
```

#### 请求体

```json
{
  "question": "我的DCB107或DCB112型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？",
  "images": ["data:image/png;base64,..."],
  "session_id": "kf_session_889900"
}
```

#### 响应体

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "...",
    "session_id": "kf_session_889900",
    "timestamp": 1741008000
  }
}
```

### 示例调用

```powershell
curl --request POST `
  --url http://127.0.0.1:8000/chat `
  --header "Authorization: Bearer sk_customer_20260304" `
  --header "Content-Type: application/json" `
  --data '{
    "question": "物流一直显示待揽收，是什么原因？",
    "images": [],
    "session_id": "demo-session-001"
  }'
```

## 常用脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/build_dual_route_kb.py` | 构建客服路由知识库 |
| `scripts/build_knowledge_base.py` | 构建手册知识库索引 |
| `scripts/evaluate_retrieval.py` | 评估检索效果 |
| `scripts/evaluate_dual_route.py` | 评估双路路由策略 |
| `scripts/generate_public_answers.py` | 批量生成公开题答案和提交文件 |
| `scripts/benchmark_chat.py` | API 性能压测 |
| `scripts/scan_route_thresholds.py` | 路由阈值扫描 |
| `scripts/build_route_classifier_dataset.py` | 构建路由分类器训练数据 |
| `scripts/train_route_classifier.py` | 训练并导出 ONNX 路由分类器 |
| `scripts/diagnose_manual_route.py` | 诊断手册路由 / 局部召回问题 |
| `scripts/process_manuals_with_model.py` | 用大模型辅助整理手册结构 |
| `scripts/convert_model_kb_to_payload.py` | 将模型整理后的手册转换为建库 payload |
| `scripts/audit_route_dataset.py` | 审核路由训练样本质量 |
| `scripts/sample_data.py` | 数据采样与抽检 |

## 评测与提交文件

### 公开题批量生成

```powershell
python -m scripts.generate_public_answers --questions question_public.csv --output-dir knowledge_base\evaluation\latest_run
```

输出目录通常包含：

- `submission_public_generated.csv`
- `public_answer_detail.csv`
- `public_answer_summary.json`
- `public_answer_summary.md`

### 检索评测

```powershell
python -m scripts.evaluate_retrieval
```

### 路由评测

```powershell
python -m scripts.evaluate_dual_route
```

## 测试

```powershell
pytest tests -v
```

## 文档索引

- `架构说明稿.md`
- `docs/项目技术架构文档.md`
- `docs/手册检索流程细化图.md`
- `docs/技术文档.md`
- `docs/验证报告.md`

## 当前最值得优先推进的事项

1. 用最新配置重跑一次知识库构建与公开题批量生成，形成可追踪评测基线。
2. 如果目标是比赛正式提交，补齐 `stream` 字段兼容、真实验证截图和 PDF 化材料。
3. 如果目标是冲分，优先继续优化手册检索、图片相关证据召回和回答结构化质量。
