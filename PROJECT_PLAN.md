# 项目计划（按当前代码状态更新）

## 1. 项目目标

围绕比赛要求，把当前项目整理为可复现、可评测、可提交的多模态客服智能体：

- 在线侧：标准 `/chat` 接口、多轮对话、双路检索、回答生成、幻觉抑制。
- 离线侧：手册知识库构建、客服路由知识库构建、公开题批量答题、检索与路由评测。
- 交付侧：源码、接口文档、技术文档、验证报告四类初赛材料保持一致。

## 2. 当前实现状态

### 核心链路

- `API 接口`：已实现
  - `src/api.py` 提供 `/chat`、`/health`、`/session/*`、`/knowledge/*`
  - Bearer Token 鉴权已实现
- `多模态理解`：已实现基础版
  - `src/modules/multimodal_understanding.py`
  - 输出结构化结果并传入后续模块
- `问题分解`：已实现
  - `src/modules/question_decomposer.py`
  - 规则优先，LLM 按需增强
- `双路检索`：已实现
  - `src/modules/dual_route_retriever.py`
  - 支持 `service / manual / mixed`
- `RAG 引擎`：已实现
  - `src/modules/rag_engine.py`
  - 支持 `hashing` 和 `sentence_transformer`
- `多轮对话`：已实现
  - `src/modules/conversation_manager.py`
- `幻觉验证 / 修正`：已实现
  - `src/modules/hallucination_controller.py`

### 离线能力

- `手册知识库构建`：已实现
  - `scripts/build_knowledge_base.py`
- `客服路由知识库构建`：已实现
  - `scripts/build_dual_route_kb.py`
- `公开题批量生成`：已实现
  - `scripts/generate_public_answers.py`
- `检索 / 路由 / 压测脚本`：已实现
  - `scripts/evaluate_retrieval.py`
  - `scripts/evaluate_dual_route.py`
  - `scripts/benchmark_chat.py`

## 3. 当前与比赛正式提交之间的差距

### P0：必须补齐

1. `评测结果统一回归`
   - 需要在最新代码和最新 `.env` 配置下重跑：
     - 路由评测
     - 检索评测
     - 公开题批量生成
   - 产出本轮正式基线结果目录

2. `材料一致性`
   - README、技术文档、验证报告、接口说明必须与当前实现对齐
   - 避免把未实现能力写成已完成能力

3. `环境可复现`
   - `.env.example`
   - `requirements.txt`
   - 启动与建库命令
   - 模型缓存全部指向 `D:` 盘

### P1：直接影响得分的优化

1. `手册检索质量`
   - 重点关注候选手册识别、局部召回、最终重排
2. `图文对应质量`
   - 当前以 `<PIC>[image_id]` 文本绑定为主，仍需提升图像证据利用率
3. `多问题回答结构`
   - 确保一次多问逐项作答，减少漏答
4. `幻觉抑制稳定性`
   - 验证器与修正器超时、降级和保守回答策略继续打磨

### P2：正式交付完善

1. 导出 PDF 技术文档与验证报告
2. 整理正式演示话术与 PPT 结构
3. 形成一版固定的比赛提交流程脚本

## 4. 近期执行顺序

### 阶段 A：基线冻结

- [ ] 检查 `.env` 与模型缓存目录
- [ ] 重建客服路由知识库
- [ ] 重建手册知识库
- [ ] 生成一轮公开题答案
- [ ] 记录输出目录与关键指标

### 阶段 B：指标提升

- [ ] 分析高风险题与 fallback 题
- [ ] 分析 manual route 误召回样本
- [ ] 调整手册候选识别和 rerank 配置
- [ ] 回归公开题结果

### 阶段 C：材料定稿

- [ ] 技术文档定稿
- [ ] 验证报告回填真实结果
- [ ] README / 接口说明最终检查
- [ ] 导出 PDF

## 5. 交付物检查清单

### 初赛四项材料

- [ ] 智能体 API 接口
- [ ] 完整源码包
- [ ] 技术文档（PDF）
- [ ] 验证报告（PDF）

### 决赛扩展材料

- [ ] 优化报告
- [ ] 演示 PPT / PDF
- [ ] 优化后的 API / 源码 / 文档 / 验证报告

## 6. 风险提示

1. 当前 `/chat` 未实现 `stream` 字段兼容，如比赛评测严格校验字段，需要补齐。
2. 当前图片检索仍以文本 chunk 中的图片 ID 绑定为主，若赛题图像理解题占比高，需继续增强图片侧证据召回。
3. 验证报告必须基于最新脚本实跑结果回填，不能直接沿用旧目录中的历史数字。
