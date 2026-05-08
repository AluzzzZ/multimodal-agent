"""
回答生成模块
负责综合检索结果生成最终回答

核心流程:
1. 问题分解 (CoT) -> 2. RAG检索 -> 3. 上下文构建 -> 4. 回答生成 -> 5. 幻觉检测 -> 6. 图片关联
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from config import settings
from .rag_engine import RAGEngine, get_rag_engine
from .dual_route_retriever import DualRouteRetriever, get_dual_route_retriever
from .hallucination_controller import (
    HallucinationController,
    ChainOfThoughtReasoner,
    get_hallucination_controller,
    get_cot_reasoner
)


class ResponseGenerator:
    """
    回答生成器

    核心职责:
    - 整合RAG检索结果与对话历史
    - 通过思维链(CoT)推理拆解复杂问题
    - 多问题逐一回答，保证答案完整性
    - 幻觉检测与上下文一致性验证
    - 智能关联相关图片引用
    """

    def __init__(self):
        # LLM客户端 - 用于生成回答
        self.llm_client = None
        # RAG引擎 - 用于知识检索
        self.rag_engine: Optional[RAGEngine] = None
        self.dual_route_retriever: Optional[DualRouteRetriever] = None
        # 幻觉控制器 - 用于答案验证
        self.hallucination_controller: Optional[HallucinationController] = None
        # 思维链推理器 - 用于问题拆解
        self.cot_reasoner: Optional[ChainOfThoughtReasoner] = None
        self._initialized = False

    def initialize(self):
        """
        初始化所有组件

        按照以下顺序初始化:
        1. LLM客户端 (根据配置选择OpenAI或本地模型)
        2. RAG引擎 (知识检索)
        3. 幻觉控制器 (答案验证)
        4. 思维链推理器 (问题拆解)
        """
        if self._initialized:
            return

        # Step 1: 初始化LLM客户端
        # 支持OpenAI API和本地Ollama模型两种provider
        try:
            if settings.llm_provider == "openai":
                from langchain_openai import ChatOpenAI
                self.llm_client = ChatOpenAI(
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens
                )
            elif settings.llm_provider == "local":
                from langchain_community.chat_models import ChatOllama
                self.llm_client = ChatOllama(
                    model=settings.llm_model,
                    temperature=settings.llm_temperature
                )
            else:
                self.llm_client = None
                logger.warning(f"未启用外部LLM生成，当前 provider={settings.llm_provider}，将使用检索驱动兜底回答")

            if self.llm_client is not None:
                logger.info("回答生成器LLM客户端初始化成功")
        except Exception as e:
            logger.warning(f"LLM客户端初始化失败，将使用检索驱动兜底回答: {e}")
            self.llm_client = None

        # Step 2: 初始化其他组件
        self.rag_engine = get_rag_engine()
        self.rag_engine.initialize()

        self.dual_route_retriever = get_dual_route_retriever()
        self.dual_route_retriever.initialize()

        self.hallucination_controller = get_hallucination_controller()
        self.hallucination_controller.initialize()

        self.cot_reasoner = get_cot_reasoner()
        self.cot_reasoner.initialize()

        self._initialized = True
        logger.info("回答生成器初始化完成")

    def generate(
        self,
        query: str,
        images: Optional[List[str]] = None,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        生成回答 - 主流程

        处理流程:
        1. 问题分解: 使用思维链将复杂问题拆分为多个简单问题
        2. RAG检索: 对每个子问题独立检索知识库
        3. 上下文构建: 整合检索结果、对话历史、额外上下文
        4. 回答生成:
           - 单一问题直接生成
           - 多问题逐一生成后综合
        5. 幻觉检测: 验证答案与上下文的一致性
        6. 图片关联: 提取检索结果中的相关图片ID

        Args:
            query: 用户问题文本
            images: 用户上传的Base64图片列表(可选)
            context: 额外上下文信息(可选)
            conversation_history: 对话历史列表(可选)

        Returns:
            包含以下字段的字典:
            - response: 生成的回复文本
            - images: 相关的图片ID列表
            - sources: 检索来源列表
            - reasoning: 思维链推理结果
            - confidence: 回答置信度
        """
        if not self._initialized:
            self.initialize()

        # 初始化结果容器
        result = {
            "response": "",
            "images": [],
            "sources": [],
            "reasoning": None,
            "confidence": 0.0,
            "routes": []
        }

        # ========== Step 1: 问题分解 (思维链) ==========
        # 复杂问题拆分为多个简单子问题，保证每个子问题都能被完整回答
        # 例如: "能送到乡镇吗？需要加运费吗？多久到？" -> 拆分为3个子问题
        sub_questions = self._split_simple_questions(query)
        if settings.enable_cot_reasoning and self._needs_deep_reasoning(query, sub_questions):
            decomposition = self.cot_reasoner.decompose_question(query)
            result["reasoning"] = decomposition
            if decomposition.get("is_complex"):
                sub_questions = decomposition.get("sub_questions", sub_questions or [query])
        else:
            result["reasoning"] = {
                "original_question": query,
                "sub_questions": sub_questions,
                "reasoning_steps": [],
                "is_complex": len(sub_questions) > 1,
                "mode": "fast_split"
            }

        # ========== Step 2: RAG检索 ==========
        # 对每个子问题独立检索，收集所有相关来源
        # 去重处理避免重复内容
        all_sources = []
        retrieved_images = []
        route_packets = []

        for sq in sub_questions:
            route_result = self.dual_route_retriever.retrieve(sq, images=images)
            retrieval_result = route_result["results"]
            route_packets.append(
                {
                    "question": sq,
                    "route_info": route_result["route_info"],
                    "service_results": route_result["service_results"],
                    "manual_results": route_result["manual_results"],
                    "results": retrieval_result,
                }
            )
            result["routes"].append(
                {
                    "question": sq,
                    **route_result["route_info"],
                }
            )

            for item in retrieval_result:
                # 基于doc_id去重，保留首次出现的结果
                if item["doc_id"] not in [s["doc_id"] for s in all_sources]:
                    all_sources.append(item)
                    if item.get("image_ids"):
                        retrieved_images.extend(item["image_ids"])

        # ========== Step 3: 构建上下文 ==========
        # 整合检索到的知识、对话历史、额外上下文
        context_text = self._build_context(
            all_sources,
            conversation_history,
            context,
            result["routes"],
            route_packets=route_packets,
        )

        # ========== Step 4: 生成回答 ==========
        # 无论单问题还是多问题，都尽量压缩为一次生成调用，减少时延
        final_answer = self._generate_answer(
            query,
            sub_questions,
            context_text,
            all_sources=all_sources,
            route_records=result["routes"],
            route_packets=route_packets,
        )

        # ========== Step 5: 幻觉检测与修正 ==========
        # 验证答案是否与知识库上下文一致
        # 如果不一致，尝试修正答案
        if settings.hallucination_detection_enabled:
            verification = self.hallucination_controller.verify_against_context(
                final_answer,
                [s["content"] for s in all_sources]
            )

            if not verification.get("is_consistent", True):
                # 一致性验证失败，尝试修正答案
                refined_answer = self.hallucination_controller.refine_answer(
                    final_answer,
                    [s["content"] for s in all_sources],
                    sub_questions
                )
                final_answer = refined_answer

            result["confidence"] = verification.get("confidence", 0.0)
        else:
            # 禁用幻觉检测时使用默认值
            result["confidence"] = 0.7

        # ========== Step 6: 提取相关图片 ==========
        # 从检索结果中提取图片ID，最多返回5张
        result["images"] = list(set(retrieved_images))[:5]
        result["sources"] = all_sources
        result["response"] = final_answer

        return result

    def _split_simple_questions(self, query: str) -> List[str]:
        """
        简单问题拆分 - 备用方案(禁用CoT时使用)

        拆分策略:
        1. 按标点符号(？;；\n)分割文本
        2. 使用正则模式检测是否包含多个问题
        3. 返回拆分后的子问题列表

        Args:
            query: 用户问题

        Returns:
            子问题列表
        """
        # Step 1: 按标点拆分
        questions = re.split(r'[?？;；\n]', query)
        questions = [q.strip() for q in questions if q.strip()]
        questions = self._merge_followup_constraints(questions)

        # Step 2: 检测多问题模式
        # 常见的问题标记: "有...吗", "怎么...", "如何...", "可以...吗"
        multi_patterns = [
            r'有.*吗', r'怎么.*', r'如何.*',
            r'可以.*吗', r'请问.*', r'.*吗.*'
        ]
        has_multiple = sum(1 for p in multi_patterns if re.search(p, query)) > 1

        # Step 3: 判断是否需要拆分
        if len(questions) > 1 or has_multiple:
            return questions if questions else [query]

        return [query]

    def _merge_followup_constraints(self, questions: List[str]) -> List[str]:
        """
        将约束句合并到前一个主问题。

        例如: ["请问如何安装？", "只需告诉我前五条"] -> ["请问如何安装？，只需告诉我前五条"]

        合并规则:
        - 句子较短(<=18字符)或包含约束提示词时视为约束句
        - 问句(?结尾)和以疑问词开头的不合并
        """
        if len(questions) <= 1:
            return questions

        constraint_cues = (
            "只需", "只要", "告诉我", "列出", "写出", "前", "后", "即可", "分别", "依次", "简要"
        )
        question_openers = ("请问", "如何", "怎么", "什么", "哪些", "哪几", "是否", "能否", "可否", "要不要")

        merged: List[str] = []
        for question in questions:
            if (
                merged
                and (len(question) <= 18 or any(cue in question for cue in constraint_cues))
                and not question.endswith("吗")
                and not question.startswith(question_openers)
            ):
                merged[-1] = merged[-1].rstrip("，,。；; ") + "，" + question
            else:
                merged.append(question)
        return merged

    def _needs_deep_reasoning(self, query: str, sub_questions: List[str]) -> bool:
        """
        控制是否启用慢速深度拆解。

        快速规则拆分足以应对大多数场景，仅在明显复杂时才调用额外LLM，
        以控制客服接口时延。阈值设置基于公开题分析:
        - 子问题数>=3 明确是复合问题
        - 文本较长(>=120字符)且子问题>=2 说明是长复合问
        - 换行>=2 次表示多行复杂输入
        """
        if len(sub_questions) >= 3:
            return True
        if len(query) >= 120 and len(sub_questions) >= 2:
            return True
        if query.count("\n") >= 2:
            return True
        return False

    def _build_context(
        self,
        sources: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]],
        extra_context: Optional[str],
        route_records: Optional[List[Dict[str, Any]]] = None,
        route_packets: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        构建上下文文本 - 用于填充LLM的prompt

        上下文组成(按优先级):
        1. 【参考知识】- 来自RAG检索的知识库内容
        2. 【对话历史】- 最近3轮对话(保持上下文连贯)
        3. 【额外信息】- 外部传入的补充信息

        Args:
            sources: RAG检索结果列表
            history: 对话历史列表
            extra_context: 额外上下文字符串

        Returns:
            格式化的上下文文本
        """
        context_parts = []

        if route_records:
            route_summary = " / ".join(
                [f"{item['question']} -> {item['route']}" for item in route_records[:4]]
            )
            context_parts.append(f"【路由结果】{route_summary}")

        # 1. 添加知识库内容，优先按子问题组织 mixed 上下文
        if route_packets:
            for idx, packet in enumerate(route_packets, 1):
                route = packet["route_info"].get("route", "manual")
                context_parts.append(f"【子问题{idx} | 路由:{route}】{packet['question']}")

                service_sources = packet.get("service_results", [])
                manual_sources = packet.get("manual_results", [])

                if service_sources:
                    for source_idx, source in enumerate(service_sources[:2], 1):
                        title = source.get("metadata", {}).get("title", f"客服资料{source_idx}")
                        content = self._clean_context_content(source["content"])
                        if content.strip():
                            context_parts.append(f"客服参考{source_idx}: [{title}] {content.strip()}")

                if manual_sources:
                    for source_idx, source in enumerate(manual_sources[:2], 1):
                        manual_name = source.get("metadata", {}).get(
                            "manual_name",
                            source.get("metadata", {}).get("title", f"手册资料{source_idx}")
                        )
                        content = self._clean_context_content(source["content"])
                        if content.strip():
                            context_parts.append(f"手册参考{source_idx}: [{manual_name}] {content.strip()}")
        elif sources:
            service_sources = [s for s in sources if s.get("metadata", {}).get("route") == "service"]
            manual_sources = [s for s in sources if s.get("metadata", {}).get("route") != "service"]

            if service_sources:
                context_parts.append("【客服政策参考】")
                for i, source in enumerate(service_sources[:3], 1):
                    title = source.get("metadata", {}).get("title", f"客服资料{i}")
                    content = self._clean_context_content(source["content"])
                    if content.strip():
                        context_parts.append(f"{i}. [{title}] {content.strip()}")

            if manual_sources:
                context_parts.append("【产品手册参考】")
                for i, source in enumerate(manual_sources[:3], 1):
                    manual_name = source.get("metadata", {}).get("manual_name", source.get("metadata", {}).get("title", f"手册资料{i}"))
                    content = self._clean_context_content(source["content"])
                    if content.strip():
                        context_parts.append(f"{i}. [{manual_name}] {content.strip()}")

        # 2. 添加对话历史
        if history:
            context_parts.append("\n【对话历史】")
            for msg in history[-3:]:
                role = "用户" if msg.get("role") == "user" else "助手"
                context_parts.append(f"{role}: {msg.get('content', '')}")

        # 3. 添加额外上下文
        if extra_context:
            context_parts.append(f"\n【额外信息】{extra_context}")

        return "\n".join(context_parts)

    def _clean_context_content(self, content: str) -> str:
        """
        清理上下文中的图片 ID 标记，保留 <PIC> 位置提示。

        处理步骤:
        1. 移除所有 [xxx] 格式的图片 ID
        2. 合并多余空白字符

        注意: 保留 <PIC> 占位符本身，用于指示图片引用位置。
        <PIC> 通常嵌入在文本中作为图片引用标记。

        Args:
            content: 原始手册内容

        Returns:
            清理后的文本
        """
        content_clean = re.sub(r'\[([^\]]+)\]', '', content)
        return re.sub(r'\s+', ' ', content_clean).strip()

    def _generate_answer(
        self,
        query: str,
        sub_questions: List[str],
        context: str,
        all_sources: Optional[List[Dict[str, Any]]] = None,
        route_records: Optional[List[Dict[str, Any]]] = None,
        route_packets: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        生成回答。

        为了降低时延，多问题场景也尽量只进行一次模型调用。

        Args:
            query: 原始用户问题
            sub_questions: 拆分后的子问题列表
            context: 格式化后的上下文文本

        Returns:
            生成的回复文本
        """
        question_block = ""
        if len(sub_questions) > 1:
            numbered = "\n".join([f"{idx}. {sq}" for idx, sq in enumerate(sub_questions, 1)])
            question_block = f"\n【拆分后的子问题】\n{numbered}\n"

        prompt = f"""你是一个专业的多模态客服助手。请根据提供的参考资料，准确、清晰地回答用户问题。

【用户问题】
{query}
{question_block}

{context}

请生成回答，要求：
1. 准确基于参考资料
2. 如需包含图片，使用 <PIC> 标记
3. 如果用户一次问了多个问题，必须逐项回答，不能遗漏
4. 回答结构清晰，优先使用编号或分段
5. 如有不确定信息，明确说明，不要编造政策或细节

回答：
"""

        if self.llm_client is None:
            return self._generate_fallback_answer(
                query,
                sub_questions,
                all_sources or [],
                route_records or [],
                route_packets or [],
            )

        try:
            response = self.llm_client.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer.strip()
        except Exception as e:
            logger.error(f"回答生成失败，改用检索兜底: {e}")
            return self._generate_fallback_answer(
                query,
                sub_questions,
                all_sources or [],
                route_records or [],
                route_packets or [],
            )

    def _generate_fallback_answer(
        self,
        query: str,
        sub_questions: List[str],
        sources: List[Dict[str, Any]],
        route_records: List[Dict[str, Any]],
        route_packets: List[Dict[str, Any]],
    ) -> str:
        """无 LLM 或外部调用失败时的检索驱动兜底回答。"""
        if not sources:
            return "抱歉，我暂时没有检索到足够的参考信息。请补充订单号、商品型号或问题图片，我再继续帮您核实。"

        if route_packets:
            return self._compose_route_aware_answer(sub_questions, route_packets)

        route_sequence = [item.get("route", "manual") for item in route_records]
        primary_route = route_sequence[0] if route_sequence else "manual"
        service_sources = [s for s in sources if s.get("metadata", {}).get("route") == "service"]
        manual_sources = [s for s in sources if s.get("metadata", {}).get("route") != "service"]

        if primary_route == "service" and service_sources:
            return self._compose_service_answer(sub_questions, service_sources)
        if primary_route == "mixed":
            service_answer = self._compose_service_answer(sub_questions, service_sources) if service_sources else ""
            manual_answer = self._compose_manual_answer(sub_questions, manual_sources) if manual_sources else ""
            mixed_parts = [part for part in [service_answer, manual_answer] if part]
            return "\n\n".join(mixed_parts) if mixed_parts else self._compose_manual_answer(sub_questions, manual_sources)
        return self._compose_manual_answer(sub_questions, manual_sources or sources)

    def _compose_route_aware_answer(self, sub_questions: List[str], route_packets: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        clarify_fields: List[str] = []

        if len(sub_questions) > 1:
            lines.append("我先按你的问题逐项说明：")

        for idx, packet in enumerate(route_packets, 1):
            question = packet["question"]
            route_info = packet["route_info"]
            route = route_info.get("route", "manual")
            service_sources = packet.get("service_results", [])
            manual_sources = packet.get("manual_results", [])

            if route == "service":
                answer_line = self._compose_service_sub_answer(question, service_sources)
            elif route == "mixed":
                answer_line = self._compose_mixed_sub_answer(question, service_sources, manual_sources, route_info)
            else:
                answer_line = self._compose_manual_sub_answer(question, manual_sources, route_info)

            if len(sub_questions) > 1:
                lines.append(f"{idx}. 关于“{question}”：{answer_line}")
            else:
                lines.append(answer_line)

            clarify_fields.extend(self._collect_clarify_fields(service_sources))

        deduped_fields = self._deduplicate_items([field for field in clarify_fields if field])
        if deduped_fields:
            lines.append(f"为尽快帮你核实处理，建议补充：{'、'.join(deduped_fields[:6])}。")

        return "\n".join(lines).strip()

    def _compose_service_answer(self, sub_questions: List[str], service_sources: List[Dict[str, Any]]) -> str:
        top_sources = service_sources[:2]
        answer_points: List[str] = []
        clarify_fields: List[str] = []

        for source in top_sources:
            answer_points.extend(self._extract_section_items(source["content"], "标准答复"))
            clarify_fields.extend(self._extract_section_items(source["content"], "需核实信息"))

        answer_points = self._deduplicate_items(answer_points)[:5]
        clarify_fields = self._deduplicate_items([field.lstrip("- ").strip() for field in clarify_fields])[:5]

        lines: List[str] = []
        if len(sub_questions) > 1:
            lines.append("我先按你的问题逐项说明：")
            for idx, question in enumerate(sub_questions, 1):
                point = self._match_best_answer_point(question, answer_points)
                lines.append(f"{idx}. 关于“{question}”：{point}")
        else:
            ranked_points = self._rank_answer_points(sub_questions[0] if sub_questions else "", answer_points)
            for idx, point in enumerate(ranked_points[:3], 1):
                lines.append(f"{idx}. {point}")

        if clarify_fields:
            lines.append(f"为尽快帮你核实处理，建议补充：{'、'.join(clarify_fields)}。")

        return "\n".join(lines).strip()

    def _compose_service_sub_answer(self, question: str, service_sources: List[Dict[str, Any]]) -> str:
        answer_points: List[str] = []
        for source in service_sources[:2]:
            answer_points.extend(self._extract_section_items(source["content"], "标准答复"))
        answer_points = self._deduplicate_items(answer_points)
        return self._match_best_answer_point(question, answer_points)

    def _collect_clarify_fields(self, service_sources: List[Dict[str, Any]]) -> List[str]:
        """
        从客服检索结果中提取"需核实信息"字段，用于追问提示。

        客服文档中通常包含"需核实信息"字段，列出回答该问题时
        需要用户补充的信息（如订单号、商品型号等）。在回答末尾
        自动追加这些提示，有助于提升客服体验。

        Args:
            service_sources: 客服检索结果列表

        Returns:
            需核实信息字段列表（去除了前缀编号）
        """
        clarify_fields: List[str] = []
        for source in service_sources[:2]:
            clarify_fields.extend(self._extract_section_items(source["content"], "需核实信息"))
        return [field.lstrip("- ").strip() for field in clarify_fields if field.strip()]

    def _match_best_answer_point(self, question: str, answer_points: List[str]) -> str:
        """
        从回答要点列表中选择最匹配当前问题的要点。

        匹配策略: 按相似度和关键词命中数排序，取第一名。

        Args:
            question: 当前问题文本
            answer_points: 可选的回答要点列表

        Returns:
            最佳匹配的要点文本，若无匹配则返回默认回复
        """
        ranked = self._rank_answer_points(question, answer_points)
        if ranked:
            return ranked[0]
        return "需要结合订单和售后信息进一步核实。"

    def _rank_answer_points(self, question: str, answer_points: List[str]) -> List[str]:
        """
        对回答要点进行排序。

        排序维度:
        1. 词项相似度: 问题与要点的相似程度
        2. 直接关键词命中数: 要点中包含问题关键词的数量

        两个维度都是降序，取相似度为主排序键、命中数为次排序键。

        Returns:
            排序后的要点列表
        """
        scored = []
        for point in answer_points:
            scored.append((point, self._simple_similarity(question, point), self._direct_keyword_hits(question, point)))
        scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
        ranked = [point for point, _, _ in scored if point]
        return ranked or answer_points

    def _compose_manual_answer(self, sub_questions: List[str], manual_sources: List[Dict[str, Any]]) -> str:
        if not manual_sources:
            return "抱歉，我暂时没有检索到对应的产品手册内容。请补充更具体的商品型号、功能名称或上传相关图片。"

        lines: List[str] = []
        top_sources = manual_sources[: min(3, len(sub_questions) or 1)]
        for idx, question in enumerate(sub_questions or [""]):
            source = top_sources[min(idx, len(top_sources) - 1)]
            snippet = self._summarize_manual_snippet(source["content"])
            manual_name = source.get("metadata", {}).get("manual_name", source.get("metadata", {}).get("title", "产品手册"))
            if len(sub_questions) > 1:
                lines.append(f"{idx + 1}. 关于“{question}”：根据[{manual_name}]，{snippet}")
            else:
                lines.append(f"根据[{manual_name}]，{snippet}")

        return "\n".join(lines).strip()

    def _compose_manual_sub_answer(
        self,
        question: str,
        manual_sources: List[Dict[str, Any]],
        route_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not manual_sources:
            return "暂时还无法精确定位到对应的产品手册，请补充商品型号、功能名称或上传产品图，我再帮你继续定位。"

        source = self._select_best_manual_source(question, manual_sources)
        snippet = self._summarize_manual_snippet(source["content"])
        manual_name = source.get("metadata", {}).get("manual_name", source.get("metadata", {}).get("title", "产品手册"))
        image_hint = self._format_manual_visual_hint(source)
        structured_points = self._extract_structured_manual_points(question, manual_sources)

        if structured_points:
            answer = f"根据[{manual_name}]，相关要点如下："
            answer += " " + " ".join([f"{idx}. {point}" for idx, point in enumerate(structured_points, 1)])
        else:
            answer = f"根据[{manual_name}]，{snippet}"
        if image_hint:
            answer += f" {image_hint}"
        return answer

    def _compose_mixed_sub_answer(
        self,
        question: str,
        service_sources: List[Dict[str, Any]],
        manual_sources: List[Dict[str, Any]],
        route_info: Dict[str, Any],
    ) -> str:
        parts: List[str] = []

        service_answer = self._compose_service_sub_answer(question, service_sources)
        if service_answer:
            parts.append(service_answer)

        should_attach_manual = self._should_attach_manual_support(question, manual_sources, route_info)
        if should_attach_manual and manual_sources:
            manual_answer = self._compose_manual_sub_answer(question, manual_sources, route_info)
            parts.append(f"另外，如果你需要对应商品的说明书或图示，可参考：{manual_answer}")
        elif route_info.get("has_manual_instruction_intent"):
            parts.append("如果你需要对应商品的电子版说明书、具体页码或图示，请补充商品型号或上传产品图片，我可以继续帮你定位到具体手册内容。")

        return " ".join(parts).strip()

    def _should_attach_manual_support(
        self,
        question: str,
        manual_sources: List[Dict[str, Any]],
        route_info: Dict[str, Any],
    ) -> bool:
        if not manual_sources:
            return False

        top_source = manual_sources[0]
        top_score = float(top_source.get("relevance_score", 0.0))
        if top_score >= settings.mixed_manual_support_threshold:
            return True
        if route_info.get("manual_keyword_hits"):
            return True
        if top_source.get("image_ids"):
            return True
        if any(keyword in question for keyword in ("说明书", "手册", "图示", "配图", "电子版", "纸质版")):
            return True
        return False

    def _extract_section_items(self, content: str, section_name: str) -> List[str]:
        """
        从客服文档内容中提取指定字段下的条目列表。

        客服文档结构示例:
        【标准答复】
        1. 先确认商品状态
        2. 然后联系客服

        Args:
            content: 文档完整内容
            section_name: 字段名（如"标准答复"、"需核实信息"）

        Returns:
            清理后的条目列表（去除了编号前缀）
        """
        pattern = rf"【{section_name}】\n(.*?)(?:\n【|$)"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return []
        block = match.group(1).strip()
        items: List[str] = []
        for line in block.splitlines():
            # 移除常见的编号前缀（如"1. "、"- "）
            cleaned = re.sub(r"^\s*(?:\d+\.\s*|-\s*)", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    def _deduplicate_items(self, items: List[str]) -> List[str]:
        """
        对字符串列表去重，同时保持原始顺序。

        Args:
            items: 待去重的字符串列表

        Returns:
            去重后的列表（首次出现的项被保留）
        """
        seen = set()
        deduped: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _summarize_manual_snippet(self, content: str) -> str:
        """
        精简手册片段文本，移除图片ID标记并截断过长文本。

        处理步骤:
        1. 移除所有[PIC_ID]图片标记
        2. 合并多余空白字符
        3. 若长度<=180字符直接返回
        4. 否则截断到177字符并去掉尾部标点，加上省略号

        Args:
            content: 原始手册片段文本

        Returns:
            精简后的文本（最长180字符）
        """
        compact = re.sub(r'\[([^\]]+)\]', '', content)
        compact = re.sub(r'\s+', ' ', compact).strip()
        if len(compact) <= 180:
            return compact
        return compact[:177].rstrip("，。；; ") + "..."

    def _select_best_manual_source(self, question: str, manual_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从多个手册检索结果中选择最佳来源。

        选择策略（加权评分）:
        - relevance_score(0.55): 检索相关性分数，权重最高
        - 章节标题相似度(0.25): 标题与问题越相关越优先
        - 手册名相似度(0.20): 产品名与问题越相关越优先

        Args:
            question: 用户问题
            manual_sources: 手册检索结果列表

        Returns:
            得分最高的手册文档
        """
        scored_sources = []
        for source in manual_sources:
            metadata = source.get("metadata", {})
            section_title = str(metadata.get("section_title", "")).strip()
            manual_name = str(metadata.get("manual_name", "")).replace("手册", "")
            score = (
                float(source.get("relevance_score", 0.0)) * 0.55
                + self._simple_similarity(question, section_title) * 0.25
                + self._simple_similarity(question, manual_name) * 0.2
            )
            scored_sources.append((source, score))
        scored_sources.sort(key=lambda item: item[1], reverse=True)
        return scored_sources[0][0]

    def _extract_structured_manual_points(self, question: str, manual_sources: List[Dict[str, Any]]) -> List[str]:
        request_mode = self._analyze_manual_request(question)
        if not request_mode["is_structured"]:
            return []

        candidate_sources = self._prepare_structured_manual_sources(question, manual_sources)
        ranked_sources = sorted(
            candidate_sources,
            key=lambda source: (
                self._simple_similarity(question, str(source.get("metadata", {}).get("section_title", ""))) * 0.45
                + self._simple_similarity(question, source.get("content", "")) * 0.2
                + float(source.get("relevance_score", 0.0)) * 0.35
            ),
            reverse=True,
        )

        collected: List[str] = []
        for source in ranked_sources[:4]:
            source_points = self._extract_points_from_source(question, source, request_mode)
            for point in source_points:
                if point not in collected:
                    collected.append(point)
            if request_mode["limit"] and len(collected) >= request_mode["limit"]:
                break

        limit = request_mode["limit"] or 4
        min_required = min(limit, 2 if limit <= 3 else 3)
        if len(collected) < min_required:
            return []
        return collected[:limit]

    def _prepare_structured_manual_sources(
        self,
        question: str,
        manual_sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged_sources: Dict[str, Dict[str, Any]] = {
            source.get("doc_id", f"manual_{index}"): source
            for index, source in enumerate(manual_sources)
        }
        if self.dual_route_retriever is None:
            return list(merged_sources.values())

        try:
            supplemental_sources = self.dual_route_retriever._retrieve_manual(
                question,
                top_k=max(8, settings.route_manual_top_k),
            )
        except Exception:
            supplemental_sources = []

        for source in supplemental_sources:
            doc_id = source.get("doc_id", f"manual_extra_{len(merged_sources)}")
            merged_sources.setdefault(doc_id, source)
        return list(merged_sources.values())

    def _extract_points_from_source(
        self,
        question: str,
        source: Dict[str, Any],
        request_mode: Dict[str, Any],
    ) -> List[str]:
        units = self._build_manual_units(source)
        if not units:
            return []

        scored_units = []
        for index, unit in enumerate(units):
            score = self._score_manual_unit(question, unit, source, request_mode)
            scored_units.append((unit, score, index))

        threshold = 0.18 if request_mode["is_structured"] else 0.24
        filtered = [item for item in scored_units if item[1] >= threshold]
        if not filtered:
            return []

        if request_mode["preserve_order"]:
            filtered.sort(key=lambda item: item[2])
        else:
            filtered.sort(key=lambda item: item[1], reverse=True)

        cleaned_points: List[str] = []
        for unit, _score, _index in filtered:
            text = unit["text"].strip()
            if text and text not in cleaned_points:
                cleaned_points.append(text)
        return cleaned_points[: request_mode["limit"] or 4]

    def _build_manual_units(self, source: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        将手册文档拆分为独立语义单元。

        拆分策略:
        1. 清理<PIC>标记中的图片ID
        2. 按行分割，遇到标题行（#开头）更新当前章节
        3. 将符号列表（●、•、·）和数字编号拆分到独立行
        4. 对超长段落（>=48字符且无结构标记）按句子边界进一步拆分
        5. 过滤无效单元

        Returns:
            手册单元列表，每个单元包含文本和所属章节标题
        """
        metadata = source.get("metadata", {})
        section_title = str(metadata.get("section_title", "")).strip()
        content = re.sub(r'\[([^\]]+)\]', '', source.get("content", ""))
        units: List[Dict[str, str]] = []

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                line = line.lstrip("#").strip()
                if line:
                    section_title = line
                continue

            # 将符号列表（●、•、·）拆分成独立行
            normalized = line.replace("●", "\n● ").replace("•", "\n• ").replace("·", "\n· ")
            # 将数字编号（如"1."、"1、"）前加换行，使每个编号项独立
            normalized = re.sub(r'(?<!\d)(\d+[\.、])', r'\n\1', normalized)

            for part in normalized.splitlines():
                cleaned = re.sub(r'^\s*(?:[-*]|[•●·]|\d+[\.、])\s*', '', part).strip()
                if not cleaned:
                    continue
                # 对超长段落（>=48字符）且无结构关键词的，按句子边界拆分
                if len(cleaned) > 48 and not re.search(r'(?:包括|包含|部件|步骤|注意|警告|请勿)', cleaned):
                    segments = [seg.strip() for seg in re.split(r'[；;。]', cleaned) if seg.strip()]
                else:
                    segments = [cleaned]
                for segment in segments:
                    if self._is_valid_manual_unit(segment):
                        units.append({"text": segment, "section_title": section_title})

        return units

    def _is_valid_manual_unit(self, text: str) -> bool:
        """
        检测是否为有效的手册单元。

        有效单元必须满足:
        1. 非空且非<PIC>占位符
        2. 长度>=4字符
        3. 包含中文或英文字符（非纯符号）
        4. 非纯大写字母/数字序列（如产品型号DCB107是型号而非说明）
        5. 非大写字母+数字为主的混合文本

        用于过滤手册切分时产生的噪音片段。

        Returns:
            是否为有效的手册单元
        """
        cleaned = text.strip()
        if not cleaned or cleaned == "<PIC>":
            return False
        if len(cleaned) < 4:
            return False
        if not re.search(r'[\u4e00-\u9fffA-Za-z]', cleaned):
            return False
        if re.fullmatch(r'[A-Z0-9_,.\- ]{3,}', cleaned):
            return False
        if len(re.findall(r'[A-Z0-9]', cleaned)) >= max(4, len(cleaned) // 2) and not re.search(r'[\u4e00-\u9fff]{2,}', cleaned):
            return False
        return True

    def _score_manual_unit(
        self,
        question: str,
        unit: Dict[str, str],
        source: Dict[str, Any],
        request_mode: Dict[str, Any],
    ) -> float:
        text = unit["text"]
        section_title = unit.get("section_title", "")
        score = (
            self._simple_similarity(question, text) * 0.55
            + self._simple_similarity(question, section_title) * 0.18
            + float(source.get("relevance_score", 0.0)) * 0.22
        )

        if request_mode["want_precautions"] and re.search(r'(注意|警告|危险|请勿|不要|必须|严禁)', text):
            score += 0.3
        if request_mode["want_steps"] and re.search(r'(步骤|先|再|然后|最后|按|安装|开启|关闭|操作)', text):
            score += 0.24
        if request_mode["want_components"] and re.search(r'(包括|包含|部件|组件|包装盒内含|由.*组成|配有)', text):
            score += 0.28
        if request_mode["want_list"] and re.search(r'(包括|包含|步骤|注意|警告|请勿|可|应|需要)', text):
            score += 0.12
        return score

    def _analyze_manual_request(self, question: str) -> Dict[str, Any]:
        limit = self._extract_requested_limit(question)
        want_steps = any(token in question for token in ("步骤", "如何", "怎么", "操作", "安装", "开启", "关闭"))
        want_components = any(token in question for token in ("部件", "组成", "构成", "包含哪些", "有哪些"))
        want_precautions = any(token in question for token in ("注意", "预防", "警告", "安全", "注意事项"))
        want_list = bool(limit) or want_steps or want_components or want_precautions or any(
            token in question for token in ("列出", "写出", "哪些", "哪几", "前", "后")
        )
        return {
            "limit": limit,
            "want_steps": want_steps,
            "want_components": want_components,
            "want_precautions": want_precautions,
            "want_list": want_list,
            "preserve_order": bool(limit),
            "is_structured": want_list,
        }

    def _extract_requested_limit(self, question: str) -> Optional[int]:
        """
        从问题文本中提取用户指定的条目数量限制。

        支持的提取模式:
        - "前N条" / "前N个" -> 提取数字 N
        - "列出N条" -> 提取数字 N

        例如: "请告诉我前五条注意事项" -> 返回 5

        Args:
            question: 用户问题文本

        Returns:
            限制数量（整数），若未找到则返回 None
        """
        match = re.search(r'前([一二两三四五六七八九十\d]+)条', question)
        if not match:
            match = re.search(r'前([一二两三四五六七八九十\d]+)个', question)
        if not match:
            match = re.search(r'列出([一二两三四五六七八九十\d]+)条', question)
        if not match:
            return None
        return self._parse_cn_number(match.group(1))

    def _parse_cn_number(self, token: str) -> Optional[int]:
        """
        将中文数字 token 转换为整数。

        支持的中文数字: 一/二/两/三/四/五/六/七/八/九/十，
        以及组合如"十一"、"二十三"等。

        Args:
            token: 中文数字字符串

        Returns:
            对应的整数，若无法解析则返回 None
        """
        token = token.strip()
        if not token:
            return None
        if token.isdigit():
            return int(token)

        mapping = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        if token == "十":
            return 10
        if len(token) == 1:
            return mapping.get(token)
        if token.startswith("十"):
            return 10 + mapping.get(token[1:], 0)
        if token.endswith("十"):
            return mapping.get(token[0], 0) * 10
        if "十" in token:
            left, right = token.split("十", 1)
            return mapping.get(left, 0) * 10 + mapping.get(right, 0)
        return mapping.get(token)

    def _format_manual_visual_hint(self, source: Dict[str, Any]) -> str:
        """
        生成手册文档的视觉提示文本。

        当检索结果包含图片引用时，在回答末尾添加图示提示，
        告知用户可参考哪些配图。

        Args:
            source: 手册检索结果条目

        Returns:
            图示提示文本，如 "相关图示：<PIC> <PIC>"，无图时返回空字符串
        """
        image_ids = source.get("image_ids", [])
        if not image_ids:
            return ""
        marker_count = min(len(image_ids), 2)
        return "相关图示：" + " ".join(["<PIC>"] * marker_count)

    def _simple_similarity(self, left: str, right: str) -> float:
        """
        查询词项覆盖率相似度。

        算法: |left_terms ∩ right_terms| / |left_terms|
        仅用Coverage而非Jaccard，因为客服场景更关注查询词是否被覆盖。

        Returns:
            [0,1]范围内的覆盖率分数
        """
        left_terms = self._extract_terms(left)
        right_terms = self._extract_terms(right)
        if not left_terms or not right_terms:
            return 0.0
        inter = len(left_terms & right_terms)
        return inter / max(1, len(left_terms))

    def _direct_keyword_hits(self, left: str, right: str) -> int:
        """
        计算left中的词项在right中直接出现的数量。

        统计条件: 词项长度>=2，避免单字符停用词干扰。

        Returns:
            命中的关键词数量
        """
        hits = 0
        for token in self._extract_terms(left):
            if len(token) >= 2 and token in right:
                hits += 1
        return hits

    def _extract_terms(self, text: str) -> set[str]:
        """
        将文本转换为词项集合。

        词项提取策略:
        - 英文/数字/下划线/连字符: 整体作为词项
        - 单字符中文: 直接作为词项
        - 连续中文(>=2字符): 提取2-gram词项

        Returns:
            词项集合
        """
        text = (text or "").strip().lower()
        if not text:
            return set()

        terms = set(re.findall(r"[a-z0-9_-]+", text))
        for segment in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(segment) == 1:
                terms.add(segment)
                continue
            for idx in range(len(segment) - 1):
                terms.add(segment[idx: idx + 2])
        return terms

    def generate_with_sources(
        self,
        query: str,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        生成带来源标注的回答 - 调试/分析用

        在回复末尾附加参考来源信息，便于验证答案准确性

        Args:
            query: 用户问题
            images: 图片列表

        Returns:
            包含完整来源信息的回答结果
        """
        result = self.generate(query, images)

        # 在回复末尾添加来源引用
        if result["sources"]:
            source_text = "\n\n【参考来源】\n"
            for i, source in enumerate(result["sources"], 1):
                # 截取前200字符避免过长
                source_text += f"{i}. {source['content'][:200]}...\n"
                if source.get("image_ids"):
                    source_text += f"   相关图片: {', '.join(source['image_ids'])}\n"

            result["response"] += source_text

        return result


# 全局单例实例 - 延迟初始化确保配置已加载
_response_generator: Optional[ResponseGenerator] = None


def get_response_generator() -> ResponseGenerator:
    """
    获取回答生成器单例实例

    使用全局单例模式避免重复初始化，每个请求共享同一实例

    Returns:
        ResponseGenerator实例
    """
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator()
    return _response_generator
