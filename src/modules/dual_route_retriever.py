"""
双路检索器

职责：
1. 根据构建的客服路由知识库识别 service / manual / mixed
2. 对客服服务类问题走轻量客服知识检索
3. 对产品使用类问题走手册 RAG 检索

设计目标：
- 不依赖额外的大模型分类，降低请求时延
- 路由和客服检索全部使用轻量规则 + 词项相似度
- 产品知识继续复用现有手册向量索引
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger 

from config import settings
from .rag_engine import RAGEngine, get_rag_engine
from .rag_engine import Document
from .route_classifier import RouteClassifier, get_route_classifier
from .route_classifier import SERVICE_HINTS, MANUAL_HINTS, MIXED_HINTS
from src.utils.text_utils import QueryProcessor
from src.utils.llm_corrector import get_spell_corrector


MANUAL_ALIAS_SEEDS: Dict[str, List[str]] = {
    "VR头显手册": ["VR头显", "头显", "VR设备", "vr"],
    "人体工学椅手册": ["人体工学椅", "工学椅", "办公椅", "椅子"],
    "健身单车手册": ["健身单车", "动感单车", "单车"],
    "健身追踪器手册": ["健身追踪器", "手环", "表带", "追踪器"],
    "儿童电动摩托车手册": ["儿童电动摩托车", "儿童摩托车", "电动摩托车"],
    "冰箱手册": ["冰箱", "冷藏室", "冷冻室"],
    "功能键盘手册": ["功能键盘", "键盘", "硬件模式"],
    "发电机手册": ["发电机", "机油", "发动机", "电池电量"],
    "可编程温控器手册": ["温控器", "可编程温控器", "恒温器"],
    "吹风机手册": ["吹风机", "冷机", "热机", "化油器"],
    "摩托艇手册": [
        "摩托艇", "划船", "钓鱼", "拖曳速度", "滑航",
        "boat", "ship", "bimini top", "anchor light", "jet wash",
        "bilge pump", "cooling system", "engine oil", "water supply button",
        "battery compartment", "sound system", "emission control certificate",
    ],
    "水泵手册": ["水泵", "泵"],
    "洗碗机手册": ["洗碗机", "亮碟剂", "餐具篮"],
    "烤箱手册": ["烤箱", "烘烤", "air fryer", "airfryer", "空气炸锅"],
    "电钻手册": ["电钻", "指示灯", "DCB107", "DCB112"],
    "相机手册": ["相机", "镜头", "快门", "闪光灯"],
    "空气净化器手册": ["空气净化器", "空气质量指示灯", "净化器"],
    "空调手册": ["空调", "遥控器", "自清洁", "等离子", "自动运行模式"],
    "蒸汽清洁机手册": ["蒸汽清洁机", "蒸汽拖把", "清洁机"],
    "蓝牙激光鼠标手册": ["蓝牙激光鼠标", "蓝牙鼠标", "鼠标"],
}

# SERVICE_ONLY_HINTS 收敛至 route_classifier.py，从那里读取
# 此处仅保留向后兼容别名
SERVICE_ONLY_HINTS = SERVICE_HINTS

# 型号/代码词检测正则：命中时强烈暗示说明书类问题，建议启用 hybrid 检索
# 规则：纯大写字母开头 + 数字结尾，或特定格式的型号串
CODE_WORD_PATTERNS = [
    re.compile(r'\b[A-Z]{2,}\d+\b'),           # DCB107, AP123, XRS800
    re.compile(r'\b[A-Z]{2,}[-_]\d+\b'),       # ABC-123, DEF_456
    re.compile(r'\b\d+[A-Z]{2,}\b'),           # 123ABC, 45DEF
    re.compile(r'\b[A-Z][a-z]*\d+[A-Z]*\b'),  # XRSmodel1, BatteryX1
]


class DualRouteRetriever:
    """双路检索编排器。"""

    def __init__(self):
        self.rag_engine: Optional[RAGEngine] = None
        self.route_examples: List[Dict[str, Any]] = []
        self.service_documents: List[Dict[str, Any]] = []
        self.intent_specs: Dict[str, Dict[str, Any]] = {}
        self.service_keywords: List[str] = []
        self.manual_keywords: List[str] = []
        self.manual_doc_map: Dict[str, List[Document]] = {}
        self.manual_alias_map: Dict[str, List[str]] = {}
        self.manual_doc_index_map: Dict[str, int] = {}
        self.manual_doc_vector_cache: Dict[str, np.ndarray] = {}
        self.route_classifier: Optional[RouteClassifier] = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        self.rag_engine = get_rag_engine()
        self.rag_engine.initialize()
        self.route_classifier = get_route_classifier()
        self.route_classifier.initialize()
        self._load_route_kb()
        self._initialized = True
        logger.info("双路检索器初始化完成")

    def _load_route_kb(self):
        route_kb_file = settings.route_kb_path / settings.service_route_kb_file
        if not route_kb_file.exists():
            raise FileNotFoundError(
                f"未找到客服路由知识库: {route_kb_file}，请先运行 scripts.build_dual_route_kb"
            )

        payload = json.loads(route_kb_file.read_text(encoding="utf-8"))
        self.route_examples = payload.get("route_examples", [])
        self.service_documents = payload.get("service_documents", [])
        self.intent_specs = payload.get("intents", {})
        self.service_keywords = self._collect_service_keywords(self.intent_specs)
        self.manual_keywords = self._collect_manual_keywords()
        self._build_manual_doc_map()

    def _collect_service_keywords(self, intent_specs: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        从客服意图规格中收集所有关键词，用于规则路由评分。

        所有意图的关键词合并后去重，按长度降序排列，
        使得 _lexical_similarity 匹配时优先匹配更长、更具体的词。

        Args:
            intent_specs: 意图规格字典，格式为 {intent_name: {keywords: [...], ...}}

        Returns:
            去重后的关键词列表（按长度降序）
        """
        keywords = []
        for spec in intent_specs.values():
            keywords.extend(spec.get("keywords", []))
        return sorted(set(keywords), key=len, reverse=True)

    def _collect_manual_keywords(self) -> List[str]:
        """
        从知识库中收集手册产品名关键词，用于规则路由评分。

        收集策略:
        1. 仅收集 route="manual" 的文档所属的手册名
        2. 对手册名做预处理：去除"手册"后缀、提取独立词项
        3. 按长度降序排列，使匹配优先匹配完整产品名

        Returns:
            手册产品名关键词列表（按长度降序）
        """
        if not self.rag_engine:
            return []

        manual_terms = set()
        for doc in self.rag_engine.knowledge_base.text_documents:
            manual_name = str(doc.metadata.get("manual_name", "")).strip()
            route = str(doc.metadata.get("route", "manual")).strip()
            if route != "manual" or not manual_name:
                continue
            manual_terms.add(manual_name)

        # 从手册名中提取更容易命中的产品词
        terms = set()
        for name in manual_terms:
            clean = name.replace("手册", "").strip()
            if clean:
                terms.add(clean)
            for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_-]+", clean):
                if token:
                    terms.add(token)

        return sorted(terms, key=len, reverse=True)

    def _build_manual_doc_map(self) -> None:
        """
        构建手册别名映射表和文档分组索引。

        别名映射是手册路由检索的核心基础设施：
        1. 将所有文档片段按手册名称分组
        2. 维护产品名称别名集合，用于从用户问题中识别手册归属
        3. 别名来源：预设种子别名 + 去"手册"后的产品名 + 产品名分词单元

        别名按长度降序排列，确保优先匹配更长的词汇。
        """
        if not self.rag_engine:
            return

        manual_doc_map: Dict[str, List[Document]] = defaultdict(list)
        alias_map: Dict[str, List[str]] = {}
        doc_index_map: Dict[str, int] = {}
        for doc_idx, doc in enumerate(self.rag_engine.knowledge_base.text_documents):
            manual_name = str(doc.metadata.get("manual_name", "")).strip()
            route = str(doc.metadata.get("route", "manual")).strip()
            if route != "manual" or not manual_name:
                continue
            manual_doc_map[manual_name].append(doc)
            doc_index_map[doc.doc_id] = doc_idx

        for manual_name in manual_doc_map:
            aliases = set(MANUAL_ALIAS_SEEDS.get(manual_name, []))
            clean = manual_name.replace("手册", "").strip()
            if clean:
                aliases.add(clean)
            for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_-]+", clean):
                if token:
                    aliases.add(token)
            alias_map[manual_name] = sorted(aliases, key=len, reverse=True)

        self.manual_doc_map = dict(manual_doc_map)
        self.manual_alias_map = alias_map
        self.manual_doc_index_map = doc_index_map

    def route_query(
        self,
        query: str,
        images: Optional[List[str]] = None,
        normalized_query: Optional[str] = None,
        image_tags: Optional[List[str]] = None,
        product_candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            query: 原始查询文本
            images: Base64 图片列表
            normalized_query: 已归一化的查询文本（可选，传入则直接使用，不重复 normalize）
            image_tags: 视觉标签列表
            product_candidates: 候选产品名列表
        """
        if not self._initialized:
            self.initialize()

        if normalized_query:
            search_query = normalized_query
        else:
            normalized = QueryProcessor.normalize_query_for_retrieval(query)
            search_query = normalized["normalized_query"]

        # 拼写纠错：纠正错别字后再进入规则匹配和检索，保护型号词
        if settings.spell_correction_enabled:
            original = search_query
            search_query = get_spell_corrector().correct(search_query)
            if search_query != original:
                logger.debug(f"拼写纠错: '{original}' -> '{search_query}'")

        # 合并外部传入的候选手册（来自多模态理解器）
        if product_candidates:
            detected = self._detect_manual_candidates(search_query)
            merged = self._merge_external_manual_candidates(detected, product_candidates)
        else:
            merged = None

        rule_info = self._compute_rule_route_info(
            search_query=search_query,
            images=images,
            external_manual_candidates=merged,
        )
        classifier_result = (
            self.route_classifier.predict(
                search_query,
                images=images,
                image_tags=image_tags,
                product_candidates=product_candidates,
                normalized_query=search_query,
            )
            if self.route_classifier is not None
            else self._empty_classifier_result()
        )
        final_route, classifier_used, fallback_reason = self._arbitrate_route(
            rule_info=rule_info,
            classifier_result=classifier_result,
        )

        return {
            **rule_info,
            "route": final_route,
            "classifier_label": classifier_result.get("label"),
            "classifier_confidence": classifier_result.get("confidence", 0.0),
            "classifier_margin": classifier_result.get("margin", 0.0),
            "classifier_probs": classifier_result.get("probs", {}),
            "classifier_used": classifier_used,
            "classifier_fallback_reason": fallback_reason,
            "classifier_available": classifier_result.get("available", False),
            "classifier_input_text": classifier_result.get("input_text", search_query),
            "classifier_debug_info": classifier_result.get("debug_info"),
            "language": normalized["language"] if normalized_query is None else "zh",
            "normalized_query": search_query,
            "translation_applied": normalized["translation_applied"] if normalized_query is None else False,
        }

    def _compute_rule_route_info(
        self,
        search_query: str,
        images: Optional[List[str]] = None,
        external_manual_candidates: Optional[List[Tuple[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        基于规则计算路由路由信息（不调用LLM）。

        计算流程:
        1. 意图匹配: 从客服意图规格中匹配query命中了哪些意图
        2. 手册候选: 用别名映射检测query是否明确指向某本手册
        3. 关键词命中: 统计service/manual关键词命中数量
        4. 示例相似度: 计算与知识库示例问题的词项相似度
        5. 加权评分: service_score和manual_score各维度加权求和
        6. 路由决策: 根据分数阈值和条件组合确定路由类型

        分数调节机制:
        - service_only_hits: 仅客服意图词时加service分，降manual分
        - 有service意图但无manual线索: 强降manual分
        - 有图+manual关键词: 额外加manual分（图片可能有助于手册理解）

        Returns:
            包含rule_route, strong_rule_route和各维度得分的字典
        """
        # 候选手册：优先使用外部传入（来自多模态理解器），兜底再做本地检测
        if external_manual_candidates is not None:
            candidate_manuals = external_manual_candidates
        else:
            candidate_manuals = self._detect_manual_candidates(search_query)

        matched_intents = self._match_service_intents(search_query)
        service_keyword_hits = [kw for kw in self.service_keywords if kw in search_query]
        manual_keyword_hits = [kw for kw in self.manual_keywords if kw in search_query]
        service_only_hits = [kw for kw in SERVICE_ONLY_HINTS if kw in search_query]
        has_manual_instruction_intent = "manual_instruction" in matched_intents
        has_service_policy_intent = any(intent != "manual_instruction" for intent in matched_intents)

        service_example_score = self._best_example_score(search_query, target_route="service")
        manual_example_score = self._best_example_score(search_query, target_route="manual")

        service_keyword_score = min(1.0, len(service_keyword_hits) / 4.0)
        manual_keyword_score = min(1.0, len(manual_keyword_hits) / 3.0)

        service_score = (
            service_keyword_score * settings.route_service_keyword_weight
            + service_example_score * settings.route_example_similarity_weight
        )
        manual_score = (
            manual_keyword_score * 0.4
            + manual_example_score * 0.6
        )

        if service_only_hits:
            service_score += min(0.22, 0.08 * len(service_only_hits))

        if has_service_policy_intent and not candidate_manuals and not manual_keyword_hits:
            service_score += 0.12
            manual_score *= 0.82

        if service_only_hits and not candidate_manuals and not manual_keyword_hits:
            manual_score *= 0.72

        if images and manual_keyword_hits:
            manual_score += 0.1

        strong_rule_route = ""
        if self._should_force_service_route(
            matched_intents=matched_intents,
            candidate_manuals=candidate_manuals,
            manual_keyword_hits=manual_keyword_hits,
            service_only_hits=service_only_hits,
        ):
            route = "service"
            strong_rule_route = "service"
        elif has_service_policy_intent and (manual_keyword_hits or bool(images) or bool(candidate_manuals)):
            route = "mixed"
            strong_rule_route = "mixed"
        elif has_manual_instruction_intent and (has_service_policy_intent or manual_keyword_hits or bool(images) or bool(candidate_manuals)):
            route = "mixed"
            strong_rule_route = "mixed"
        elif service_score >= settings.route_service_threshold and manual_score >= settings.route_manual_threshold:
            route = "mixed"
        elif service_score >= settings.route_service_threshold:
            route = "service"
        else:
            route = "manual"

        if (
            route != "mixed"
            and abs(service_score - manual_score) <= settings.route_mixed_gap_threshold
            and service_score >= settings.route_service_threshold * 0.7
            and manual_score >= settings.route_manual_threshold
        ):
            route = "mixed"

        return {
            "rule_route": route,
            "strong_rule_route": strong_rule_route,
            "service_score": round(service_score, 4),
            "manual_score": round(manual_score, 4),
            "matched_intents": matched_intents[:6],
            "has_manual_instruction_intent": has_manual_instruction_intent,
            "has_service_policy_intent": has_service_policy_intent,
            "service_keyword_hits": service_keyword_hits[:8],
            "manual_keyword_hits": manual_keyword_hits[:8],
            "service_only_hits": service_only_hits[:8],
            "manual_candidates": [manual_name for manual_name, _score in candidate_manuals[:3]],
        }

    def _arbitrate_route(
        self,
        rule_info: Dict[str, Any],
        classifier_result: Dict[str, Any],
    ) -> Tuple[str, bool, str]:
        """
        规则路由与分类器路由的最终裁决。

        裁决策略（共 9 条，按优先级顺序）:
        1. 强规则路由(strong_rule): 由强制路由条件直接决定，不使用分类器
        2. 高置信分类器(>=high_threshold)直接采纳（分类器不可用时回退规则路由）
        3. 低置信分类器(<low_threshold)回退规则（分类器不可用时回退规则路由）
        4. 分类器与规则一致且置信度>=0.50时采纳分类器
        5. 单侧 mixed：仅当其中一方为 mixed、另一方明确时升为 mixed
        6. 规则强偏好 service(有客服意图且无手册信号)→ service
        7. 规则强偏好 manual(有手册信号且无客服强意图)→ manual
        8. 分类器与规则不一致(双方均非 mixed 且无强规则偏好)→ 采纳规则

        Args:
            rule_info: 规则路由计算结果
            classifier_result: ONNX分类器预测结果

        Returns:
            (最终路由, 是否使用分类器, 回退原因)
        """
        rule_route = rule_info["rule_route"]
        strong_rule_route = rule_info.get("strong_rule_route", "")

        # 优先级1: 强规则强制路由
        if strong_rule_route:
            return strong_rule_route, False, f"strong_rule:{strong_rule_route}"

        # 优先级2: 分类器不可用时回退规则路由
        if not classifier_result.get("available", False):
            return rule_route, False, classifier_result.get("fallback_reason", "classifier_unavailable")

        classifier_label = classifier_result.get("label")
        classifier_confidence = float(classifier_result.get("confidence", 0.0))
        classifier_margin = float(classifier_result.get("margin", 0.0))

        # 优先级3: 高置信（绝对值 >= high_threshold）或高 margin（top1-top2 差距大）
        # margin 信号在分布模糊时比绝对 confidence 更可靠
        if (classifier_confidence >= settings.route_classifier_high_threshold
                or classifier_margin >= settings.route_classifier_high_margin):
            return classifier_label, True, "classifier_high_confidence"

        # 优先级4: 低置信 且 低 margin → 分类器过于模糊，回退规则
        if (classifier_confidence < settings.route_classifier_low_threshold
                or classifier_margin < settings.route_classifier_low_margin):
            return rule_route, False, "classifier_low_confidence"

        # 优先级5: 分类器与规则一致且置信度和 margin 都足够
        if classifier_label == rule_route and classifier_confidence >= 0.50 and classifier_margin >= settings.route_classifier_low_margin:
            return classifier_label, True, "classifier_matches_rule"

        # 优先级6: 单侧 mixed——仅当其中一方为 mixed 而另一方已明确时升为 mixed
        # 若双方同时为 mixed 则视为"一致"，走优先级 5（已在上方处理）
        # 若分类器为 mixed 而规则已明确，则升 mixed；若反之，优先信任规则的明确方向
        if classifier_label == "mixed" and rule_route != "mixed":
            return "mixed", True, "classifier_mixed_promoted"
        if rule_route == "mixed" and classifier_label != "mixed":
            # 分类器有明确 label，优先采纳分类器方向
            return classifier_label, True, "rule_mixed_classifier_certain"

        # 优先级7: 规则强偏好 service(有客服意图 且 无手册信号)
        has_service = rule_info.get("has_service_policy_intent")
        has_manual_signal = rule_info.get("manual_candidates") or rule_info.get("manual_keyword_hits")
        if has_service and not has_manual_signal:
            return "service", False, "rule_prefers_service_policy"

        # 优先级8: 规则强偏好 manual(有手册信号 且 无客服强意图)
        # 与优先级7互斥，排除 service 强意图场景，避免"退货说明"这类词被误路由到 manual
        if has_manual_signal and not has_service:
            return "manual", False, "rule_prefers_manual_candidate"

        # 优先级9: 分类器与规则不一致，双方均非 mixed 且无强规则偏好 → 采纳规则
        # 分类器模糊时信任规则，规则也比随机猜测强
        return rule_route, False, "rule_overrides_classifier"

    def _empty_classifier_result(self) -> Dict[str, Any]:
        return {
            "available": False,
            "label": None,
            "confidence": 0.0,
            "margin": 0.0,
            "probs": {},
            "input_text": "",
            "fallback_reason": "classifier_not_initialized",
            "debug_info": {},
        }

    def retrieve(
        self,
        query: str,
        images: Optional[List[str]] = None,
        normalized_query: Optional[str] = None,
        image_tags: Optional[List[str]] = None,
        product_candidates: Optional[List[str]] = None,
        use_rerank: bool = True,
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        route_info = self.route_query(
            query,
            images=images,
            normalized_query=normalized_query,
            image_tags=image_tags,
            product_candidates=product_candidates,
        )
        route = route_info["route"]
        search_query = route_info["normalized_query"]

        service_results: List[Dict[str, Any]] = []
        manual_results: List[Dict[str, Any]] = []

        if route in ("service", "mixed"):
            service_results = self._retrieve_service(search_query, top_k=settings.route_service_top_k)

        if route in ("manual", "mixed"):
            # 从 route_info 取已合并的候选手册（来自外部 product_candidates + 内部别名检测）
            manual_candidates = route_info.get("manual_candidates", [])
            manual_results = self._retrieve_manual(
                search_query,
                top_k=settings.route_manual_top_k,
                use_rerank=use_rerank,
                candidate_manuals=manual_candidates,
            )

        merged_results = self._merge_results(route, service_results, manual_results)
        return {
            "route_info": route_info,
            "service_results": service_results,
            "manual_results": manual_results,
            "results": merged_results,
        }

    def _retrieve_manual(
        self,
        query: str,
        top_k: int,
        use_rerank: bool = False,
        candidate_manuals: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        手册检索主入口。

        Args:
            query: 检索 query
            top_k: 返回上限
            use_rerank: 是否使用 rerank
            candidate_manuals: 候选手册名列表（来自 route_info，来自 product_candidates +
                                别名检测的合并结果）。若传入则优先使用，不再重复检测。
        """
        assert self.rag_engine is not None

        # 优先使用外部传入的候选手册；若未传入则退回到本地检测
        if candidate_manuals:
            # candidate_manuals 是手册名列表，需要转换为 (name, score) 格式
            # 外部传入的候选手册来源于 product_candidates 映射，直接给默认分 1.0
            detected = [(name, 1.0) for name in candidate_manuals[: settings.route_manual_candidate_top_k]]
        else:
            detected = self._detect_manual_candidates(query)

        if detected:
            candidate_docs: List[Document] = []
            for manual_name, _score in detected[: settings.route_manual_candidate_top_k]:
                candidate_docs.extend(self.manual_doc_map.get(manual_name, []))

            broad_by_doc_id: Dict[str, Dict[str, Any]] = {}
            if self._should_use_local_manual_recall(detected):
                broad_by_doc_id = self._local_semantic_recall(
                    query=query,
                    candidate_docs=candidate_docs,
                    top_k=max(top_k, settings.route_manual_broad_top_k),
                )

                # 对局部召回结果加一次 cross-encoder rerank
                if broad_by_doc_id and self.rag_engine and hasattr(self.rag_engine, "reranker"):
                    # 将 dict 列表转换为 List[Tuple[Document, float]] 供 reranker 使用
                    from .rag_engine import Document
                    doc_tuples: List[Tuple[Document, float]] = []
                    for doc_id, item in broad_by_doc_id.items():
                        # 尝试从索引映射恢复 Document 对象
                        kb_doc_idx = self.manual_doc_index_map.get(doc_id)
                        if kb_doc_idx is not None and self.rag_engine.knowledge_base.text_documents:
                            doc_obj = self.rag_engine.knowledge_base.text_documents[kb_doc_idx]
                            doc_tuples.append((doc_obj, float(item.get("relevance_score", 0.0))))

                    if doc_tuples:
                        reranked_tuples = self.rag_engine.reranker.rerank(query, doc_tuples, top_k)
                        broad_by_doc_id = {item[0].doc_id: {
                            "content": item[0].content,
                            "doc_id": item[0].doc_id,
                            "relevance_score": score,
                            "metadata": dict(item[0].metadata),
                            "has_image": "<PIC>" in item[0].content,
                            "image_ids": re.findall(r'\[([^\]]+)\]', item[0].content),
                        } for item, score in reranked_tuples}
                        logger.debug(f"局部召回 rerank: {len(broad_by_doc_id)} 个文档已重排")

            if not broad_by_doc_id:
                # 型号/代码词存在时启用 hybrid，改善精确匹配的召回效果
                use_hybrid = self._contains_code_word(query)
                broad_results = self.rag_engine.retrieve(
                    query,
                    top_k=max(top_k, settings.route_manual_broad_top_k),
                    use_rerank=use_rerank,
                    use_hybrid=use_hybrid,
                )
                broad_by_doc_id = {item["doc_id"]: item for item in broad_results}

            reranked = self._rerank_manual_docs(
                query=query,
                candidate_docs=candidate_docs,
                broad_by_doc_id=broad_by_doc_id,
                candidate_manuals=detected,
                top_k=top_k,
            )
        else:
            use_hybrid = self._contains_code_word(query)
            broad_results = self.rag_engine.retrieve(
                query,
                top_k=max(top_k, settings.route_manual_broad_top_k),
                use_rerank=use_rerank,
                use_hybrid=use_hybrid,
            )
            reranked = self._rerank_manual_results(query, broad_results, top_k=top_k)

        for item in reranked:
            item.setdefault("metadata", {})
            item["metadata"]["route"] = item["metadata"].get("route", "manual")
        return reranked

    def _detect_manual_candidates(self, query: str) -> List[Tuple[str, float]]:
        """
        检测query明确指向哪些手册。

        使用别名映射检测：若query文本中包含某手册的别名
        （如"人体工学椅"或"工学椅"），则将该手册加入候选。
        别名命中得分 = sum(min(len(alias), 8) for alias in hits)，
        长别名命中权重更高。

        Returns:
            按得分降序排列的手册候选列表[(手册名, 别名得分)]
        """
        scores: List[Tuple[str, float]] = []
        for manual_name, aliases in self.manual_alias_map.items():
            hits = [alias for alias in aliases if alias and alias.lower() in query.lower()]
            if hits:
                # 长别名命中得分更高（min(len,8)防止极端别名）
                alias_score = sum(min(len(alias), 8) for alias in hits)
                scores.append((manual_name, float(alias_score)))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def _merge_external_manual_candidates(
        self,
        detected_candidates: List[Tuple[str, float]],
        external_candidates: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        合并外部传入的候选手册与本地检测结果。

        外部候选（如来自多模态理解器的 product_candidates）
        与本地别名检测结果合并后去重，按得分降序排列。

        Args:
            detected_candidates: 本地别名检测结果
            external_candidates: 外部传入的候选手册名列表

        Returns:
            合并后的手册候选列表
        """
        if not external_candidates:
            return detected_candidates

        # 构建已有手册名集合
        existing = {name for name, _ in detected_candidates}

        # 外部候选中不在已有集合里的，加入结果（给默认分 1.0）
        for name in external_candidates:
            if name not in existing:
                detected_candidates.append((name, 1.0))
                existing.add(name)

        # 重新按得分降序
        detected_candidates.sort(key=lambda item: item[1], reverse=True)
        return detected_candidates

    def _contains_code_word(self, text: str) -> bool:
        """
        检测文本中是否包含产品型号/代码词。

        代码词（如 DCB107、ABC-123）是说明书中常见的精确匹配单元，
        检测到后建议在检索时启用 hybrid 模式，以改善 BM25 对精确词项的召回能力。

        Returns:
            是否检测到代码词
        """
        for pattern in CODE_WORD_PATTERNS:
            if pattern.search(text):
                logger.debug(f"检测到代码词，启用 hybrid 检索: {pattern.pattern} in '{text}'")
                return True
        return False

    def _should_use_local_manual_recall(
        self,
        candidate_manuals: Sequence[Tuple[str, float]],
    ) -> bool:
        """高置信候选手册优先走局部召回，低置信候选仍保留全库兜底。"""
        if not settings.route_manual_local_recall_enabled or not candidate_manuals:
            return False
        if len(candidate_manuals) > settings.route_manual_local_recall_max_manuals:
            return False

        top_score = float(candidate_manuals[0][1])
        second_score = float(candidate_manuals[1][1]) if len(candidate_manuals) > 1 else 0.0

        if top_score >= settings.route_manual_local_recall_strong_score:
            return True
        if len(candidate_manuals) == 1 and top_score >= settings.route_manual_local_recall_min_score:
            return True
        if (
            top_score >= settings.route_manual_local_recall_min_score
            and (top_score - second_score) >= settings.route_manual_local_recall_min_gap
        ):
            return True
        return False

    def _local_semantic_recall(
        self,
        query: str,
        candidate_docs: Sequence[Document],
        top_k: int,
    ) -> Dict[str, Dict[str, Any]]:
        """在候选手册子集内做局部语义召回，减少全库热门手册噪音。"""
        if not candidate_docs or self.rag_engine is None:
            return {}

        embedding_model = self.rag_engine.knowledge_base.embedding_model
        if embedding_model is None:
            return {}

        query_embedding = embedding_model.encode(
            [query],
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_vector = np.asarray(query_embedding[0], dtype="float32")
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm > 0:
            query_vector = query_vector / query_norm

        rescored: List[Tuple[Dict[str, Any], float]] = []
        for doc in candidate_docs:
            doc_vector = self._get_manual_doc_vector(doc)
            if doc_vector is None:
                continue
            semantic_score = float(np.dot(query_vector, doc_vector))
            item = {
                "content": doc.content,
                "doc_id": doc.doc_id,
                "relevance_score": semantic_score,
                "metadata": dict(doc.metadata),
                "has_image": "<PIC>" in doc.content,
                "image_ids": re.findall(r'\[([^\]]+)\]', doc.content),
            }
            rescored.append((item, semantic_score))

        rescored.sort(key=lambda pair: pair[1], reverse=True)
        return {
            item["doc_id"]: item
            for item, _score in rescored[:top_k]
        }

    def _get_manual_doc_vector(self, doc: Document) -> Optional[np.ndarray]:
        """从 FAISS 索引中恢复并缓存文档向量，避免重复编码候选文档。"""
        cached = self.manual_doc_vector_cache.get(doc.doc_id)
        if cached is not None:
            return cached

        if self.rag_engine is None or self.rag_engine.knowledge_base.text_embeddings is None:
            return None

        doc_idx = self.manual_doc_index_map.get(doc.doc_id)
        if doc_idx is None:
            return None

        vector = self.rag_engine.knowledge_base.text_embeddings.reconstruct(doc_idx)
        vector = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        self.manual_doc_vector_cache[doc.doc_id] = vector
        return vector

    def _should_force_service_route(
        self,
        matched_intents: Sequence[str],
        candidate_manuals: Sequence[Tuple[str, float]],
        manual_keyword_hits: Sequence[str],
        service_only_hits: Sequence[str],
    ) -> bool:
        """
        判断是否强制路由到service。

        强制service的条件（全部满足）:
        1. 无手册候选（query未明确指定产品）
        2. 无手册关键词命中
        3. 有纯客服意图词（如"以旧换新"）或有非manual_instruction的客服意图
        """
        if candidate_manuals or manual_keyword_hits:
            return False

        if service_only_hits:
            return True

        service_intents = [intent for intent in matched_intents if intent != "manual_instruction"]
        return bool(service_intents)

    def _rerank_manual_results(
        self,
        query: str,
        broad_results: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        rescored: List[Tuple[Dict[str, Any], float]] = []
        for item in broad_results:
            manual_name = item.get("metadata", {}).get("manual_name", "")
            final_score = self._score_manual_result(
                query=query,
                item=item,
                manual_name=manual_name,
                candidate_bonus=0.0,
                semantic_score=float(item.get("relevance_score", 0.0)),
            )
            rescored.append((item, final_score))

        rescored.sort(key=lambda pair: pair[1], reverse=True)
        reranked = [self._clone_with_score(item, score) for item, score in rescored[:top_k]]
        return self._apply_diversity_penalty(reranked)

    def _rerank_manual_docs(
        self,
        query: str,
        candidate_docs: Sequence[Document],
        broad_by_doc_id: Dict[str, Dict[str, Any]],
        candidate_manuals: Sequence[Tuple[str, float]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        manual_bonus_map = {manual_name: score for manual_name, score in candidate_manuals}
        rescored: List[Tuple[Dict[str, Any], float]] = []

        for doc in candidate_docs:
            semantic_item = broad_by_doc_id.get(doc.doc_id)
            semantic_score = float(semantic_item.get("relevance_score", 0.0)) if semantic_item else 0.0
            manual_name = str(doc.metadata.get("manual_name", "")).strip()
            candidate_bonus = manual_bonus_map.get(manual_name, 0.0)
            item = {
                "content": doc.content,
                "doc_id": doc.doc_id,
                "relevance_score": semantic_score,
                "metadata": dict(doc.metadata),
                "has_image": "<PIC>" in doc.content,
                "image_ids": re.findall(r'\[([^\]]+)\]', doc.content),
            }
            final_score = self._score_manual_result(
                query=query,
                item=item,
                manual_name=manual_name,
                candidate_bonus=candidate_bonus,
                semantic_score=semantic_score,
            )
            rescored.append((item, final_score))

        rescored.sort(key=lambda pair: pair[1], reverse=True)
        reranked = [self._clone_with_score(item, score) for item, score in rescored[:top_k]]
        return self._apply_diversity_penalty(reranked)

    def _score_manual_result(
        self,
        query: str,
        item: Dict[str, Any],
        manual_name: str,
        candidate_bonus: float,
        semantic_score: float,
    ) -> float:
        """
        手册检索结果的多维度评分函数。

        评分维度与权重:
        - semantic_score(0.34): 向量语义相似度，最重要
        - content_score(0.26): 查询词与内容文本的词项相似度
        - section_score(0.22): 查询词与章节标题的相似度
        - alias_score(0.12): 查询命中手册别名数量
        - candidate_scaled(0.12): 手册候选加分（query明确指定了手册时提高相关文档）
        - manual_name_score(0.06): 查询与手册名的相似度
        - image_bonus(+0.06): 查询涉及图示时给含图文档加分

        所有分项先归一化到[0,1]再加权求和，确保各维度可比。

        Args:
            query: 用户查询文本
            item: 检索结果条目
            manual_name: 该文档所属手册名
            candidate_bonus: 来自_detect_manual_candidates的手册候选加分
            semantic_score: 向量检索的语义相似度

        Returns:
            [0,1]范围内的综合相关性分数
        """
        metadata = item.get("metadata", {})
        section_title = str(metadata.get("section_title", "")).strip()
        content = item.get("content", "")
        aliases = self.manual_alias_map.get(manual_name, [])

        # 内容词项相似度
        content_score = self._lexical_similarity(query, content)
        # 章节标题词项相似度
        section_score = self._lexical_similarity(query, section_title)
        # 别名命中数量
        alias_hits = sum(1 for alias in aliases if alias and alias.lower() in query.lower())
        # 别名得分: 命中数量 / min(别名总数,4)，最多归一化为1.0
        alias_score = min(1.0, alias_hits / max(1, min(len(aliases), 4)))
        # 手册名相似度（去掉"手册"后缀）
        manual_name_score = self._lexical_similarity(query, manual_name.replace("手册", ""))
        # 图片加分: 查询涉及图示类词汇时，给含图文档额外加分
        image_bonus = 0.06 if item.get("image_ids") and any(token in query for token in ("图", "图片", "图示", "标识", "指示灯", "说明书")) else 0.0

        # 候选加分归一化: candidate_bonus是别名匹配得分，归一化到[0,1]
        candidate_scaled = min(1.0, candidate_bonus / 8.0) if candidate_bonus else 0.0
        return (
            semantic_score * 0.34
            + content_score * 0.26
            + section_score * 0.22
            + alias_score * 0.12
            + manual_name_score * 0.06
            + candidate_scaled * 0.12
            + image_bonus
        )

    # 多样性惩罚系数：Top2/Top3 命中同 section_title 时额外降权
    _SECTION_DIVERSITY_PENALTY: float = 0.05
    # 相邻 chunk 惩罚系数：同 section_title + 相邻 chunk_index 时降权
    _CHUNK_DIVERSITY_PENALTY: float = 0.03

    def _apply_diversity_penalty(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        对重排后的候选结果做轻量多样性惩罚，防止 top3 全是同一章节的连续块。

        惩罚策略（Top2/Top3 才生效，Top1 不降权）：
        - 同 section_title：降 _SECTION_DIVERSITY_PENALTY（0.05）
        - 同 section_title + 相邻 chunk_index：叠加 _CHUNK_DIVERSITY_PENALTY（0.03）
        - 同手册但不同章节：不做惩罚（保留步骤类问题的上下文连续性）

        这样比 MMR-lite 更轻量，不会压掉真正最相关的连续步骤块。

        Args:
            items: 已按相关性分数降序排列的检索结果列表

        Returns:
            多样性惩罚后的结果列表（分数可能下降，顺序不变）
        """
        if len(items) <= 1:
            return items

        result = [dict(item) for item in items]

        for i in range(1, len(result)):
            if i > 2:  # 只对 Top2/Top3 生效
                break

            current = result[i]
            current_meta = current.get("metadata", {})
            current_section = str(current_meta.get("section_title", "")).strip()
            current_chunk_idx = int(current_meta.get("chunk_index", -1))

            for j in range(i):
                prior = result[j]
                prior_meta = prior.get("metadata", {})
                prior_section = str(prior_meta.get("section_title", "")).strip()
                prior_chunk_idx = int(prior_meta.get("chunk_index", -1))

                if current_section != prior_section:
                    continue

                # 同 section_title 触发基础惩罚
                penalty = self._SECTION_DIVERSITY_PENALTY
                # 同 section + 相邻 chunk 叠加额外惩罚
                if current_chunk_idx >= 0 and prior_chunk_idx >= 0:
                    if abs(current_chunk_idx - prior_chunk_idx) <= 2:
                        penalty += self._CHUNK_DIVERSITY_PENALTY

                current["relevance_score"] = round(
                    max(0.0, current["relevance_score"] - penalty), 6
                )

        return result

    def _clone_with_score(self, item: Dict[str, Any], score: float) -> Dict[str, Any]:
        """
        深拷贝检索结果条目并附加最终相关性分数。

        注意: 对 metadata 和 image_ids 做深拷贝，防止多引用场景下意外修改原始数据。

        Args:
            item: 检索结果条目字典
            score: 计算后的最终相关性分数

        Returns:
            包含 relevance_score 字段的新字典
        """
        cloned = {
            **item,
            "metadata": dict(item.get("metadata", {})),
            "image_ids": list(item.get("image_ids", [])),
        }
        cloned["relevance_score"] = round(float(score), 6)
        return cloned

    def _retrieve_service(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        客服知识库检索。

        检索流程:
        1. 用意图匹配确定query命中了哪些客服意图
        2. 对每个客服文档计算多维度相关性分数
        3. 按分数降序截取top_k个文档

        评分维度:
        - score(0.52): 查询与文档内容的词项相似度
        - title_score(0.12): 查询与文档标题的相似度
        - keyword_bonus(0.18): 文档所属意图的关键词命中加分
        - intent_bonus(+0.18): 文档意图与query命中意图一致时额外加分
        - doc_type_bonus: 不同文档类型有不同的基础加分
        - priority: 文档优先级权重（service_policy=0.82优先级最高）

        Args:
            query: 用户查询文本
            top_k: 返回的最大文档数

        Returns:
            格式化后的客服检索结果列表
        """
        scored_docs: List[Tuple[Dict[str, Any], float]] = []
        # 匹配query命中了哪些客服意图
        matched_intents = self._match_service_intents(query)
        for doc in self.service_documents:
            metadata = doc.get("metadata", {})
            # 若query有明确的意图筛选，则只考虑匹配意图的文档
            if matched_intents and metadata.get("intent", "") not in matched_intents:
                continue
            score = self._lexical_similarity(query, doc["content"])
            title_score = self._lexical_similarity(query, metadata.get("title", ""))
            keyword_bonus = self._intent_keyword_bonus(query, metadata.get("intent", ""))
            # 命中意图时额外加分
            intent_bonus = 0.18 if metadata.get("intent", "") in matched_intents else 0.0
            doc_type = metadata.get("doc_type", "")
            # 文档优先级权重，service_policy文档权重最高(0.82)
            priority = float(metadata.get("priority", 0.8))
            doc_type_bonus = {
                "service_policy": 0.12,   # 客服政策文档质量最高
                "service_playbook": 0.05,  # 客服话术手册次之
                "question_example": 0.0,    # 题目标例不加分
            }.get(doc_type, 0.02)
            # 计算加权综合分数，再乘以优先级权重
            final_score = (
                score * 0.52
                + title_score * 0.12
                + keyword_bonus * 0.18
                + intent_bonus
                + doc_type_bonus
            ) * priority
            if final_score > 0:
                scored_docs.append((doc, final_score))

        # 按分数降序排列
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for doc, score in scored_docs[:top_k]:
            results.append(
                {
                    "content": doc["content"],
                    "doc_id": doc["doc_id"],
                    "relevance_score": round(float(score), 6),
                    "metadata": {
                        **doc.get("metadata", {}),
                        "route": "service",
                    },
                    "has_image": False,
                    "image_ids": [],
                }
            )
        return results

    def _intent_keyword_bonus(self, query: str, intent: str) -> float:
        """
        计算指定意图的关键词命中加分。

        计算方式: 命中数 / max(2, len(keywords)/3)，归一化到[0,1]。
        分母中取max(2, ...)是为了避免关键词极少的意图被过度放大。
        例如: 意图有6个关键词，命中2个 -> 2/max(2,2) = 1.0

        Args:
            query: 用户查询文本
            intent: 客服意图类型

        Returns:
            [0,1]范围内的关键词命中加分
        """
        spec = self.intent_specs.get(intent, {})
        keywords = spec.get("keywords", [])
        hits = sum(1 for keyword in keywords if keyword in query)
        if not keywords:
            return 0.0
        return min(1.0, hits / max(2, len(keywords) / 3))

    def _match_service_intents(self, query: str) -> List[str]:
        """
        从意图规格中匹配query命中的客服意图。

        匹配方式: 遍历所有意图规格，若query包含该意图的任一关键词，
        则将该意图加入匹配列表。

        Returns:
            匹配的意图类型列表（可能为空）
        """
        intents: List[str] = []
        for intent, spec in self.intent_specs.items():
            if any(keyword in query for keyword in spec.get("keywords", [])):
                intents.append(intent)
        return intents

    def _merge_results(
        self,
        route: str,
        service_results: Sequence[Dict[str, Any]],
        manual_results: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        合并service和manual两路检索结果。

        合并策略:
        - pure service/manual: 直接返回对应路由的结果
        - mixed: 先各取top_k个结果，service结果排在前面，
          在排序时service结果优先(主排序键)，再按相关性分数降序

        Args:
            route: 当前路由类型
            service_results: 客服检索结果
            manual_results: 手册检索结果

        Returns:
            合并后的检索结果列表
        """
        if route == "service":
            return list(service_results)
        if route == "manual":
            return list(manual_results)

        # mixed路由: 各取top_k后合并
        service_primary = list(service_results[: settings.route_service_top_k])
        manual_primary = list(manual_results[: settings.route_manual_top_k])
        merged = service_primary + manual_primary
        # 排序: service优先，再按相关性分数降序
        merged.sort(
            key=lambda item: (
                1 if item.get("metadata", {}).get("route") == "service" else 0,
                item.get("relevance_score", 0.0),
            ),
            reverse=True,
        )
        return merged

    def _best_example_score(self, query: str, target_route: str) -> float:
        """
        计算query与目标路由的示例问题的最佳相似度。

        用途: 作为规则路由评分的参考维度之一，
        相似度越高说明query越可能属于该路由。

        Returns:
            [0,1]范围内的最高相似度分数
        """
        best = 0.0
        for example in self.route_examples:
            if example.get("metadata", {}).get("route") != target_route:
                continue
            score = self._lexical_similarity(query, example.get("content", ""))
            if score > best:
                best = score
        return best

    def _lexical_similarity(self, left: str, right: str) -> float:
        """
        词项级别的文本相似度计算。

        算法: Jaccard与Coverage的加权融合
        - Jaccard(权重0.45): 两个文本词项集合的交并比
        - Coverage(权重0.55): 查询词项在目标文本中的覆盖率
        Coverage权重略高是因为客服场景更关注查询词是否被覆盖

        词项提取使用2-gram和3-gram中文字符切分，
        无需依赖外部分词器，适合CPU低资源环境。

        Args:
            left: 查询文本
            right: 目标文本

        Returns:
            [0,1]范围内的相似度分数
        """
        left_terms = self._extract_terms(left)
        right_terms = self._extract_terms(right)
        if not left_terms or not right_terms:
            return 0.0

        inter = len(left_terms & right_terms)
        union = len(left_terms | right_terms)
        if union == 0:
            return 0.0

        # Jaccard: 集合相似度
        jaccard = inter / union
        # Coverage: 查询词项在目标文本中的覆盖率
        coverage = inter / max(1, len(left_terms))
        # 加权融合，Coverage权重略高
        return round(jaccard * 0.45 + coverage * 0.55, 6)

    def _extract_terms(self, text: str) -> set[str]:
        """
        将文本转换为词项集合，用于相似度计算。

        词项提取策略:
        - 英文/数字/下划线/连字符: 整体作为词项
        - 单字符中文: 直接作为词项
        - 连续中文(>=2字符): 同时提取 2-gram 和 3-gram 词项
          例如: "人体工学椅" -> {"人体", "体工", "工学", "学椅", "人体工", ...}

        采用 2-gram 策略是为了在不使用分词器的情况下捕获中文词项的重叠，
        适合 CPU 低资源环境。2-gram 产生的词项更短，匹配概率更高，
        但也会引入更多噪音，因此同时计算 3-gram 作为补充。

        Args:
            text: 输入文本

        Returns:
            词项集合（小写）
        """
        text = (text or "").strip().lower()
        if not text:
            return set()

        terms = set(re.findall(r"[a-z0-9_-]+", text))
        chinese_segments = re.findall(r"[\u4e00-\u9fff]+", text)
        for segment in chinese_segments:
            if len(segment) == 1:
                terms.add(segment)
                continue
            # 使用 2-gram 提高中文词项重合度，避免必须依赖分词器
            for idx in range(len(segment) - 1):
                terms.add(segment[idx: idx + 2])
            for idx in range(len(segment) - 2):
                terms.add(segment[idx: idx + 3])
        return terms


def get_dual_route_retriever() -> DualRouteRetriever:
    global _dual_route_retriever
    if _dual_route_retriever is None:
        _dual_route_retriever = DualRouteRetriever()
    return _dual_route_retriever


def reset_dual_route_retriever() -> None:
    """重置全局双路检索器单例，便于测试和阈值扫描。"""
    global _dual_route_retriever
    _dual_route_retriever = None
