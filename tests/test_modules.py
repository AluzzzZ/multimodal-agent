"""
模块单元测试

测试覆盖:
1. 文本处理工具
2. 图片处理工具
3. 对话管理器
4. RAG引擎(单元层面)
5. 幻觉控制器(单元层面)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===== 文本处理测试 =====

class TestTextProcessor:
    """文本处理器测试"""

    def test_clean_text(self):
        """测试文本清理"""
        from src.utils.text_utils import TextProcessor
        text = "  你好   世界  \n\n  测试  "
        result = TextProcessor.clean_text(text)
        assert "你好" in result
        assert "世界" in result

    def test_split_sentences(self):
        """测试分句"""
        from src.utils.text_utils import TextProcessor
        text = "这是第一句。这是第二句？这是第三句！"
        sentences = TextProcessor.split_sentences(text)
        assert len(sentences) >= 3

    def test_truncate_text(self):
        """测试文本截断"""
        from src.utils.text_utils import TextProcessor
        text = "这是一个很长的文本内容"
        result = TextProcessor.truncate_text(text, max_length=10)
        assert len(result) <= 13  # 10 + "..."


class TestQueryProcessor:
    """查询处理器测试"""

    def test_expand_query(self):
        """测试查询扩展"""
        from src.utils.text_utils import QueryProcessor
        query = "电钻故障"
        expanded = QueryProcessor.expand_query(query)
        assert query in expanded

    def test_extract_entities(self):
        """测试实体提取"""
        from src.utils.text_utils import QueryProcessor
        query = "DCB107电钻无法启动"
        entities = QueryProcessor.extract_entities(query)
        assert "DCB107" in entities["product_models"]

    def test_detect_language(self):
        """测试语言检测"""
        from src.utils.text_utils import QueryProcessor
        zh_text = "这是一个中文文本"
        en_text = "This is English text"

        assert QueryProcessor.detect_language(zh_text) == "zh"
        assert QueryProcessor.detect_language(en_text) == "en"

    def test_normalize_english_query_for_retrieval(self):
        """英文题应先归一化为中文检索友好查询"""
        from src.utils.text_utils import QueryProcessor

        result = QueryProcessor.normalize_query_for_retrieval(
            "How to remove the bimini top of the boat?"
        )
        assert result["language"] == "en"
        assert "遮阳篷" in result["normalized_query"]
        assert "摩托艇" in result["normalized_query"]
        assert result["translation_applied"] is True


class TestRouteClassifier:
    """路由分类器测试"""

    def test_classifier_featureizer_builds_input_text(self):
        from src.modules.route_classifier import RouteClassifierFeatureizer

        featureizer = RouteClassifierFeatureizer(dim=64)
        text = featureizer.build_input_text(
            normalized_query="如何 安装 锚灯 摩托艇",
            images=["data:image/png;base64,abc"],
            image_tags=["安装部件", "产品外观"],
        )
        assert "[HAS_IMAGE]" in text
        assert "安装部件" in text

    def test_classifier_disabled_result_when_model_missing(self):
        from src.modules.route_classifier import RouteClassifier

        classifier = RouteClassifier()
        classifier.model_dir = Path("nonexistent_model_dir")
        result = classifier.predict("退款 规则")
        assert result["available"] is False
        assert result["fallback_reason"] in {"classifier_unavailable", "classifier_not_initialized"}


# ===== 图片处理测试 =====

class TestImageProcessor:
    """图片处理器测试"""

    def test_extract_image_ids(self):
        """测试图片ID提取"""
        from src.utils.image_utils import extract_image_ids_from_text
        text = "请参考图片[PIC001]和[PIC002]"
        ids = extract_image_ids_from_text(text)
        assert "PIC001" in ids
        assert "PIC002" in ids

    def test_extract_image_ids_empty(self):
        """测试无图片ID的情况"""
        from src.utils.image_utils import extract_image_ids_from_text
        text = "这是一段普通文本"
        ids = extract_image_ids_from_text(text)
        assert len(ids) == 0

    def test_extract_pic_markers(self):
        """测试<PIC>标记提取"""
        from src.utils.image_utils import extract_image_ids_from_text
        text = "电池充电中<PIC>[drill0_04]\n充满电<PIC>[drill0_05]"
        ids = extract_image_ids_from_text(text)
        assert "drill0_04" in ids
        assert "drill0_05" in ids


# ===== 对话管理器测试 =====

class TestConversationManager:
    """对话管理器测试"""

    def test_create_session(self):
        """测试创建会话"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session_id = manager.create_session()
        assert session_id is not None
        assert len(session_id) > 0

    def test_create_session_with_user_id(self):
        """测试带用户ID创建会话"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session_id = manager.create_session(user_id="user_123")
        assert session_id is not None

        # 验证用户索引
        sessions = manager.list_sessions(user_id="user_123")
        assert session_id in sessions

    def test_add_message(self):
        """测试添加消息"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session_id = manager.create_session()

        success = manager.add_message(
            session_id=session_id,
            role="user",
            content="测试消息"
        )
        assert success is True

        history = manager.get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0]["content"] == "测试消息"
        assert history[0]["role"] == "user"

    def test_add_message_with_images(self):
        """测试添加带图片的消息"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session_id = manager.create_session()

        success = manager.add_message(
            session_id=session_id,
            role="user",
            content="这是带图片的消息",
            images=["img1", "img2"]
        )
        assert success is True

        history = manager.get_conversation_history(session_id)
        assert len(history[0]["images"]) == 2

    def test_multi_turn_conversation(self):
        """测试多轮对话"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session_id = manager.create_session()

        # 添加多轮对话
        manager.add_message(session_id, "user", "第一轮问题")
        manager.add_message(session_id, "assistant", "第一轮回答")
        manager.add_message(session_id, "user", "第二轮问题")

        history = manager.get_conversation_history(session_id)
        assert len(history) == 3

    def test_get_history_with_limit(self):
        """测试限制返回的历史数量"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session_id = manager.create_session()

        for i in range(10):
            manager.add_message(session_id, "user", f"消息{i}")

        history = manager.get_conversation_history(session_id, limit=3)
        assert len(history) == 3

    def test_get_nonexistent_session(self):
        """测试获取不存在的会话"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session = manager.get_session("nonexistent-id")
        assert session is None

    def test_clear_session(self):
        """测试清除会话"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session_id = manager.create_session()
        manager.add_message(session_id, "user", "测试")

        # 清除会话
        success = manager.clear_session(session_id)
        assert success is True

        # 验证会话已删除
        session = manager.get_session(session_id)
        assert session is None

    def test_context_summary(self):
        """测试上下文摘要"""
        from src.modules.conversation_manager import ConversationManager
        manager = ConversationManager()
        manager.initialize()

        session_id = manager.create_session()
        manager.add_message(session_id, "user", "用户问题")
        manager.add_message(session_id, "assistant", "助手回答")

        summary = manager.get_context_summary(session_id, last_n=2)
        assert "用户问题" in summary
        assert "助手回答" in summary


class TestDialogueState:
    """对话状态测试"""

    def test_single_question(self):
        """测试单问题场景"""
        from src.modules.conversation_manager import DialogueState
        state = DialogueState()
        state.set_questions(["只有一个问题"])

        assert not state.is_multi_question
        assert state.get_next_question() == "只有一个问题"
        assert not state.is_complete()

    def test_multi_question(self):
        """测试多问题场景"""
        from src.modules.conversation_manager import DialogueState
        state = DialogueState()
        questions = ["问题1", "问题2", "问题3"]
        state.set_questions(questions)

        assert state.is_multi_question
        assert state.get_next_question() == "问题1"

        state.mark_answered("问题1")
        assert state.get_next_question() == "问题2"
        assert not state.is_complete()

        state.mark_answered("问题2")
        state.mark_answered("问题3")
        assert state.is_complete()


# ===== RAG引擎测试 =====

class TestRAGEngine:
    """RAG引擎测试 - 使用mock"""

    def test_document_creation(self):
        """测试Document创建"""
        from src.modules.rag_engine import Document
        doc = Document(
            content="测试内容",
            doc_id="test_001",
            metadata={"category": "test"}
        )
        assert doc.content == "测试内容"
        assert doc.doc_id == "test_001"

        # 测试序列化
        doc_dict = doc.to_dict()
        assert doc_dict["doc_id"] == "test_001"
        assert doc_dict["metadata"]["category"] == "test"

    def test_reranker_initialization(self):
        """测试重排序器初始化"""
        from src.modules.rag_engine import Reranker
        reranker = Reranker()
        reranker.initialize()
        assert reranker._initialized is True


# ===== 幻觉控制器测试 =====

class TestHallucinationController:
    """幻觉控制器测试"""

    def test_simple_verification(self):
        """测试简单验证(无LLM)"""
        from src.modules.hallucination_controller import HallucinationController
        controller = HallucinationController()
        controller.initialize()

        answer = "电池充电中时指示灯会闪烁"
        context = ["电池充电中时指示灯会闪烁表示正在充电"]

        result = controller._simple_verification(answer, context)
        assert "confidence" in result

    def test_confidence_issue_detection(self):
        """测试置信度问题检测"""
        from src.modules.hallucination_controller import HallucinationController
        controller = HallucinationController()
        controller.initialize()

        text = "这个问题可能大概也许我知道"
        issues = controller.detect_confidence_issues(text)
        assert len(issues) > 0


class TestChainOfThoughtReasoner:
    """思维链推理器测试"""

    def test_simple_decomposition(self):
        """测试简单问题分解(无LLM)"""
        from src.modules.hallucination_controller import ChainOfThoughtReasoner
        reasoner = ChainOfThoughtReasoner()
        reasoner.initialize()

        # 简单问题应该被识别为非复杂
        result = reasoner.decompose_question("你好")
        assert "is_complex" in result


# ===== 知识库构建脚本测试 =====

class TestKnowledgeBaseBuilder:
    """知识库构建器测试"""

    def test_manual_parser(self):
        """测试手册解析"""
        from scripts.build_knowledge_base import ManualParser

        content = '["# 标题\\n这是测试内容<PIC>", ["img1"]]'
        text, image_ids = ManualParser.parse_manual_file(content)
        bound = ManualParser.inject_image_ids(text, image_ids)
        assert "测试内容" in text
        assert "img1" in image_ids
        assert "<PIC>[img1]" in bound

    def test_section_extraction(self):
        """测试章节提取"""
        from scripts.build_knowledge_base import ManualParser

        content = "# 第一章\n内容1\n\n# 第二章\n内容2"
        sections = ManualParser.extract_sections(content)
        assert len(sections) >= 2
        assert sections[0]["title"] == "第一章"

    def test_extract_image_ids(self):
        """测试从文本块提取图片ID"""
        from scripts.build_knowledge_base import ManualParser

        content = "测试内容<PIC>[img1]\n其他说明<PIC>[img2]"
        result = ManualParser.extract_image_ids(content)
        assert result == ["img1", "img2"]


class TestDualRouteMixed:
    """双路 mixed 联动测试"""

    def test_route_query_marks_manual_instruction_as_mixed(self):
        """当问题同时包含客服政策与说明书诉求时，应进入 mixed 路。"""
        from src.modules.dual_route_retriever import DualRouteRetriever

        retriever = DualRouteRetriever()
        retriever._initialized = True
        retriever.service_keywords = ["安装", "配件费", "说明书", "电子版"]
        retriever.manual_keywords = ["空调"]
        retriever.intent_specs = {
            "installation_service": {"keywords": ["安装", "配件费"]},
            "manual_instruction": {"keywords": ["说明书", "电子版"]},
        }
        retriever._best_example_score = MagicMock(side_effect=[0.72, 0.34])

        route_info = retriever.route_query("空调安装需要额外配件费吗？说明书电子版在哪里找？")
        assert route_info["route"] == "mixed"
        assert route_info["has_manual_instruction_intent"] is True

    def test_mixed_answer_contains_service_and_manual_support(self):
        """mixed 场景下应同时给出客服政策答复和手册图示引导。"""
        from src.modules.response_generator import ResponseGenerator

        generator = ResponseGenerator()
        answer = generator._compose_mixed_sub_answer(
            question="安装收费怎么处理？能给我说明书图示吗？",
            service_sources=[
                {
                    "content": "【标准答复】\n1. 若原本承诺免费安装，但现场另行收费，需要先核实收费项目。\n【需核实信息】\n- 订单号\n- 收费明细",
                    "metadata": {"route": "service", "title": "上门安装与上门检修"},
                }
            ],
            manual_sources=[
                {
                    "content": "# 空调遥控器支架安装\n请按图示安装支架<PIC>[air_conditioner_01]",
                    "metadata": {"route": "manual", "manual_name": "空调手册"},
                    "relevance_score": 0.81,
                    "image_ids": ["air_conditioner_01"],
                }
            ],
            route_info={
                "route": "mixed",
                "manual_score": 0.81,
                "manual_keyword_hits": ["空调"],
                "has_manual_instruction_intent": True,
            },
        )
        assert "收费" in answer
        assert "空调手册" in answer
        assert "<PIC>" in answer


class TestManualRouteRefinement:
    """manual 路产品级收缩与章节重排测试"""

    def test_detect_manual_candidates_by_alias(self):
        from src.modules.dual_route_retriever import DualRouteRetriever

        retriever = DualRouteRetriever()
        retriever.manual_alias_map = {
            "空调手册": ["空调", "遥控器", "自动运行模式"],
            "洗碗机手册": ["洗碗机", "亮碟剂"],
        }
        candidates = retriever._detect_manual_candidates("单冷型空调如何开启自动运行模式？")
        assert candidates[0][0] == "空调手册"

    def test_manual_section_rerank_prefers_relevant_section(self):
        from src.modules.dual_route_retriever import DualRouteRetriever
        from src.modules.rag_engine import Document

        retriever = DualRouteRetriever()
        retriever.manual_alias_map = {
            "空调手册": ["空调", "自动运行模式"],
        }

        docs = [
            Document(
                content="# 自动运行模式\n按 AUTO 键即可开启自动运行模式。",
                doc_id="ac_good",
                metadata={"manual_name": "空调手册", "section_title": "自动运行模式", "route": "manual"},
            ),
            Document(
                content="# 室内空气净化\n本章节介绍空气净化。",
                doc_id="ac_bad",
                metadata={"manual_name": "空调手册", "section_title": "室内空气净化", "route": "manual"},
            ),
        ]

        ranked = retriever._rerank_manual_docs(
            query="单冷型空调如何开启自动运行模式？",
            candidate_docs=docs,
            broad_by_doc_id={
                "ac_good": {
                    "content": docs[0].content,
                    "doc_id": "ac_good",
                    "relevance_score": 0.70,
                    "metadata": docs[0].metadata,
                    "has_image": False,
                    "image_ids": [],
                },
                "ac_bad": {
                    "content": docs[1].content,
                    "doc_id": "ac_bad",
                    "relevance_score": 0.72,
                    "metadata": docs[1].metadata,
                    "has_image": False,
                    "image_ids": [],
                },
            },
            candidate_manuals=[("空调手册", 8.0)],
            top_k=2,
        )
        assert ranked[0]["doc_id"] == "ac_good"

    def test_force_service_route_for_generic_commerce_questions(self):
        from src.modules.dual_route_retriever import DualRouteRetriever

        retriever = DualRouteRetriever()
        retriever._initialized = True
        retriever.service_keywords = ["质量问题", "客服", "售后"]
        retriever.manual_keywords = []
        retriever.manual_alias_map = {"空调手册": ["空调"]}
        retriever.intent_specs = {
            "repair_warranty": {"keywords": ["质量问题", "售后"]},
        }
        retriever._best_example_score = MagicMock(side_effect=[0.18, 0.62])

        route_info = retriever.route_query("你们的商品存在质量问题，我使用了一次就坏了，联系客服没人管！")
        assert route_info["route"] == "service"

    def test_force_service_route_for_coupon_question(self):
        from src.modules.dual_route_retriever import DualRouteRetriever

        retriever = DualRouteRetriever()
        retriever._initialized = True
        retriever.service_keywords = []
        retriever.manual_keywords = []
        retriever.manual_alias_map = {"冰箱手册": ["冰箱"]}
        retriever.intent_specs = {}
        retriever._best_example_score = MagicMock(side_effect=[0.12, 0.58])

        route_info = retriever.route_query("请问你们的优惠券能用于所有商品吗？")
        assert route_info["route"] == "service"

    def test_english_boat_query_detects_manual_candidate(self):
        from src.modules.dual_route_retriever import DualRouteRetriever

        retriever = DualRouteRetriever()
        retriever.manual_alias_map = {
            "摩托艇手册": ["摩托艇", "boat", "遮阳篷", "bimini top"],
        }
        candidates = retriever._detect_manual_candidates("如何拆卸摩托艇 遮阳篷 boat bimini top")
        assert candidates[0][0] == "摩托艇手册"

    def test_classifier_high_confidence_overrides_rule_route(self):
        from src.modules.dual_route_retriever import DualRouteRetriever

        retriever = DualRouteRetriever()
        final_route, classifier_used, reason = retriever._arbitrate_route(
            rule_info={
                "rule_route": "manual",
                "strong_rule_route": "",
                "has_service_policy_intent": False,
                "manual_candidates": [],
                "manual_keyword_hits": [],
            },
            classifier_result={
                "available": True,
                "label": "service",
                "confidence": 0.91,
                "probs": {"service": 0.91, "manual": 0.05, "mixed": 0.04},
            },
        )
        assert final_route == "service"
        assert classifier_used is True
        assert reason == "classifier_high_confidence"

    def test_strong_rule_route_overrides_classifier(self):
        from src.modules.dual_route_retriever import DualRouteRetriever

        retriever = DualRouteRetriever()
        final_route, classifier_used, reason = retriever._arbitrate_route(
            rule_info={
                "rule_route": "mixed",
                "strong_rule_route": "mixed",
                "has_service_policy_intent": True,
                "manual_candidates": ["空调手册"],
                "manual_keyword_hits": ["空调"],
            },
            classifier_result={
                "available": True,
                "label": "manual",
                "confidence": 0.88,
                "probs": {"service": 0.02, "manual": 0.88, "mixed": 0.10},
            },
        )
        assert final_route == "mixed"
        assert classifier_used is False
        assert reason == "strong_rule:mixed"


class TestManualStructuredAnswer:
    """manual 路列表/步骤型答案抽取测试"""

    def test_merge_followup_constraints(self):
        from src.modules.response_generator import ResponseGenerator

        generator = ResponseGenerator()
        parts = generator._split_simple_questions("使用冰箱冰柜时需要注意什么？只需告诉我手册中的前五条。")
        assert len(parts) == 1
        assert "前五条" in parts[0]

    def test_extract_structured_manual_points(self):
        from src.modules.response_generator import ResponseGenerator

        generator = ResponseGenerator()
        manual_sources = [
            {
                "content": "# 使用注意事项\n1. 请勿将热食直接放入冰箱。\n2. 连接电源前请检查插座接地。\n3. 请勿遮挡出风口。\n4. 清洁前请先断电。\n5. 儿童应在监护下使用。\n6. 请定期检查密封条。",
                "metadata": {"manual_name": "冰箱手册", "section_title": "使用注意事项"},
                "relevance_score": 0.82,
                "image_ids": [],
            }
        ]

        answer = generator._compose_manual_sub_answer(
            "使用冰箱冰柜时需要注意什么？只需告诉我手册中的前五条。",
            manual_sources,
            route_info={"route": "manual"},
        )
        assert "相关要点如下" in answer
        assert "1. 请勿将热食直接放入冰箱" in answer
        assert "5. 儿童应在监护下使用" in answer
        assert "6. 请定期检查密封条" not in answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
