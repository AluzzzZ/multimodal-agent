"""
多模态理解专项测试

重点覆盖 evidence_type 的规则打分逻辑：
1. 明显客服政策题 -> service_like
2. 明显说明书题 -> manual_like
3. 服务 + 手册复合题 -> mixed_like
4. 带图的纯 manual 问题不应被误判成 mixed_like
5. 缺乏业务语义的短句 -> unknown
"""

from src.modules.multimodal_understanding import MultimodalUnderstanding
from src.modules.understanding_types import ProductCandidate


class TestEvidenceTypeInference:
    """evidence_type 打分与判定测试"""

    def setup_method(self):
        self.mm = MultimodalUnderstanding()

    def test_service_like_for_refund_policy_question(self):
        question = "请问你们家的商品支持7天无理由退换货吗？需要自己承担运费吗？"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=[],
            product_candidates=[],
            visual_intents=[],
            history_context=None,
        )

        result = self.mm._infer_evidence_type(
            question,
            image_tags=[],
            product_candidates=[],
            visual_intents=[],
            history_context=None,
            evidence_scores=scores,
        )

        assert result == "service_like"
        assert scores["service_score"] > scores["manual_score"]

    def test_manual_like_for_manual_usage_question(self):
        question = "我的DCB107指示灯闪烁时，这些闪烁标识代表什么含义？"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=["指示灯", "充电器"],
            product_candidates=[ProductCandidate(name="电钻")],
            visual_intents=["查看指示灯"],
            history_context=None,
        )

        result = self.mm._infer_evidence_type(
            question,
            image_tags=["指示灯", "充电器"],
            product_candidates=[ProductCandidate(name="电钻")],
            visual_intents=["查看指示灯"],
            history_context=None,
            evidence_scores=scores,
        )

        assert result == "manual_like"
        assert scores["manual_score"] > scores["service_score"]

    def test_mixed_like_for_service_and_manual_compound_question(self):
        question = "这个商品坏了怎么申请售后？顺便问下这个按钮是什么意思？"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=["按钮", "控制面板"],
            product_candidates=[ProductCandidate(name="空调")],
            visual_intents=["查看按钮"],
            history_context=None,
        )

        result = self.mm._infer_evidence_type(
            question,
            image_tags=["按钮", "控制面板"],
            product_candidates=[ProductCandidate(name="空调")],
            visual_intents=["查看按钮"],
            history_context=None,
            evidence_scores=scores,
        )

        assert result == "mixed_like"
        assert scores["service_score"] >= 0.26
        assert scores["manual_score"] >= 0.26

    def test_manual_like_with_image_should_not_fall_back_to_mixed(self):
        question = "这个按钮怎么用？"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=["按钮", "控制面板"],
            product_candidates=[ProductCandidate(name="空调")],
            visual_intents=["查看按钮"],
            history_context=None,
        )

        result = self.mm._infer_evidence_type(
            question,
            image_tags=["按钮", "控制面板"],
            product_candidates=[ProductCandidate(name="空调")],
            visual_intents=["查看按钮"],
            history_context=None,
            evidence_scores=scores,
        )

        assert result == "manual_like"
        assert scores["mixed_score"] < scores["manual_score"]

    def test_unknown_for_non_task_short_message(self):
        question = "好的，知道了"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=[],
            product_candidates=[],
            visual_intents=[],
            history_context=None,
        )

        result = self.mm._infer_evidence_type(
            question,
            image_tags=[],
            product_candidates=[],
            visual_intents=[],
            history_context=None,
            evidence_scores=scores,
        )

        assert result == "unknown"

    def test_service_like_for_manual_delivery_request(self):
        question = "之前那款产品的说明书还能再发一份吗？"
        history_context = {
            "summary": "用户说：电钻DCB107充不进电怎么解决；助手说：DCB107电池充电器充不进电可能原因如下。",
            "recent_product": "电钻",
            "had_images": False,
            "turns_used": 2,
        }

        scores = self.mm._score_evidence_type(
            question,
            image_tags=[],
            product_candidates=[],
            visual_intents=[],
            history_context=history_context,
        )

        result = self.mm._infer_evidence_type(
            question,
            image_tags=[],
            product_candidates=[],
            visual_intents=[],
            history_context=history_context,
            evidence_scores=scores,
        )

        assert result == "service_like"
        assert scores["service_score"] > scores["manual_score"]

    def test_mixed_like_for_manual_plus_onsite_service(self):
        question = "空调滤网怎么清洗？可以上门服务吗？"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=[],
            product_candidates=[ProductCandidate(name="空调")],
            visual_intents=[],
            history_context=None,
        )

        result = self.mm._infer_evidence_type(
            question,
            image_tags=[],
            product_candidates=[ProductCandidate(name="空调")],
            visual_intents=[],
            history_context=None,
            evidence_scores=scores,
        )

        assert result == "mixed_like"
        assert scores["service_score"] >= 0.26
        assert scores["manual_score"] >= 0.26

    def test_service_like_with_packaging_tags_should_not_be_overbiased_to_manual(self):
        question = "配件收到了，帮我看下是正品吗？"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=["包装", "标签", "配件"],
            product_candidates=[],
            visual_intents=[],
            history_context=None,
        )

        result = self.mm._infer_evidence_type(
            question,
            image_tags=["包装", "标签", "配件"],
            product_candidates=[],
            visual_intents=[],
            history_context=None,
            evidence_scores=scores,
        )

        assert result == "service_like"
        assert scores["service_score"] >= scores["manual_score"]

    def test_infer_product_candidates_for_filter_question(self):
        candidates = self.mm._infer_product_candidates(
            "净水器滤芯多久换一次？不换会影响水质吗？",
            image_tags=[],
            history_context=None,
        )

        assert candidates
        assert candidates[0].name == "净水器"

    def test_service_like_for_history_based_manual_resend_request_with_product_candidate_object(self):
        question = "之前那款产品的说明书还能再发一份吗？"
        history_context = {
            "summary": "用户说：电钻DCB107充不进电怎么解决？；助手说：DCB107电池充电器充不进电可能原因如下。",
            "recent_product": ProductCandidate(name="电钻", score=1.0, source="text_alias"),
            "had_images": False,
            "turns_used": 2,
        }

        candidates = self.mm._infer_product_candidates(
            question,
            image_tags=[],
            history_context=history_context,
        )
        assert candidates
        assert candidates[0].name == "电钻"
        assert candidates[0].source == "history"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=[],
            product_candidates=candidates,
            visual_intents=["查看说明书"],
            history_context=history_context,
        )
        result = self.mm._infer_evidence_type(
            question,
            image_tags=[],
            product_candidates=candidates,
            visual_intents=["查看说明书"],
            history_context=history_context,
            evidence_scores=scores,
        )
        assert result == "service_like"

    def test_mixed_like_for_step_reference_with_usability_friction(self):
        question = "说明书上第3步说按A键，但我找不到A键在哪里"

        scores = self.mm._score_evidence_type(
            question,
            image_tags=["按键"],
            product_candidates=[],
            visual_intents=["查看位置", "查看说明书"],
            history_context=None,
        )
        result = self.mm._infer_evidence_type(
            question,
            image_tags=["按键"],
            product_candidates=[],
            visual_intents=["查看位置", "查看说明书"],
            history_context=None,
            evidence_scores=scores,
        )

        assert result == "mixed_like"
        assert scores["service_score"] >= 0.26
        assert scores["mixed_score"] >= 0.2
