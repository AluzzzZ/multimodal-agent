"""
VLM 结构化理解引擎

通过视觉大模型（GPT-4V / Qwen-VL 等）将用户问题 + 图片 + 对话历史
结构化为 MultimodalUnderstandingResult。

使用方式：
    from src.modules.vlm_understanding import VLMUnderstandingEngine
    engine = VLMUnderstandingEngine()
    result = engine.understand(question, images, history)

设计原则：
- 严格结构化 JSON 输出，强制校验
- 字段规范化（枚举值纠正、列表裁剪、空值补全）
- 不可用时静默回退，不抛异常污染主链路
- 只在有图片时才调用，无图片时直接返回 None
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, List, Optional

from PIL import Image
from loguru import logger

from config import settings
from src.modules.understanding_types import MultimodalUnderstandingResult, ProductCandidate
from src.utils.domain_knowledge import (
    PRODUCT_ALIAS_MAP,
    ALIAS_TO_PRODUCT,
    VISUAL_INTENT_MAP,
    ROUTE_SERVICE_HINTS,
    ROUTE_MANUAL_HINTS,
    TAG_PRIORITY,
)


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

_VLM_SYSTEM_PROMPT = """你是客服多模态理解器，负责从用户问题、图片和对话历史中提取结构化信息。
你只输出JSON，不输出任何解释、说明或额外文本。
输出必须是有效的JSON对象。

字段说明：
- normalized_query: 归一化后的查询文本，提取核心问题，去除口语化表达
- language: 语言，"zh" 或 "en"
- product_candidates: 候选产品列表，最多5个。每个元素格式为：{{"name": "产品名", "score": 0.0~1.0, "source": "来源"}}。来源取以下之一：text_alias（图文联合判断）、image_tag（仅图像判断）、vlm（视觉语言模型推断）。必须从以下产品中选择：{product_list}
- image_tags: 图片中提取的标签，最多12个，分类如下：
  部件词：指示灯、按钮、表带、屏幕、面板、充电器、滤网、电池 等
  状态词：充电中、待机、亮红灯、闪烁、破损、过热、故障 等
  图示词：示意图、爆炸图、电路图、操作面板、说明书 等
- visual_intents: 视觉意图列表，最多5个，从以下枚举中选择：
  查看位置、查看指示灯、查看按钮、查看说明书、查看图示、查看型号、查看屏幕、查看警告、查看表带、查看摇杆
- evidence_type: 问题证据类型，从以下枚举中选择：service_like, manual_like, mixed_like, unknown
  service_like：涉及退款/退货/发票/保修/物流/投诉/售后 等客服服务问题
  manual_like：涉及使用说明/操作步骤/故障排查/功能设置 等产品手册问题
  mixed_like：同时涉及服务和手册，或有图片辅助判断
  unknown：无法判断
- referenced_previous_object: 若问题引用了历史中的对象（如"那个按钮"、"上面那张图"），填写具体描述；否则填 null
- requires_history_resolution: 是否需要历史上下文才能准确理解
- confidence: 置信度，0.0~1.0，反映你对整体理解的确信程度"""


def _build_user_prompt(
    question: str,
    history_context: Optional[Dict[str, Any]],
) -> str:
    """构建用户侧 prompt，附带历史摘要。"""
    history_section = ""
    if history_context:
        summary = history_context.get("summary", "")
        recent_product = history_context.get("recent_product", "")
        if isinstance(recent_product, ProductCandidate):
            recent_product = recent_product.name
        had_images = history_context.get("had_images", False)
        history_section = f"""
对话历史摘要：
{summary}
最近提及产品：{recent_product or "无"}
历史中是否有图片：{"是" if had_images else "否"}
""".strip()

    return f"""{history_section}

用户问题：{question}

请根据以上信息，严格输出JSON："""


def _build_image_content(image_bytes_list: List[bytes]) -> List[Dict[str, Any]]:
    """将图片字节列表转换为 API content 块。"""
    content: List[Dict[str, Any]] = []
    for img_bytes in image_bytes_list:
        buffered = io.BytesIO()
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
    return content


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    从 VLM 返回文本中提取 JSON 对象。
    尝试三种策略：
    1. 直接 json.loads（VLM 已返回干净 JSON）
    2. 找 ```json ... ``` 代码块
    3. 找第一个 { ... } 对象
    """
    text = text.strip()

    # 策略 1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略 2：代码块
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 策略 3：找第一个 JSON 对象
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    """将任意值安全转换为 [0.0, 1.0] 范围的置信度。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_vlm_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化 VLM 输出 payload：
    - 补全缺失字段（用默认值）
    - 纠正枚举值
    - 裁剪列表长度
    - 标准化产品名
    """
    VALID_VISUAL_INTENTS = set(VISUAL_INTENT_MAP.keys())
    VALID_EVIDENCE_TYPES = {"service_like", "manual_like", "mixed_like", "unknown"}
    MAX_PRODUCTS = 5
    MAX_TAGS = 12
    MAX_INTENTS = 5

    # normalized_query
    normalized = payload.get("normalized_query", "")
    if not isinstance(normalized, str):
        normalized = str(normalized) if normalized else ""

    # language
    lang = str(payload.get("language", "zh")).lower()
    if lang not in ("zh", "en"):
        lang = "zh"

    # product_candidates：标准化为产品标准名，保留分数和来源
    raw_products: List[Any] = payload.get("product_candidates", [])
    if not isinstance(raw_products, list):
        raw_products = []
    product_candidates: List[Dict[str, Any]] = []
    seen_products: set = set()
    for p in raw_products[:MAX_PRODUCTS]:
        if isinstance(p, dict):
            name = str(p.get("name", "")).strip()
            score = _clamp_confidence(p.get("score"))
            source = str(p.get("source", "vlm")).strip() or "vlm"
        elif isinstance(p, str) and p.strip():
            name = p.strip()
            score = 0.8
            source = "vlm"
        else:
            continue
        if not name:
            continue
        # 用别名表标准化
        normalized_name = ALIAS_TO_PRODUCT.get(name.lower(), name)
        # 用产品表过滤（只保留已知产品）
        if normalized_name in PRODUCT_ALIAS_MAP and normalized_name not in seen_products:
            product_candidates.append({"name": normalized_name, "score": score, "source": source})
            seen_products.add(normalized_name)
        elif name in PRODUCT_ALIAS_MAP and name not in seen_products:
            product_candidates.append({"name": name, "score": score, "source": source})
            seen_products.add(name)

    # image_tags
    raw_tags: List[str] = payload.get("image_tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []
    image_tags: List[str] = []
    seen_tags: set = set()
    for tag in raw_tags[:MAX_TAGS]:
        if not isinstance(tag, str) or not tag.strip():
            continue
        tag = tag.strip()
        if tag not in seen_tags:
            image_tags.append(tag)
            seen_tags.add(tag)

    # 按优先级排序标签
    def tag_priority(t: str) -> int:
        return TAG_PRIORITY.get(t, 100)

    image_tags.sort(key=tag_priority)

    # visual_intents
    raw_intents: List[str] = payload.get("visual_intents", [])
    if not isinstance(raw_intents, list):
        raw_intents = []
    visual_intents: List[str] = []
    for intent in raw_intents[:MAX_INTENTS]:
        if not isinstance(intent, str):
            continue
        intent = intent.strip()
        if intent in VALID_VISUAL_INTENTS and intent not in visual_intents:
            visual_intents.append(intent)

    # evidence_type
    raw_evidence = str(payload.get("evidence_type", "unknown")).strip().lower()
    if raw_evidence not in VALID_EVIDENCE_TYPES:
        # 尝试从关键词推断
        question = normalized
        svc = sum(1 for kw in ROUTE_SERVICE_HINTS if kw in question)
        man = sum(1 for kw in ROUTE_MANUAL_HINTS if kw in question)
        if svc > man:
            raw_evidence = "service_like"
        elif man > svc:
            raw_evidence = "manual_like"
        else:
            raw_evidence = "unknown"
    evidence_type = raw_evidence

    # referenced_previous_object
    ref_obj = payload.get("referenced_previous_object")
    if ref_obj is None or ref_obj == "":
        ref_obj = None

    # requires_history_resolution
    raw_history = payload.get("requires_history_resolution")
    requires_history = bool(raw_history)

    # confidence
    confidence = _clamp_confidence(payload.get("confidence", 0.5))

    return {
        "normalized_query": normalized,
        "language": lang,
        "product_candidates": [
            ProductCandidate(name=p["name"], score=p["score"], source=p["source"])
            for p in product_candidates
        ],
        "image_tags": image_tags,
        "visual_intents": visual_intents,
        "evidence_type": evidence_type,
        "referenced_previous_object": ref_obj,
        "requires_history_resolution": requires_history,
        "confidence": confidence,
    }


def _validate_result(result: Dict[str, Any]) -> bool:
    """校验结果字段完整性。"""
    required_fields = [
        "normalized_query", "language", "product_candidates",
        "image_tags", "visual_intents", "evidence_type",
        "referenced_previous_object", "requires_history_resolution", "confidence",
    ]
    for field in required_fields:
        if field not in result:
            return False
    if result["language"] not in ("zh", "en"):
        return False
    if result["evidence_type"] not in ("service_like", "manual_like", "mixed_like", "unknown"):
        return False
    if not isinstance(result["product_candidates"], list):
        return False
    if not isinstance(result["image_tags"], list):
        return False
    return True


# ---------------------------------------------------------------------------
# 主引擎
# ---------------------------------------------------------------------------

class VLMUnderstandingEngine:
    """
    VLM 结构化理解引擎。

    将用户问题 + 图片 + 历史 → 结构化 JSON → MultimodalUnderstandingResult。

    调用流程：
    1. 组装 prompt（system + user）
    2. 调用 VLM API
    3. 解析 JSON（三层策略）
    4. 规范化字段（标准化产品名、纠正枚举、裁剪列表）
    5. 类型校验
    6. 返回结果，或返回 None 表示不可用
    """

    def __init__(self):
        self._client = None
        self._initialized = False

    def _ensure_initialized(self):
        """延迟初始化 API client 和 system prompt。"""
        if self._initialized:
            return
        if not settings.vision_llm_enabled:
            self._initialized = True
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.vision_llm_api_key or settings.llm_api_key,
                base_url=settings.vision_llm_base_url or settings.llm_base_url,
                timeout=settings.vlm_timeout_seconds,
            )
            self._initialized = True
            logger.info(f"VLMUnderstandingEngine initialized: {settings.vision_llm_model}")
        except Exception as e:
            logger.warning(f"VLMUnderstandingEngine 初始化失败: {e}")
            self._initialized = True

    def _get_system_prompt(self) -> str:
        """构建 system prompt，产品列表从 domain_knowledge 动态生成。"""
        product_list = "、".join(sorted(PRODUCT_ALIAS_MAP.keys()))
        return _VLM_SYSTEM_PROMPT.format(product_list=product_list)

    def understand(
        self,
        question: str,
        images: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        history_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[MultimodalUnderstandingResult]:
        """
        通过 VLM 进行结构化理解。

        Args:
            question: 用户问题
            images: Base64 图片列表（可选，无图片时返回 None）
            conversation_history: 对话历史（可选）
            history_context: 外部已构建的历史上下文（可选，若传入则优先使用，
                             可避免重复构建，提升性能）

        Returns:
            MultimodalUnderstandingResult，或 None（不可用时）
        """
        if not self._initialized:
            self._ensure_initialized()

        # 无 VLM 或无图片时不可用
        if not settings.vision_llm_enabled or self._client is None:
            return None
        if not images:
            return None

        # 解析图片（最多 1 张，避免超长 context）
        max_images = settings.vlm_max_images
        selected_images = images[:max_images]

        image_bytes_list: List[bytes] = []
        for img_data in selected_images:
            try:
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                image_bytes = base64.b64decode(img_data)
                image_bytes_list.append(image_bytes)
            except Exception as e:
                logger.warning(f"图片 Base64 解析失败: {e}")
                continue

        if not image_bytes_list:
            return None

        # 若外部已传入 history_context 直接使用；否则自行构建
        ctx: Optional[Dict[str, Any]] = history_context
        if ctx is None and conversation_history:
            ctx = self._build_history_context(conversation_history)

        try:
            user_prompt = _build_user_prompt(question, ctx)
            image_content = _build_image_content(image_bytes_list)

            response = self._client.chat.completions.create(
                model=settings.vision_llm_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            *image_content,
                        ],
                    },
                ],
                max_tokens=settings.vlm_max_tokens,
                temperature=settings.vlm_temperature,
            )

            raw_text = response.choices[0].message.content.strip()
            if settings.vlm_debug_log:
                logger.debug(f"VLM raw output: {raw_text[:200]}")
            else:
                logger.debug(f"VLM raw output: {raw_text[:100]}...")

        except Exception as e:
            logger.warning(f"VLM API 调用失败: {e}")
            return None

        # 解析 JSON
        payload = _extract_json_block(raw_text)
        if payload is None:
            logger.warning(f"VLM 输出无法解析为 JSON: {raw_text[:100]}")
            return None

        # 规范化
        normalized = _normalize_vlm_payload(payload)

        # 校验
        if not _validate_result(normalized):
            logger.warning(f"VLM 输出字段校验失败，使用规范化后结果: {normalized}")
            # 不完全失败，用校验后的结果继续

        return MultimodalUnderstandingResult(
            normalized_query=normalized["normalized_query"],
            language=normalized["language"],
            product_candidates=normalized["product_candidates"],
            image_tags=normalized["image_tags"],
            visual_intents=normalized["visual_intents"],
            evidence_type=normalized["evidence_type"],
            referenced_previous_object=normalized["referenced_previous_object"],
            requires_history_resolution=normalized["requires_history_resolution"],
            confidence=normalized["confidence"],
            notes={
                "vlm_source": True,
                "image_count": len(image_bytes_list),
                "raw_evidence_type": payload.get("evidence_type"),
            },
        )

    def _build_history_context(
        self,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从对话历史构建摘要。"""
        from src.utils.domain_knowledge import match_product_alias

        recent = history[-3:] if len(history) >= 3 else history
        parts: List[str] = []
        had_images = False
        recent_product: Optional[str] = None

        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            if msg.get("images"):
                had_images = True
            if content:
                snippet = content[:100] + "..." if len(content) > 100 else content
                parts.append(f"{role}说：{snippet}")
            if role == "助手" and not recent_product:
                candidates = match_product_alias(content)
                if candidates:
                    recent_product = candidates[0]

        return {
            "summary": "；".join(parts),
            "recent_product": recent_product,
            "had_images": had_images,
        }
