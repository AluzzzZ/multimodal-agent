"""
多模态理解模块
负责解析用户输入的文本和图片，识别用户意图
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger

from config import settings
from src.modules.understanding_types import MultimodalUnderstandingResult, ProductCandidate
from src.utils.domain_knowledge import (
    ALIAS_TO_PRODUCT,
    match_product_alias,
    match_caption_keywords,
    PART_KEYWORDS,
    STATE_KEYWORDS,
    DIAGRAM_KEYWORDS,
    ROUTE_SERVICE_HINTS,
    ROUTE_MANUAL_HINTS,
    TAG_STOP_WORDS,
    TAG_PRIORITY,
    VISUAL_INTENT_MAP,
    CAPTION_PRODUCT_TYPE_MAP,
)
from src.modules.vlm_understanding import VLMUnderstandingEngine


# ---------------------------------------------------------------------------
# 模块级 SentenceTransformer 单例（避免每次调用重复加载模型）
# ---------------------------------------------------------------------------
_sem_encoder = None
_product_desc_map: Dict[str, str] = {}
_product_desc_embs: Optional[np.ndarray] = None


def _ensure_sem_encoder() -> Optional[Any]:
    """延迟初始化语义编码器，单例模式。"""
    global _sem_encoder, _product_desc_map, _product_desc_embs
    if _sem_encoder is not None:
        return _sem_encoder
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    try:
        device = settings.embedding_device
        _sem_encoder = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device=device,
        )
        _product_desc_map = {
            "VR头显": "VR头显 虚拟现实 头戴显示器",
            "遥控器": "遥控器 空调遥控 红外遥控",
            "健身器材": "健身器材 动感单车 跑步机 健身设备",
            "健身追踪器": "健身追踪器 智能手环 运动手环",
            "智能手表": "智能手表 电子手表 手环",
            "空调": "空调 室内机 室外机 冷气机",
            "空气净化器": "空气净化器 净化器 空气过滤器",
            "键盘": "键盘 机械键盘 功能键盘",
            "鼠标": "鼠标 蓝牙鼠标 无线鼠标",
            "路由器": "路由器 WiFi路由器 网络设备",
            "相机": "相机 快门 镜头",
            "投影仪": "投影仪 投影机 投影",
            "耳机": "耳机 蓝牙耳机 头戴耳机",
            "电钻": "电钻 电动工具 手电钻",
            "冰箱": "冰箱 冷藏柜 冷柜",
            "烤箱": "烤箱 微波炉 空气炸锅",
            "洗碗机": "洗碗机",
            "洗衣机": "洗衣机",
            "吸尘器": "吸尘器 扫地机 清洁机",
            "发电机": "发电机 引擎 电机",
            "水泵": "水泵 抽水机",
            "咖啡机": "咖啡机 意式咖啡机",
            "吹风机": "吹风机",
            "电暖器": "电暖器 暖气机 电热器",
            "可编程温控器": "温控器 恒温器 智能温控",
            "摩托艇": "摩托艇 船 汽艇",
            "儿童电动摩托车": "儿童摩托车 儿童电动车",
            "蓝牙激光鼠标": "蓝牙鼠标 无线鼠标",
            "蒸汽清洁机": "蒸汽清洁机 蒸汽拖把",
            "人体工学椅": "人体工学椅 办公椅",
            "游戏手柄": "游戏手柄 手柄 游戏控制器",
        }
        _product_desc_embs = _sem_encoder.encode(
            list(_product_desc_map.values()), normalize_embeddings=True
        )
        logger.info("语义编码器 paraphrase-multilingual-MiniLM-L12-v2 加载完成")
        return _sem_encoder
    except Exception:
        _sem_encoder = None
        return None


class TextParser:
    """文本解析器"""
    
    def __init__(self):
        self.llm_client = None  # 延迟初始化
        self._initialized = False
    
    def initialize(self):
        """初始化LLM客户端（延迟加载，仅在 parse() 等旧方法实际调用时才加载）"""
        if self._initialized:
            return
        # LLM 客户端延迟到 parse() 首次调用时加载，不在此处触发 langchain 导入
        self._initialized = True
        
    def _ensure_llm(self):
        """延迟初始化 LLM 客户端，避免在 initialize() 时触发 langchain 导入"""
        if self.llm_client is not None:
            return
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
            logger.info(f"TextParser LLM客户端初始化成功: {settings.llm_provider}")
        except Exception as e:
            logger.warning(f"TextParser LLM客户端初始化失败: {e}")
            self.llm_client = None
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        解析文本输入
        
        Args:
            text: 用户输入的文本
            
        Returns:
            解析结果，包含意图、关键实体等
        """
        if not text or not text.strip():
            return {
                "original_text": "",
                "intent": None,
                "entities": [],
                "questions": [],
                "requires_images": False,
                "confidence": 0.0
            }
        
        # 意图识别提示
        intent_prompt = f"""分析以下用户问题，识别用户意图：

用户问题：{text}

请输出JSON格式的分析结果：
{{
    "intent": "意图类别（product_inquiry/maintenance/shipping/payment/complaint/other）",
    "entities": ["关键实体列表"],
    "questions": ["拆分后的子问题列表"],
    "requires_images": 是否需要图片来回答,
    "confidence": 置信度(0-1)
}}
"""
        
        try:
            self._ensure_llm()
            if self.llm_client is None:
                return {
                    "original_text": text,
                    "intent": "unknown",
                    "entities": [],
                    "questions": [text],
                    "requires_images": False,
                    "confidence": 0.3,
                    "_fallback": True,
                }
            
            response = self.llm_client.invoke(intent_prompt)
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # 简单解析JSON（实际应用中应使用更 robust 的解析方式）
            import json
            import re
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "original_text": text,
                    **parsed
                }
            
            return {
                "original_text": text,
                "intent": "unknown",
                "entities": [],
                "questions": [text],
                "requires_images": False,
                "confidence": 0.5
            }
            
        except Exception as e:
            logger.error(f"文本解析失败: {e}")
            return {
                "original_text": text,
                "intent": "unknown",
                "entities": [],
                "questions": [text],
                "requires_images": False,
                "confidence": 0.0
            }
    
    def split_questions(self, text: str) -> List[str]:
        """拆分复杂问题为多个子问题"""
        self._ensure_llm()
        if self.llm_client is None:
            return [text]
        
        split_prompt = f"""将以下复杂问题拆分为多个简单问题，每个问题应该可以独立回答：

问题：{text}

请列出所有子问题，每行一个：
"""
        
        try:
            response = self.llm_client.invoke(split_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            return questions if questions else [text]
        except Exception as e:
            logger.error(f"问题拆分失败: {e}")
            return [text]


class ImageParser:
    """图片解析器"""

    MOCK_CAPTION_PREFIX = "__mock_caption__:"
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.torch = None
        self._caption_model = None
        self._caption_processor = None
        self._initialized = False
    
    def initialize(self):
        """初始化视觉理解模块

        注意：当启用视觉大模型 API (VISION_LLM_ENABLED=true) 时，
        不需要加载本地 CLIP 模型，所有图片分析通过外部 API 完成。
        """
        if self._initialized:
            return
        
        # 加载 caption 模型
        self._load_caption_model()
        
        self._initialized = True
        logger.info("多模态理解模块初始化完成")
    
    def parse_image(self, image_data: str) -> Dict[str, Any]:
        """
        解析图片输入，获取图片信息和描述。

        当启用视觉大模型 API 时，使用 VLM 生成描述。
        
        Args:
            image_data: Base64编码的图片数据，或评测专用的 mock 格式
            
        Returns:
            图片解析结果
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # 解码Base64图片
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            raw_image = Image.open(io.BytesIO(image_bytes))
            image_format = raw_image.format or "unknown"
            image = raw_image.convert('RGB')

            # 使用视觉大模型生成描述
            description = self.describe_image(image_data)
            
            return {
                "format": image_format,
                "size": image.size,
                "mode": image.mode,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"图片解析失败: {e}")
            return {
                "error": str(e),
                "description": "图片解析失败"
            }
    
    def _extract_mock_caption_text(self, image_data: str) -> Optional[str]:
        """解析评测专用的 mock caption 输入。"""
        if not image_data or not image_data.startswith(self.MOCK_CAPTION_PREFIX):
            return None
        return image_data[len(self.MOCK_CAPTION_PREFIX):]

    def _load_caption_model(self):
        """选择图片描述策略。当启用视觉大模型 API 时，使用 VLM；否则使用规则 fallback。"""
        if settings.vision_llm_enabled:
            self._caption_model = "vision_llm"
            logger.info(f"图片描述使用视觉大模型 API: {settings.vision_llm_model}")
        else:
            self._caption_model = "rule_fallback"
            logger.info("图片描述使用规则 fallback（无视觉模型）")
    
    def describe_image(self, image_data: str) -> str:
        """
        生成图片描述。

        当启用视觉大模型 API 时，通过 VLM 分析图片；
        否则使用规则 fallback。
        
        Args:
            image_data: Base64编码的图片数据，或评测专用的 mock 格式
            
        Returns:
            图片的自然语言描述
        """
        if not self._initialized:
            self.initialize()
        
        # 评测专用：mock caption 直接返回
        mock_caption = self._extract_mock_caption_text(image_data)
        if mock_caption is not None:
            logger.debug(f"Mock caption: {mock_caption[:100]}...")
            return mock_caption
        
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 策略: 视觉大模型 API 或规则 fallback
            if self._caption_model == "vision_llm":
                return self._caption_via_vision_llm(image)
            else:
                return self._caption_via_rule_fallback(image)
            
        except Exception as e:
            logger.error(f"图片描述生成失败: {e}")
            return "图片描述生成失败"

    def _caption_via_vision_llm(self, image: Image.Image) -> str:
        """通过外部视觉大模型 API 生成图片描述。"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        try:
            return self._caption_via_openai_vision(img_base64)
        except Exception as e:
            logger.warning(f"视觉大模型 API 调用失败: {e}，使用规则 fallback")
            return self._caption_via_rule_fallback(image)

    def _caption_via_text_enhancement(self, text_description: str) -> str:
        """通过视觉大模型对文字描述进行结构化分析和增强（用于评测模式）。"""
        prompt = f"""你是一个产品图片分析助手。请分析以下图片描述，并输出一句话总结：

图片描述：{text_description}

要求：
1. 提取产品类型（如：电钻、遥控器、健身器材等）
2. 提取可见部件（如：指示灯、按钮、屏幕、表带等）
3. 提取状态/外观（如：指示灯闪烁、屏幕显示、破损、划痕等）

输出格式：产品类型 + 主要部件 + 状态
控制在30字以内，用中文。"""

        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.vision_llm_api_key or settings.llm_api_key,
                base_url=settings.vision_llm_base_url or settings.llm_base_url
            )
            response = client.chat.completions.create(
                model=settings.vision_llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.3,
            )
            result = response.choices[0].message.content.strip()
            logger.debug(f"Text enhancement caption: {result}")
            return result
        except Exception as e:
            logger.warning(f"文字增强失败: {e}，使用原始描述")
            return text_description

    def _caption_via_openai_vision(self, img_base64: str, api_key: str = None, base_url: str = None) -> str:
        """调用视觉大模型 API（Qwen3-VL / GPT-4V 等）"""
        from openai import OpenAI

        api_key = api_key or settings.vision_llm_api_key or settings.llm_api_key
        base_url = base_url or settings.vision_llm_base_url or settings.llm_base_url

        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=settings.vision_llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请分析这张图片，用一句话描述内容，包括产品类型、可见部件、状态。回答控制在50字以内，用中文。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    ],
                }
            ],
            max_tokens=100,
            temperature=0.3,
        )
        caption = response.choices[0].message.content.strip()
        logger.debug(f"Vision caption: {caption}")
        return caption

    def _caption_via_rule_fallback(self, image: Image.Image) -> str:
        """无视觉模型时的规则兜底：仅根据图像尺寸推断描述。"""
        if image is None:
            return "图片已接收"

        width, height = image.size
        aspect = width / height if height > 0 else 1.0

        if aspect > 1.5:
            size_desc = "横向长图"
        elif aspect < 0.7:
            size_desc = "纵向长图"
        else:
            size_desc = "方形图片"

        return f"{size_desc}，{width}x{height}像素"


class MultimodalUnderstanding:
    """多模态理解主类"""

    _HYBRID_STAGES = ("observe", "product_candidates", "image_tags", "evidence_type", "full")
    
    def __init__(self):
        self.text_parser = TextParser()
        self.image_parser = ImageParser()
        self._vlm_engine: Optional[VLMUnderstandingEngine] = None
        self._text_initialized = False
        self._image_initialized = False
        self._initialized = False
    
    def _ensure_text_initialized(self):
        """按需初始化文本侧组件。"""
        if self._text_initialized:
            return
        self.text_parser.initialize()
        self._text_initialized = True

    def _ensure_image_initialized(self):
        """按需初始化图片侧组件。"""
        if self._image_initialized:
            return
        self.image_parser.initialize()
        self._image_initialized = True

    def _ensure_vlm_initialized(self):
        """按需初始化 VLM 引擎。"""
        if self._vlm_engine is not None:
            return
        self._vlm_engine = VLMUnderstandingEngine()

    def _get_hybrid_stage(self) -> str:
        """获取合法的 hybrid 灰度阶段配置。"""
        stage = str(getattr(settings, "multimodal_hybrid_stage", "observe") or "observe").strip().lower()
        if stage not in self._HYBRID_STAGES:
            logger.warning(
                f"未知 multimodal_hybrid_stage={stage}，可用值: {self._HYBRID_STAGES}，回退为 observe"
            )
            return "observe"
        return stage

    def initialize(self, include_image: bool = False):
        """初始化多模态理解组件，默认仅初始化文本侧。"""
        self._ensure_text_initialized()
        if include_image:
            self._ensure_image_initialized()
            if settings.multimodal_backend in {"vlm", "hybrid"}:
                self._ensure_vlm_initialized()
        self._initialized = self._text_initialized and (self._image_initialized if include_image else True)
        logger.info(
            "多模态理解模块初始化完成"
            + ("（含图片侧）" if include_image else "（文本侧）")
        )
    
    def understand(
        self,
        text: Optional[str] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        综合理解用户输入（向后兼容入口）。

        内部委托给 analyze()，将 MultimodalUnderstandingResult 适配为旧格式。
        """
        if not text and not images:
            return {
            "text_analysis": None,
            "image_analysis": [],
            "combined_intent": None,
            "requires_multimodal": False,
            "confidence": 0.0
        }
        
        mm_result = self.analyze(
            question=text or "",
            images=images,
            conversation_history=None,
        )

        return {
            "text_analysis": {
                "intent": mm_result.evidence_type,
                "entities": [c.name for c in mm_result.product_candidates],
                "questions": [mm_result.normalized_query] if mm_result.normalized_query else [],
                "requires_images": bool(mm_result.image_tags),
                "confidence": mm_result.confidence,
            },
            "image_analysis": [
                {"description": tag} for tag in mm_result.image_tags
            ],
            "combined_intent": {
                "primary_intent": mm_result.evidence_type,
                "entities": [c.name for c in mm_result.product_candidates],
                "questions": [mm_result.normalized_query] if mm_result.normalized_query else [],
                "requires_images": bool(mm_result.image_tags),
                "confidence": mm_result.confidence,
            },
            "requires_multimodal": bool(mm_result.image_tags),
            "confidence": mm_result.confidence,
        }
    
    def analyze(
        self,
        question: str,
        images: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> MultimodalUnderstandingResult:
        """
        结构化多模态理解主入口（编排层）。

        根据 multimodal_backend 配置选择后端：
        - rule:    只走规则链（原有逻辑）
        - vlm:     只走 VLM 结构化输出
        - hybrid:  VLM 优先，失败或不可用时回退规则链

        融合策略（hybrid 模式）：
        - product_candidates: VLM 为主，rule 补全
        - image_tags:         VLM 为主，与 rule tags 去重合并
        - visual_intents:     VLM 为主，rule 补全
        - evidence_type:      VLM 为主，rule 纠错
        - history reference:  VLM 为主
        - confidence:         VLM 原值，fallback 时降权
        """
        if not question or not question.strip():
            return MultimodalUnderstandingResult()

        backend = settings.multimodal_backend

        # --- vlm-only ---
        if backend == "vlm":
            return self._analyze_via_vlm(question, images, conversation_history)

        # --- rule-only ---
        if backend == "rule":
            return self._analyze_via_rule(question, images, conversation_history)

        # --- hybrid（默认）---
        return self._analyze_hybrid(question, images, conversation_history)

    # -------------------------------------------------------------------------
    # VLM 路径
    # -------------------------------------------------------------------------

    def _analyze_via_vlm(
        self,
        question: str,
        images: Optional[List[str]],
        history: Optional[List[Dict[str, Any]]],
    ) -> MultimodalUnderstandingResult:
        """只用 VLM 结构化输出；不可用时回退规则链。"""
        self._ensure_vlm_initialized()
        history_context = self._parse_conversation_context(history)
        vlm_result = self._vlm_engine.understand(question, images, history, history_context)
        if vlm_result is not None:
            notes = dict(vlm_result.notes) if vlm_result.notes else {}
            notes["multimodal_backend"] = "vlm"
            notes["hybrid_stage"] = "full"
            vlm_result.notes = notes
            return vlm_result
        rule_result = self._analyze_via_rule(question, images, history)
        notes = dict(rule_result.notes) if rule_result.notes else {}
        notes["multimodal_backend"] = "vlm"
        notes["vlm_fallback"] = True
        notes["reason"] = "vlm_unavailable"
        rule_result.notes = notes
        return rule_result

    # -------------------------------------------------------------------------
    # 规则路径（原有逻辑迁移）
    # -------------------------------------------------------------------------

    def _analyze_via_rule(
        self,
        question: str,
        images: Optional[List[str]],
        history: Optional[List[Dict[str, Any]]],
    ) -> MultimodalUnderstandingResult:
        """纯规则链理解。迁移自原 analyze() 逻辑。"""
        self._ensure_text_initialized()
        if images:
            self._ensure_image_initialized()

        history_context = self._parse_conversation_context(history)

        enriched_question = question
        if history_context:
            enriched_question = f"{history_context['summary']} {question}"

        from src.utils.text_utils import QueryProcessor
        norm_result = QueryProcessor.normalize_query_for_retrieval(enriched_question)
        normalized = norm_result["normalized_query"]
        lang = norm_result["language"]

        image_tags = self._extract_image_tags(images, question=question)

        ref_obj, needs_history = self._resolve_history_reference(
            question, history_context, image_tags
        )
        if ref_obj:
            logger.debug(f"检测到多轮引用: '{ref_obj}' <- 问题: {question}")

        product_candidates = self._infer_product_candidates(
            question, image_tags, history_context=history_context
        )
        visual_intents = self._infer_visual_intents(question, image_tags)
        evidence_scores = self._score_evidence_type(
            question,
            image_tags,
            product_candidates,
            visual_intents=visual_intents,
            history_context=history_context,
        )
        evidence_type = self._infer_evidence_type(
            question,
            image_tags,
            product_candidates,
            visual_intents=visual_intents,
            history_context=history_context,
            evidence_scores=evidence_scores,
        )

        return MultimodalUnderstandingResult(
            normalized_query=normalized,
            language=lang,
            image_tags=image_tags,
            product_candidates=product_candidates,
            visual_intents=visual_intents,
            evidence_type=evidence_type,
            confidence=0.8 if image_tags else 0.6,
            notes={
                "images_provided": len(images) if images else 0,
                "history_turns_used": history_context.get("turns_used", 0) if history_context else 0,
                "evidence_scoring": evidence_scores,
            },
            referenced_previous_object=ref_obj,
            requires_history_resolution=needs_history,
        )

    # -------------------------------------------------------------------------
    # Hybrid 融合路径
    # -------------------------------------------------------------------------

    def _analyze_hybrid(
        self,
        question: str,
        images: Optional[List[str]],
        history: Optional[List[Dict[str, Any]]],
    ) -> MultimodalUnderstandingResult:
        """
        Hybrid 融合：VLM 为主，规则补全。

        流程：
        1. 先走 VLM（最快路径）
        2. VLM 成功 → 用 VLM 结果 + 规则补全
        3. VLM 失败 → 回退规则链
        """
        self._ensure_vlm_initialized()
        history_context = self._parse_conversation_context(history)
        vlm_result = self._vlm_engine.understand(question, images, history, history_context)
        stage = self._get_hybrid_stage()

        if vlm_result is not None:
            return self._merge_with_rule(vlm_result, question, images, history, stage=stage)
        else:
            rule_result = self._analyze_via_rule(question, images, history)
            notes = dict(rule_result.notes) if rule_result.notes else {}
            notes["multimodal_backend"] = "hybrid"
            notes["hybrid_stage"] = stage
            notes["vlm_fallback"] = True
            notes["reason"] = "vlm_unavailable"
            rule_result.notes = notes
            return rule_result

    def _merge_with_rule(
        self,
        vlm_result: MultimodalUnderstandingResult,
        question: str,
        images: Optional[List[str]],
        history: Optional[List[Dict[str, Any]]],
        stage: str,
    ) -> MultimodalUnderstandingResult:
        """
        用规则结果增强 VLM 结果。

        融合规则：
        - normalized_query: VLM 优先，规则纠错（如空值）
        - language: VLM 优先
        - product_candidates: VLM + 规则去重合并（规则兜底补全）
        - image_tags: VLM + 规则去重合并
        - visual_intents: VLM 为主，规则补全（VLM 为空时用规则）
        - evidence_type: VLM 为主，规则纠错（VLM 输出不合枚举时用规则）
        - history reference: VLM 优先
        - confidence: VLM 原值，规则 override 则降权
        """
        rule_result = self._analyze_via_rule(question, images, history)

        # product_candidates: 按名称去重，保留最高分，按分数降序
        all_candidates = vlm_result.product_candidates + rule_result.product_candidates
        best_by_name: Dict[str, ProductCandidate] = {}
        for c in all_candidates:
            existing = best_by_name.get(c.name)
            if existing is None or c.score > existing.score:
                best_by_name[c.name] = c
        merged_products = sorted(best_by_name.values(), key=lambda c: c.score, reverse=True)[:5]

        # image_tags: 合并去重
        merged_tags = list(dict.fromkeys(
            vlm_result.image_tags + rule_result.image_tags
        ))[:12]

        # visual_intents: VLM 为主，规则补全
        merged_intents = list(dict.fromkeys(
            vlm_result.visual_intents + rule_result.visual_intents
        ))[:5]

        # evidence_type: 枚举校验，违规时用规则结果
        valid_evidence = {"service_like", "manual_like", "mixed_like", "unknown"}
        if vlm_result.evidence_type not in valid_evidence:
            merged_evidence = rule_result.evidence_type
        else:
            merged_evidence = vlm_result.evidence_type

        # confidence: VLM 为主，规则 override 则降权
        # 如果规则认为 evidence_type 不同（关键字段有分歧），降低置信度
        if vlm_result.evidence_type != rule_result.evidence_type:
            confidence = min(vlm_result.confidence, rule_result.confidence, 0.7)
        else:
            confidence = vlm_result.confidence

        if stage == "observe":
            notes = dict(rule_result.notes) if rule_result.notes else {}
            notes.update({
                "multimodal_backend": "hybrid",
                "hybrid_stage": stage,
                "vlm_observed": True,
                "vlm_product_candidates": vlm_result.product_candidates,
                "vlm_image_tags": vlm_result.image_tags,
                "vlm_visual_intents": vlm_result.visual_intents,
                "vlm_evidence_type": vlm_result.evidence_type,
                "vlm_confidence": vlm_result.confidence,
            })
            rule_result.notes = notes
            logger.debug(
                "Hybrid observe: rule(products={}, evidence={}) vs vlm(products={}, evidence={})",
                rule_result.product_candidates,
                rule_result.evidence_type,
                vlm_result.product_candidates,
                vlm_result.evidence_type,
            )
            return rule_result

        notes = dict(vlm_result.notes) if vlm_result.notes else {}
        notes.update({
            "multimodal_backend": "hybrid",
            "hybrid_stage": stage,
            "rule_merged": True,
            "rule_evidence_type": rule_result.evidence_type,
            "images_provided": len(images) if images else 0,
        })

        final_result = MultimodalUnderstandingResult(
            normalized_query=rule_result.normalized_query,
            language=rule_result.language,
            product_candidates=rule_result.product_candidates,
            image_tags=rule_result.image_tags,
            visual_intents=rule_result.visual_intents,
            evidence_type=rule_result.evidence_type,
            confidence=rule_result.confidence,
            notes=notes,
            referenced_previous_object=rule_result.referenced_previous_object,
            requires_history_resolution=rule_result.requires_history_resolution,
        )

        if stage == "product_candidates" or stage == "image_tags" or stage == "evidence_type" or stage == "full":
            final_result.product_candidates = merged_products

        if stage == "image_tags" or stage == "evidence_type" or stage == "full":
            final_result.image_tags = merged_tags
            final_result.visual_intents = merged_intents

        if stage == "evidence_type" or stage == "full":
            final_result.evidence_type = merged_evidence
            final_result.referenced_previous_object = (
                vlm_result.referenced_previous_object or rule_result.referenced_previous_object
            )
            final_result.requires_history_resolution = (
                vlm_result.requires_history_resolution or rule_result.requires_history_resolution
            )

        if stage == "full":
            final_result.normalized_query = vlm_result.normalized_query or rule_result.normalized_query
            final_result.language = vlm_result.language or rule_result.language
            final_result.confidence = confidence
        elif stage != "observe" and vlm_result.confidence > final_result.confidence:
            final_result.confidence = min(vlm_result.confidence, 0.78)

        return final_result

    # -------------------------------------------------------------------------
    # 多轮上下文处理
    # -------------------------------------------------------------------------

    def _parse_conversation_context(
        self,
        history: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """
        从对话历史中提取最近 2-3 轮的关键信息，用于增强多模态理解。

        提取内容：
        - 最近用户问题摘要（帮助解析"刚才那个"、"上面那张图"等指代）
        - 最近图片是否出现（影响本轮图片的理解方向）
        - 最近回答中提及的产品/部件（帮助缩小候选范围）

        Returns:
            包含 summary / recent_product / had_images 的字典，无历史时返回 None
        """
        if not history:
            return None

        # 取最近 3 轮（user / assistant / user...）
        recent = history[-3:] if len(history) >= 3 else history
        parts: List[str] = []
        recent_product: Optional[str] = None
        had_images = False

        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            if msg.get("images"):
                had_images = True
            if content:
                # 截断避免上下文过长
                snippet = content[:120] + "..." if len(content) > 120 else content
                parts.append(f"{role}说：{snippet}")
            # 从最近助手回答中尝试抽取产品名
            if role == "助手" and not recent_product:
                candidates = self._infer_product_candidates(content, [])
                if candidates:
                    recent_product = candidates[0].name

        summary = "；".join(parts)
        return {
            "summary": summary,
            "recent_product": recent_product,
            "had_images": had_images,
            "turns_used": len(recent),
        }

    def _resolve_history_reference(
        self,
        question: str,
        history_context: Optional[Dict[str, Any]],
        current_image_tags: List[str],
    ) -> Tuple[Optional[str], bool]:
        """
        检测当前问题是否引用了历史中的对象（如"那个按钮"、"上面那张图"）。

        Returns:
            (引用的对象描述, 是否需要历史才能准确理解)
        """
        if not question or not question.strip():
            return None, False

        # 检测多轮引用关键词
        history_ref_patterns = [
            r"刚才", r"上面", r"上面那张", r"上面那个", r"那张图",
            r"那个按钮", r"那个灯", r"这个按钮", r"这个灯",
            r"之前的", r"之前那张", r"之前那个", r"前述",
            r"第二张", r"第二个", r"第三个", r"第.*张",
        ]
        has_ref = any(re.search(p, question) for p in history_ref_patterns)

        if not has_ref:
            return None, False

        # 有关键词但无历史上下文，无法解析
        if not history_context:
            return "（历史上下文不足，无法确定引用对象）", True

        # 从历史摘要中推断引用的是什么
        summary = history_context.get("summary", "")

        # 尝试从当前问题的上下文推断引用对象
        ref_target = None
        # 按钮类引用
        if re.search(r"按钮|按键", question):
            ref_target = "指示灯/按钮区域"
        elif re.search(r"灯|指示", question):
            ref_target = "指示灯"
        elif re.search(r"图|照片|图片", question):
            ref_target = "图片内容"
        elif re.search(r"表带|尺寸", question):
            ref_target = "表带/外观"
        elif re.search(r"包装|破损", question):
            ref_target = "包装/外观"

        # 若无法明确推断但确实有引用，给出通用提示
        if not ref_target:
            ref_target = "前序对话中提到的对象"

        return ref_target, True

    # -------------------------------------------------------------------------
    # 私有方法
    # -------------------------------------------------------------------------

    def _normalize_query(self, question: str) -> str:
        """归一化查询文本：委托 QueryProcessor 处理（包含英文归一化/翻译等完整逻辑）"""
        from src.utils.text_utils import QueryProcessor
        result = QueryProcessor.normalize_query_for_retrieval(question)
        return result["normalized_query"]

    def _extract_image_tags(
        self,
        images: Optional[List[str]],
        question: str = "",
    ) -> List[str]:
        """
        提取图片稳定标签。

        采用两层结构：
        1. Caption 层：先用 describe_image() 生成短描述（VLM 或规则 fallback）
        2. 标签抽取层：从 caption 中按规则抽取产品类型 / 部件词 / 状态词 / 图示词
        """
        if not images:
            return []

        all_tags: List[str] = []
        for img in images[:3]:  # 最多处理 3 张
            try:
                # Step 1: 生成图片 caption；mock caption 直接使用，不走 VLM 增强
                mock_caption = self.image_parser._extract_mock_caption_text(img)
                if mock_caption is not None:
                    caption = mock_caption
                else:
                    caption = self.image_parser.describe_image(img)

                # Step 2: 从 caption 中按规则抽取稳定标签
                # 传入 question 以支持 caption 为通用描述时的文本回退
                tags = self._extract_tags_from_caption(caption, question=question)
                all_tags.extend(tags)

                logger.debug(f"图片标签: caption='{caption}', tags={tags}")
            except Exception as e:
                logger.warning(f"图片标签提取失败: {e}")

        # 去重并按优先级排序（产品类型词优先）
        # 当无视觉模型时，caption 是通用尺寸描述，标签提取主要依赖：
        # 1. 问题文本中的视觉意图词（如"指示灯"、"按钮"）→ 从 visual_intents 补充
        # 2. 规则标签体系（产品类型 / 部件词 / 状态词 / 图示词）
        seen: set = set()
        result: List[str] = []
        for tag in all_tags:
            if tag and tag not in seen:
                seen.add(tag)
                result.append(tag)
        return self._prioritize_and_deduplicate_tags(result)

    def _extract_tags_from_caption(
        self,
        caption: str,
        question: str = "",
    ) -> List[str]:
        """
        从图片 caption 中抽取稳定标签。

        采用三层策略：
        1. 关键词精确匹配（部件 / 状态 / 图示 / 产品类型）
        2. 语义相似度匹配（中文术语与产品词典的余弦相似度）
        3. 规则过滤（停用词 / 长度 / 位置模式）

        caption 为通用尺寸描述时 fallback 到从问题文本抽取。
        """
        generic_prefixes = ("宽幅图片", "竖幅图片", "图片（")
        generic_set = {
            "图片已接收", "图片描述生成失败",
            "无法生成描述，视觉模型未配置",
            "无法生成描述，视觉分析待配置",
        }
        is_generic = (
            not caption
            or any(caption.startswith(p) for p in generic_prefixes)
            or caption in generic_set
        )

        if is_generic:
            if question:
                return self._extract_tags_from_question(question)
            return []

        tags: List[str] = []

        # --- Step 1: 关键词精确匹配 ---
        # 产品类型
        for product, keywords in CAPTION_PRODUCT_TYPE_MAP.items():
            if any(kw in caption for kw in keywords):
                tags.append(product)
                break

        # 部件 / 状态 / 图示
        tags.extend(match_caption_keywords(caption, PART_KEYWORDS))
        tags.extend(match_caption_keywords(caption, STATE_KEYWORDS))
        tags.extend(match_caption_keywords(caption, DIAGRAM_KEYWORDS))

        # --- Step 2: 语义相似度匹配（中文术语） ---
        self._semantic_tag_augment(caption, tags)

        # --- Step 3: 规则过滤 + 去重 ---
        return self._prioritize_and_deduplicate_tags(tags)

    def _semantic_tag_augment(self, caption: str, tags: List[str]) -> None:
        """
        用语义相似度对 caption 中的中文术语做二次匹配，增强关键词匹配的召回率。

        模块级单例 _ensure_sem_encoder() 保证模型只加载一次，
        避免每次图片都重新加载 SentenceTransformer（199 层，约 30s/次）。
        """
        sem = _ensure_sem_encoder()
        if sem is None or _product_desc_embs is None:
            return

        try:
            # 提取 caption 中 2-8 字的中文术语（与原逻辑一致）
            chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", caption)
            filtered = []
            for term in chinese_terms:
                if term in TAG_STOP_WORDS or term in tags or len(term) > 8:
                    continue
                if any(re.match(p, term) for p in [
                    r".*正面", r".*侧面", r".*顶部", r".*底部",
                    r".*可见$", r".*照片$", r".*特写$",
                    r".*图示$", r".*标注$", r".*显示$",
                ]):
                    continue
                filtered.append(term)

            if not filtered:
                return

            term_embs = sem.encode(filtered, normalize_embeddings=True)
            sim_matrix = np.dot(term_embs, _product_desc_embs.T)

            for i, term in enumerate(filtered):
                best_score = float(sim_matrix[i].max())
                if best_score < 0.45:
                    continue
                best_product = list(_product_desc_map.keys())[int(np.argmax(sim_matrix[i]))]
                if best_product not in tags:
                    tags.append(best_product)
        except Exception:
            pass

    def _extract_tags_from_question(self, question: str) -> List[str]:
        """
        无视觉模型时：从问题文本中直接抽取标签。
        数据来源统一为 domain_knowledge。
        """
        if not question:
            return []

        tags: List[str] = []

        # 部件词 / 状态词 / 图示词 精确匹配
        tags.extend(match_caption_keywords(question, PART_KEYWORDS))
        tags.extend(match_caption_keywords(question, STATE_KEYWORDS))
        tags.extend(match_caption_keywords(question, DIAGRAM_KEYWORDS))

        # 产品别名匹配
        tags.extend(match_product_alias(question))

        return self._prioritize_and_deduplicate_tags(tags)

    def _prioritize_and_deduplicate_tags(self, tags: List[str]) -> List[str]:
        """
        对标签去重并按优先级排序。
        优先级数值越小越靠前，未知标签优先级为 100（最后）。
        """
        seen: set = set()
        buckets: Dict[int, List[str]] = {}

        for tag in tags:
            if not tag or tag in seen:
                continue
            seen.add(tag)
            priority = TAG_PRIORITY.get(tag, 100)
            if priority not in buckets:
                buckets[priority] = []
            buckets[priority].append(tag)

        result: List[str] = []
        for priority in sorted(buckets.keys()):
            result.extend(buckets[priority])

        return result[:12]

    def _infer_product_candidates(
        self,
        question: str,
        image_tags: List[str],
        history_context: Optional[Dict[str, Any]] = None,
    ) -> List["ProductCandidate"]:
        """
        从问题和图片标签中推断候选产品名，返回带分数和来源的候选列表。

        置信度：文本别名匹配(1.0) > 图片标签推断(0.7) > 历史上下文(0.5)。
        按分数降序、去重后最多返回 5 个。
        """
        # Step 1: 从问题中提取产品名（精确匹配，优先级最高）
        question_matched = match_product_alias(question)

        # Step 2: 从图片标签中补充（仅当问题中没有匹配时）
        image_matched: List[str] = []
        if not question_matched:
            for tag in image_tags:
                product = ALIAS_TO_PRODUCT.get(tag)
                if product and product not in image_matched:
                    image_matched.append(product)

        # Step 3: 从历史上下文中补充（兜底）
        history_product = None
        if history_context:
            recent = history_context.get("recent_product")
            if isinstance(recent, ProductCandidate):
                history_product = recent.name
            elif recent:
                history_product = str(recent)

        # 按分数降序合并（去重）
        score_map: Dict[str, float] = {}
        for name in question_matched:
            score_map[name] = max(score_map.get(name, 0), 1.0)
        for name in image_matched:
            score_map[name] = max(score_map.get(name, 0), 0.7)
        if history_product:
            score_map[history_product] = max(score_map.get(history_product, 0), 0.5)

        sorted_names = sorted(score_map.keys(), key=lambda n: score_map[n], reverse=True)
        return [
            ProductCandidate(name=n, score=score_map[n], source=self._candidate_source(n, question_matched, image_matched, history_product))
            for n in sorted_names[:5]
        ]

    def _candidate_source(self, name: str, text_matched: List[str], image_matched: List[str], history_product: Optional[str]) -> str:
        if name in text_matched:
            return "text_alias"
        if name in image_matched:
            return "image_tag"
        if name == history_product:
            return "history"
        return "text_alias"

    def _infer_visual_intents(self, question: str, image_tags: List[str]) -> List[str]:
        """从问题和图片标签中推断视觉意图（中英双语）。数据来源统一为 domain_knowledge。"""
        intents: List[str] = []
        q_lower = question.lower()

        for intent, keywords in VISUAL_INTENT_MAP.items():
            if any(kw.lower() in q_lower for kw in keywords):
                intents.append(intent)

        # 从图片标签补充（如果问题中没有明确意图）
        if not intents:
            for tag in image_tags:
                if any(kw in tag for kw in ["灯", "按钮", "图", "位置", "屏幕", "icon"]):
                    if tag not in intents:
                        intents.append(tag)

        return intents[:5]

    @staticmethod
    def _collect_keyword_hits(text: str, keywords: List[str]) -> List[str]:
        """按顺序收集命中的关键词，避免重复计数。"""
        lowered = (text or "").lower()
        hits: List[str] = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower and kw_lower in lowered and kw not in hits:
                hits.append(kw)
        return hits

    @staticmethod
    def _collect_tag_hits(tags: List[str], keywords: List[str]) -> List[str]:
        """从图片标签中收集命中的标签关键词。"""
        if not tags:
            return []

        hits: List[str] = []
        for tag in tags:
            tag_lower = str(tag).lower()
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower and kw_lower in tag_lower and kw not in hits:
                    hits.append(kw)
        return hits

    def _score_evidence_type(
        self,
        question: str,
        image_tags: List[str],
        product_candidates: List["ProductCandidate"],
        visual_intents: Optional[List[str]] = None,
        history_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        对 evidence_type 做多信号打分。

        目标：
        - service/manual 使用独立分数累积，避免只靠关键词数量比较
        - mixed 只有在两侧证据都成立时才触发，不再用“有图就 mixed”
        - 输出调试信息，方便离线评测和调参
        """
        text = (question or "").strip()
        lowered = text.lower()
        visual_intents = visual_intents or []

        strong_service_hits = self._collect_keyword_hits(
            lowered,
            [
                "退款", "退货", "换货", "发票", "物流", "运费", "到账",
                "订单", "发货", "揽收", "配送", "补发", "补寄",
                "投诉", "售后", "客服", "保修", "质保", "赔偿",
                "假货", "正品", "真假", "乡镇", "签收", "维修费用", "人为损坏",
                "上门服务", "说明书补发", "再发一份", "发一份", "电子版说明书",
            ],
        )
        weak_service_hits = self._collect_keyword_hits(
            lowered,
            [
                kw for kw in ROUTE_SERVICE_HINTS
                if kw not in strong_service_hits
            ],
        )

        strong_manual_hits = self._collect_keyword_hits(
            lowered,
            [
                "说明书", "手册", "指示灯", "按钮", "按键", "图示",
                "配图", "操作面板", "遥控器", "滤网", "表带",
                "故障代码", "报码", "闪烁", "充电", "安装",
                "清洁", "保养", "更换", "设置", "步骤", "模式",
                "怎么用", "如何使用", "滤芯", "水质", "调温度",
                "温度", "制冷", "不制冷", "不亮", "怎么调",
                "充不进电", "不开机", "没反应", "无反应", "不工作",
            ],
        )
        weak_manual_hits = self._collect_keyword_hits(
            lowered,
            [
                kw for kw in ROUTE_MANUAL_HINTS
                if kw not in strong_manual_hits
            ] + ["怎么", "如何", "在哪", "哪里", "哪个", "位置", "操作"],
        )

        manual_tag_hits = self._collect_tag_hits(
            image_tags,
            [
                "指示灯", "按钮", "按键", "控制面板", "操作面板",
                "遥控器", "表带", "屏幕", "充电器", "滤网",
                "说明书", "图示", "示意图", "型号", "铭牌",
                "摇杆",
            ],
        )
        contextual_manual_tag_hits = self._collect_tag_hits(
            image_tags,
            ["标签", "面板", "配件"],
        )
        service_tag_hits = self._collect_tag_hits(
            image_tags,
            [
                "包装破损", "破损", "物流", "面单", "订单", "发票",
                "维修单", "售后", "快递", "污渍", "拆封",
            ],
        )
        contextual_service_tag_hits = self._collect_tag_hits(
            image_tags,
            ["包装", "包装盒", "标签", "品牌标识", "配件"],
        )

        manual_visual_hits = [
            intent for intent in visual_intents
            if intent in {
                "查看位置", "查看指示灯", "查看按钮", "查看说明书",
                "查看图示", "查看型号", "查看屏幕", "查看警告",
                "查看表带", "查看摇杆",
            }
        ]
        service_visual_hits = [
            intent for intent in visual_intents
            if intent in {
                "查看物流凭证", "查看订单信息", "查看损坏情况", "查看维修单据",
            }
        ]

        service_score = 0.0
        manual_score = 0.0
        mixed_score = 0.0

        has_doc_delivery_service = (
            ("说明书" in text or "手册" in text)
            and any(token in text for token in ("再发", "补发", "发一份", "发给我", "发我", "电子版", "再给我"))
        )
        has_manual_symptom_phrase = any(
            token in text for token in (
                "灯一直闪", "灯闪", "不亮", "怎么调温度", "调温度",
                "滤芯多久换", "滤网怎么清洗", "充不进电", "不开机",
                "没反应", "无反应", "不工作",
            )
        )
        has_usability_friction = any(
            token in text for token in ("找不到", "没找到", "对不上", "没有这个", "看不懂", "不一致")
        )

        service_score += min(0.56, len(strong_service_hits) * 0.28)
        service_score += min(0.24, len(weak_service_hits) * 0.06)

        manual_score += min(0.56, len(strong_manual_hits) * 0.28)
        manual_score += min(0.24, len(weak_manual_hits) * 0.06)

        service_score += min(0.18, len(service_tag_hits) * 0.09)
        manual_score += min(0.18, len(manual_tag_hits) * 0.09)
        if strong_service_hits or weak_service_hits or has_doc_delivery_service or has_usability_friction:
            service_score += min(0.12, len(contextual_service_tag_hits) * 0.06)
        if (
            not strong_service_hits
            and not has_doc_delivery_service
            and (strong_manual_hits or weak_manual_hits or manual_visual_hits or product_candidates or has_manual_symptom_phrase)
        ):
            manual_score += min(0.10, len(contextual_manual_tag_hits) * 0.05)

        service_score += min(0.12, len(service_visual_hits) * 0.12)
        manual_score += min(0.20, len(manual_visual_hits) * 0.10)

        if product_candidates:
            manual_score += 0.18

        if has_doc_delivery_service:
            service_score += 0.24
            manual_score = max(0.0, manual_score - 0.08)

        if has_manual_symptom_phrase and (manual_tag_hits or manual_visual_hits or product_candidates):
            manual_score += 0.12

        if has_usability_friction and (strong_manual_hits or weak_manual_hits or manual_tag_hits):
            service_score += 0.14
            mixed_score += 0.10

        has_history_ref = bool(re.search(r"(刚才|上面|之前|那个|这个|第二个|第三个)", text))
        if history_context:
            recent_product = history_context.get("recent_product")
            summary = str(history_context.get("summary", "") or "")
            if recent_product and (has_history_ref or not strong_service_hits) and not has_doc_delivery_service:
                manual_score += 0.08
            if has_history_ref and self._collect_keyword_hits(summary, ROUTE_SERVICE_HINTS):
                service_score += 0.06

        has_mixed_connector = any(
            token in text for token in ("顺便", "另外", "同时", "以及", "还想问", "还能", "再问", "先", "再", "然后", "一边", "同时想问")
        )

        segments = [
            segment.strip()
            for segment in re.split(r"[\n；;。！？?!]+", text)
            if segment.strip()
        ]
        has_service_segment = False
        has_manual_segment = False
        for segment in segments:
            seg_service = bool(self._collect_keyword_hits(segment, ROUTE_SERVICE_HINTS))
            seg_manual = bool(
                self._collect_keyword_hits(segment, ROUTE_MANUAL_HINTS)
                or self._collect_keyword_hits(segment, ["滤芯", "滤网", "调温度", "温度", "按钮", "指示灯", "说明书", "步骤", "清洗"])
            )
            has_service_segment = has_service_segment or seg_service
            has_manual_segment = has_manual_segment or seg_manual

        if has_mixed_connector and service_score >= 0.22 and manual_score >= 0.22:
            mixed_score += 0.24

        manual_side_present = bool(has_manual_segment or manual_tag_hits or manual_visual_hits or has_manual_symptom_phrase)
        service_side_present = bool(has_service_segment or strong_service_hits or weak_service_hits or service_tag_hits or has_doc_delivery_service or has_usability_friction)
        if service_side_present and manual_side_present and (len(segments) > 1 or has_mixed_connector):
            mixed_score += 0.24

        service_score = min(service_score, 1.0)
        manual_score = min(manual_score, 1.0)
        mixed_score = min(mixed_score, 1.0)

        return {
            "service_score": round(service_score, 4),
            "manual_score": round(manual_score, 4),
            "mixed_score": round(mixed_score, 4),
            "debug": {
                "strong_service_hits": strong_service_hits,
                "weak_service_hits": weak_service_hits,
                "strong_manual_hits": strong_manual_hits,
                "weak_manual_hits": weak_manual_hits,
                "service_tag_hits": service_tag_hits,
                "contextual_service_tag_hits": contextual_service_tag_hits,
                "manual_tag_hits": manual_tag_hits,
                "contextual_manual_tag_hits": contextual_manual_tag_hits,
                "service_visual_hits": service_visual_hits,
                "manual_visual_hits": manual_visual_hits,
                "has_product_candidates": bool(product_candidates),
                "has_history_reference": has_history_ref,
                "has_mixed_connector": has_mixed_connector,
                "has_service_segment": has_service_segment,
                "has_manual_segment": has_manual_segment,
                "has_doc_delivery_service": has_doc_delivery_service,
                "has_manual_symptom_phrase": has_manual_symptom_phrase,
                "has_usability_friction": has_usability_friction,
            },
        }

    def _infer_evidence_type(
        self,
        question: str,
        image_tags: List[str],
        product_candidates: List["ProductCandidate"],
        visual_intents: Optional[List[str]] = None,
        history_context: Optional[Dict[str, Any]] = None,
        evidence_scores: Optional[Dict[str, Any]] = None,
    ) -> str:
        """根据打分结果判断问题最可能需要哪种证据来源。"""
        scored = evidence_scores or self._score_evidence_type(
            question,
            image_tags,
            product_candidates,
            visual_intents=visual_intents,
            history_context=history_context,
        )

        service_score = float(scored.get("service_score", 0.0))
        manual_score = float(scored.get("manual_score", 0.0))
        mixed_score = float(scored.get("mixed_score", 0.0))

        if service_score < 0.2 and manual_score < 0.2:
            return "unknown"

        if service_score >= 0.42 and manual_score >= 0.42 and abs(service_score - manual_score) <= 0.18:
            return "mixed_like"

        if mixed_score >= 0.2 and service_score >= 0.26 and manual_score >= 0.26:
            return "mixed_like"

        if service_score >= manual_score + 0.12:
            return "service_like"

        if manual_score >= service_score + 0.12:
            return "manual_like"

        if manual_score >= 0.28 and product_candidates:
            return "manual_like"

        if service_score >= 0.28:
            return "service_like"

        if manual_score >= 0.28:
            return "manual_like"

        return "unknown"


# 全局实例
_multimodal_understanding: Optional[MultimodalUnderstanding] = None


def get_multimodal_understanding() -> MultimodalUnderstanding:
    """获取多模态理解实例"""
    global _multimodal_understanding
    if _multimodal_understanding is None:
        _multimodal_understanding = MultimodalUnderstanding()
    return _multimodal_understanding
