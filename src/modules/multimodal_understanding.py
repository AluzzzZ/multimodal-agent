"""
多模态理解模块
负责解析用户输入的文本和图片，识别用户意图
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from loguru import logger

from config import settings


# ---------------------------------------------------------------------------
# 结构化理解结果
# ---------------------------------------------------------------------------

@dataclass
class MultimodalUnderstandingResult:
    """多模态理解结构化输出"""
    normalized_query: str = ""           # 归一化后的查询文本
    language: str = "zh"                 # 语言 zh/en
    image_tags: List[str] = field(default_factory=list)   # 从图片提取的稳定标签
    product_candidates: List[str] = field(default_factory=list)  # 候选产品名
    visual_intents: List[str] = field(default_factory=list)     # 视觉意图
    evidence_type: str = "unknown"       # service_like / manual_like / mixed_like / unknown
    confidence: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_query": self.normalized_query,
            "language": self.language,
            "image_tags": self.image_tags,
            "product_candidates": self.product_candidates,
            "visual_intents": self.visual_intents,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "notes": self.notes,
        }


class TextParser:
    """文本解析器"""
    
    def __init__(self):
        self.llm_client = None  # 延迟初始化
        self._initialized = False
    
    def initialize(self):
        """初始化LLM客户端"""
        if self._initialized:
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
                # 本地模型支持
                from langchain_community.chat_models import ChatOllama
                self.llm_client = ChatOllama(
                    model=settings.llm_model,
                    temperature=settings.llm_temperature
                )
            logger.info(f"LLM客户端初始化成功: {settings.llm_provider}")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            raise
        
        self._initialized = True
    
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
            if not self._initialized:
                self.initialize()
            
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
        """
        拆分复杂问题为多个子问题
        
        Args:
            text: 用户输入的复杂问题
            
        Returns:
            子问题列表
        """
        if not self._initialized:
            self.initialize()
        
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

    def __init__(self):
        self.model = None
        self.processor = None
        self.torch = None
        self._caption_model = None
        self._caption_processor = None
        self._initialized = False

    def initialize(self):
        """初始化视觉模型"""
        if self._initialized:
            return

        try:
            if not settings.enable_vision_model:
                logger.info("视觉模型已禁用，使用轻量模式运行")
                self.model = None
                self.processor = None
                self._initialized = True
                return

            import torch
            from transformers import CLIPProcessor, CLIPModel

            self.torch = torch
            logger.info(f"加载视觉模型: {settings.vision_model}")
            self.model = CLIPModel.from_pretrained(settings.vision_model)
            self.processor = CLIPProcessor.from_pretrained(settings.vision_model)

            if settings.embedding_device == "cuda" and self.torch.cuda.is_available():
                self.model = self.model.to("cuda")

            self.model.eval()
            logger.info("视觉模型加载成功")
        except Exception as e:
            logger.error(f"视觉模型加载失败: {e}")
            self.model = None
            self.processor = None

        self._initialized = True
    
    def parse_image(self, image_data: str) -> Dict[str, Any]:
        """
        解析图片输入
        
        Args:
            image_data: Base64编码的图片数据
            
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
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 如果没有加载模型，返回基本信息
            if self.model is None:
                return {
                    "format": image.format or "unknown",
                    "size": image.size,
                    "mode": image.mode,
                    "description": "图片已接收，视觉分析待配置"
                }
            
            # 使用CLIP提取图片特征
            inputs = self.processor(images=image, return_tensors="pt")
            if settings.embedding_device == "cuda" and self.torch and self.torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with self.torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            return {
                "format": image.format or "unknown",
                "size": image.size,
                "mode": image.mode,
                "features": image_features.cpu().numpy().flatten(),
                "description": "图片特征已提取"
            }
            
        except Exception as e:
            logger.error(f"图片解析失败: {e}")
            return {
                "error": str(e),
                "description": "图片解析失败"
            }
    
        try:
            if not settings.enable_vision_model:
                logger.info("视觉模型已禁用，使用轻量模式运行")
                self.model = None
                self.processor = None
                self._initialized = True
                return

            import torch
            from transformers import CLIPProcessor, CLIPModel

            self.torch = torch
            logger.info(f"加载视觉模型: {settings.vision_model}")
            self.model = CLIPModel.from_pretrained(settings.vision_model)
            self.processor = CLIPProcessor.from_pretrained(settings.vision_model)

            if settings.embedding_device == "cuda" and self.torch.cuda.is_available():
                self.model = self.model.to("cuda")

            self.model.eval()
            logger.info("视觉模型加载成功")

            # 延迟加载图片描述模型（caption 模型），避免主链路时延增加
            self._load_caption_model()
        except Exception as e:
            logger.error(f"视觉模型加载失败: {e}")
            self.model = None
            self.processor = None

        self._initialized = True

    def _load_caption_model(self):
        """
        按需加载轻量 caption 模型，支持两种部署路径：
        1. Ollama 本地 VLM（如 llava、moondream）
        2. 托管 VLM API（通过 OpenAI 兼容接口调用）
        """
        if not settings.enable_vision_model:
            return

        try:
            # 优先尝试 Ollama 本地 VLM（llava 或 moondream）
            if settings.llm_provider == "local":
                from ollama import chat
                self._caption_model = "ollama"
                logger.info("图片描述使用 Ollama 本地 VLM")
                return
        except ImportError:
            pass

        # 降级：使用 CLIP zero-shot classification 作为场景描述生成
        # 通过预设的候选标签池做 top-k 标签，不依赖额外模型
        self._caption_model = "clip_classify"
        logger.info("图片描述使用 CLIP zero-shot 分类（标签池）")

    def describe_image(self, image_data: str) -> str:
        """
        生成图片描述。

        优先顺序：
        1. Ollama 本地 VLM（llava/moondream）
        2. CLIP zero-shot 分类 + 规则生成描述
        3. 模型未配置时返回降级说明

        Args:
            image_data: Base64编码的图片数据

        Returns:
            图片的自然语言描述
        """
        if not self._initialized:
            self.initialize()

        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            if self.model is None and self._caption_model is None:
                return "图片已接收，视觉分析待配置"

            # 策略1: Ollama 本地 VLM
            if self._caption_model == "ollama":
                return self._caption_via_ollama(image)

            # 策略2: CLIP zero-shot 分类 + 规则生成
            return self._caption_via_clip_classify(image)

        except Exception as e:
            logger.error(f"图片描述生成失败: {e}")
            return "图片描述生成失败"

    def _caption_via_ollama(self, image: Image.Image) -> str:
        """通过 Ollama 本地 VLM 生成图片描述"""
        try:
            from ollama import chat
            from ollama import Message

            response = chat(
                model="llava",
                messages=[
                    Message(
                        role="user",
                        content="请用一句话描述这张图片的内容，包括产品类型、外观特征、可能的用途。回答控制在20个字以内。",
                        images=[image],
                    )
                ],
                options={"temperature": 0.3, "num_predict": 50},
            )
            caption = response.message.content.strip()
            logger.debug(f"Ollama caption: {caption}")
            return caption
        except Exception as e:
            logger.warning(f"Ollama caption 失败: {e}，降级 CLIP 分类")
            return self._caption_via_clip_classify(image)

    def _caption_via_clip_classify(self, image: Image.Image) -> str:
        """
        通过 CLIP zero-shot 分类生成图片场景描述。

        流程：
        1. 用预设标签池做 top-k 分类
        2. 按标签组合生成结构化描述
        3. 结合图像尺寸信息形成最终 caption
        """
        if self.model is None or self.processor is None:
            return "无法生成描述，视觉模型未配置"

        # 预设标签池：覆盖主要产品类别和使用场景
        label_pool = [
            # 产品类型
            "电钻", "电池充电器", "电动工具", "健身器材", "健身手环",
            "智能手表", "空气净化器", "冰箱", "空调", "烤箱", "洗碗机",
            "键盘", "鼠标", "显示器", "相机", "游戏手柄", "耳机",
            "路由器", "打印机", "投影仪",
            # 外观特征
            "金属质感", "塑料外壳", "工业设计", "小型便携", "大型家用",
            "LCD显示屏", "LED指示灯", "触摸屏", "物理按键",
            # 使用场景
            "室内使用", "室外使用", "工作场景", "家居场景", "实验室",
            # 状态
            "指示灯亮", "屏幕显示", "充电中", "待机状态", "运行中",
            # 图示/文档
            "产品说明书", "电路图", "示意图", "爆炸图", "操作面板",
        ]

        try:
            import torch
            inputs = self.processor(
                text=label_pool,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            if settings.embedding_device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                logits_per_image, _ = self.model(**inputs)
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            # 取 top-3 标签
            top_indices = probs.argsort()[-3:][::-1]
            top_labels = [label_pool[i] for i in top_indices]
            top_scores = [round(float(probs[i]), 3) for i in top_indices]

            logger.debug(f"CLIP top-3: {list(zip(top_labels, top_scores))}")

            # 按场景 > 产品 > 外观的顺序生成描述
            scene_labels = {"室内使用", "室外使用", "工作场景", "家居场景", "实验室"}
            product_labels = set(top_labels) - scene_labels
            scene = next((l for l in top_labels if l in scene_labels), None)

            if scene:
                primary = [l for l in top_labels if l not in scene_labels]
            else:
                primary = top_labels[:2]

            parts = primary[:2]
            if scene:
                parts.append(scene)

            description = "、".join(parts)
            return description

        except Exception as e:
            logger.warning(f"CLIP 分类失败: {e}")
            return "图片内容分析完成"


class MultimodalUnderstanding:
    """多模态理解主类"""
    
    def __init__(self):
        self.text_parser = TextParser()
        self.image_parser = ImageParser()
        self._initialized = False
    
    def initialize(self):
        """初始化所有组件"""
        self.text_parser.initialize()
        self.image_parser.initialize()
        self._initialized = True
        logger.info("多模态理解模块初始化完成")
    
    def understand(
        self,
        text: Optional[str] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        综合理解用户输入
        
        Args:
            text: 用户输入的文本
            images: Base64编码的图片列表
            
        Returns:
            综合理解结果
        """
        if not self._initialized:
            self.initialize()
        
        result = {
            "text_analysis": None,
            "image_analysis": [],
            "combined_intent": None,
            "requires_multimodal": False,
            "confidence": 0.0
        }
        
        # 文本分析
        if text:
            result["text_analysis"] = self.text_parser.parse(text)
        
        # 图片分析
        if images:
            for img in images:
                img_result = self.image_parser.parse_image(img)
                result["image_analysis"].append(img_result)
            result["requires_multimodal"] = True
        
        # 综合意图判断
        result["combined_intent"] = self._combine_intents(result)
        
        return result
    
    def _combine_intents(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """综合文本和图片的意图"""
        text_intent = analysis.get("text_analysis", {})
        has_images = len(analysis.get("image_analysis", [])) > 0
        
        combined = {
            "primary_intent": text_intent.get("intent", "unknown"),
            "entities": text_intent.get("entities", []),
            "questions": text_intent.get("questions", []),
            "requires_images": text_intent.get("requires_images", has_images),
            "confidence": text_intent.get("confidence", 0.5)
        }
        
        return combined

    def analyze(
        self,
        question: str,
        images: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> MultimodalUnderstandingResult:
        """
        结构化多模态理解主入口。

        流程：
        1. 归一化 query（语言检测 + 翻译）
        2. 提取图片标签（视觉模型或备用规则）
        3. 推断候选产品
        4. 推断视觉意图
        5. 判断证据类型（service_like / manual_like / mixed_like）

        Args:
            question: 用户问题
            images: Base64 图片列表（可选）
            conversation_history: 最近几轮对话（可选）

        Returns:
            MultimodalUnderstandingResult
        """
        if not self._initialized:
            self.initialize()

        if not question or not question.strip():
            return MultimodalUnderstandingResult()

        # Step 1: 归一化（只调一次 QueryProcessor，同时拿到 language / translation 信息）
        from src.utils.text_utils import QueryProcessor
        norm_result = QueryProcessor.normalize_query_for_retrieval(question)
        normalized = norm_result["normalized_query"]
        lang = norm_result["language"]

        # Step 2: 图片标签提取
        image_tags = self._extract_image_tags(images)

        # Step 3: 推断候选产品
        product_candidates = self._infer_product_candidates(question, image_tags)

        # Step 4: 推断视觉意图
        visual_intents = self._infer_visual_intents(question, image_tags)

        # Step 5: 证据类型判断
        evidence_type = self._infer_evidence_type(question, image_tags, product_candidates)

        return MultimodalUnderstandingResult(
            normalized_query=normalized,
            language=lang,
            image_tags=image_tags,
            product_candidates=product_candidates,
            visual_intents=visual_intents,
            evidence_type=evidence_type,
            confidence=0.8 if image_tags else 0.6,
            notes={"images_provided": len(images) if images else 0},
        )

    # -------------------------------------------------------------------------
    # 私有方法
    # -------------------------------------------------------------------------

    def _normalize_query(self, question: str) -> str:
        """归一化查询文本：委托 QueryProcessor 处理（包含英文归一化/翻译等完整逻辑）"""
        from src.utils.text_utils import QueryProcessor
        result = QueryProcessor.normalize_query_for_retrieval(question)
        return result["normalized_query"]

    def _detect_language(self, text: str) -> str:
        """简单语言检测"""
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        if chinese_chars >= 3:
            return "zh"
        return "en"

    def _extract_image_tags(self, images: Optional[List[str]]) -> List[str]:
        """
        提取图片标签。
        优先使用视觉模型，模型未配置时用规则兜底。
        """
        if not images:
            return []

        all_tags = []
        for img in images[:3]:  # 最多处理 3 张
            try:
                parsed = self.image_parser.parse_image(img)
                # 尝试从描述中抽取标签词
                desc = parsed.get("description", "")
                tags = self._parse_tags_from_description(desc)
                all_tags.extend(tags)
            except Exception as e:
                logger.warning(f"图片标签提取失败: {e}")

        return list(set(all_tags))[:10]  # 去重，最多 10 个

    def _parse_tags_from_description(self, description: str) -> List[str]:
        """从图片描述中抽取关键词作为标签"""
        if not description or description in ("图片已接收，视觉分析待配置", "图片解析失败"):
            return self._fallback_image_tags(description)

        keywords = re.findall(r"[\u4e00-\u9fff]{2,}", description)
        # 过滤常见无意义词
        stop = {"图片", "内容", "分析", "完成", "尺寸", "格式"}
        return [k for k in keywords if k not in stop][:5]

    def _fallback_image_tags(self, description: str) -> List[str]:
        """无视觉模型时的备用标签"""
        tags = []
        if "DRILL" in description or "电钻" in description:
            tags.append("电钻")
        if "refrigerator" in description or "冰箱" in description:
            tags.append("冰箱")
        return tags

    def _infer_product_candidates(self, question: str, image_tags: List[str]) -> List[str]:
        """从问题和图片标签中推断候选产品名"""
        candidates = set()

        # 从问题中提取产品名（词典匹配）
        product_aliases = {
            "VR头显": ["VR头显", "头显", "VR设备"],
            "人体工学椅": ["人体工学椅", "工学椅", "办公椅"],
            "健身追踪器": ["健身追踪器", "手环", "表带"],
            "电钻": ["电钻", "DCB107", "DCB112"],
            "冰箱": ["冰箱", "冷藏室", "冷冻室"],
            "空气净化器": ["空气净化器", "净化器"],
            "功能键盘": ["功能键盘", "键盘"],
            "发电机": ["发电机", "发动机"],
            "相机": ["相机", "镜头"],
            "空调": ["空调", "遥控器"],
            "烤箱": ["烤箱", "空气炸锅"],
            "洗碗机": ["洗碗机"],
            "电钻": ["电钻"],
        }
        for product, aliases in product_aliases.items():
            for alias in aliases:
                if alias in question:
                    candidates.add(product)
                    break

        # 从图片标签中补充
        for tag in image_tags:
            for product, aliases in product_aliases.items():
                if tag in aliases:
                    candidates.add(product)

        return list(candidates)[:5]

    def _infer_visual_intents(self, question: str, image_tags: List[str]) -> List[str]:
        """从问题和图片标签中推断视觉意图"""
        intents = []
        intent_map = {
            "查看位置": ["在哪", "位置", "哪里"],
            "查看指示灯": ["指示灯", "灯", "亮"],
            "查看按钮": ["按钮", "按键", "操作"],
            "查看说明书": ["说明书", "手册"],
            "查看图示": ["图示", "示意图", "图"],
            "查看型号": ["型号", "哪个"],
        }
        for intent, keywords in intent_map.items():
            if any(kw in question for kw in keywords):
                intents.append(intent)
        # 从图片标签补充
        for tag in image_tags:
            if any(kw in tag for kw in ["灯", "按钮", "图", "位置"]):
                if tag not in intents:
                    intents.append(tag)
        return intents[:5]

    def _infer_evidence_type(
        self,
        question: str,
        image_tags: List[str],
        product_candidates: List[str],
    ) -> str:
        """判断问题最可能需要哪种证据来源"""
        service_keywords = [
            "退款", "退货", "换货", "发票", "保修", "质保", "物流",
            "运费", "投诉", "赔偿", "补寄", "签收", "优惠券", "发货", "揽收",
        ]
        manual_keywords = [
            "说明书", "指示灯", "按钮", "怎么", "如何", "安装",
            "使用", "设置", "功能", "模式", "在哪", "哪个",
        ]

        svc_hits = sum(1 for kw in service_keywords if kw in question)
        man_hits = sum(1 for kw in manual_keywords if kw in question)

        if svc_hits > man_hits:
            return "service_like"
        if man_hits > svc_hits:
            return "manual_like"
        if image_tags or product_candidates:
            return "mixed_like"
        return "unknown"


# 全局实例
_multimodal_understanding: Optional[MultimodalUnderstanding] = None


def get_multimodal_understanding() -> MultimodalUnderstanding:
    """获取多模态理解实例"""
    global _multimodal_understanding
    if _multimodal_understanding is None:
        _multimodal_understanding = MultimodalUnderstanding()
    return _multimodal_understanding
