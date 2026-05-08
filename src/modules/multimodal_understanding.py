"""
多模态理解模块
负责解析用户输入的文本和图片，识别用户意图
"""

import base64
import io
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from loguru import logger

from config import settings


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
            # 使用备用方案
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
    
    def describe_image(self, image_data: str) -> str:
        """
        生成图片描述
        
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
            
            if self.model is None:
                return "无法生成描述，视觉模型未配置"
            
            # 生成描述
            describe_prompt = "描述这张图片的内容，包括产品类型、外观特征、可能的用途等"
            inputs = self.processor(text=[describe_prompt], images=image, return_tensors="pt")
            if settings.embedding_device == "cuda" and self.torch and self.torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            
            # 简单的描述生成（实际应用中可使用更复杂的模型）
            return f"图片尺寸: {image.size}, 内容分析完成"
            
        except Exception as e:
            logger.error(f"图片描述生成失败: {e}")
            return "图片描述生成失败"


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


# 全局实例
_multimodal_understanding: Optional[MultimodalUnderstanding] = None


def get_multimodal_understanding() -> MultimodalUnderstanding:
    """获取多模态理解实例"""
    global _multimodal_understanding
    if _multimodal_understanding is None:
        _multimodal_understanding = MultimodalUnderstanding()
    return _multimodal_understanding
