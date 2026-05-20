"""
轻量路由分类器。

职责：
1. 将 query 归一化为固定输入文本；
2. 使用 LLM API 或本地 ONNX 小模型做 service/manual/mixed 三分类；
3. 模型缺失或运行失败时自动降级，不阻断主链路。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from config import settings
from src.utils.domain_knowledge import (
    ROUTE_MANUAL_HINTS,
    ROUTE_MIXED_HINTS,
    ROUTE_SERVICE_HINTS,
)


# 路由分类标签名称，与 ONNX 模型输出顺序一一对应
# service: 客服服务类问题（如退款、投诉、发票）
# manual: 产品使用类问题（如说明书、操作步骤）
# mixed: 混合问题（同时涉及客服和手册）
ROUTE_LABELS = ["service", "manual", "mixed"]


class RouteClassifierFeatureizer:
    """将单条 query 转为固定维度特征向量。"""

    SERVICE_HINTS = ROUTE_SERVICE_HINTS
    MANUAL_HINTS = ROUTE_MANUAL_HINTS
    MIXED_HINTS = ROUTE_MIXED_HINTS

    def __init__(self, dim: int = 512):
        self.dim = dim

    def build_input_text(
        self,
        normalized_query: str,
        images: Optional[List[str]] = None,
        image_tags: Optional[List[str]] = None,
        product_candidates: Optional[List[str]] = None,
    ) -> str:
        """
        构建分类器的输入文本。

        拼接规则:
        - 归一化查询文本（必选）
        - [HAS_IMAGE]标记（当有图片时）
        - [IMAGE_TAGS] + 图片标签列表（当启用且有标签时，最多6个）
        - [PRODUCT] + 候选手册名（当有产品候选时）

        Args:
            normalized_query: 归一化后的查询文本
            images: Base64图片列表
            image_tags: 图片标签列表
            product_candidates: 候选手册名称列表

        Returns:
            分类器输入文本字符串
        """
        parts = [normalized_query.strip()]
        if images:
            parts.append("[HAS_IMAGE]")
        if image_tags and settings.route_classifier_use_image_tags:
            parts.append("[IMAGE_TAGS] " + " / ".join([tag.strip() for tag in image_tags if tag.strip()][:6]))
        if product_candidates:
            parts.append("[PRODUCT] " + " / ".join(str(c).strip() for c in product_candidates[:4]))
        return " ".join([part for part in parts if part]).strip()

    def encode(
        self,
        normalized_query: str,
        images: Optional[List[str]] = None,
        image_tags: Optional[List[str]] = None,
        product_candidates: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        将query编码为固定维度特征向量。

        向量组成:
        1. 哈希嵌入向量（前dim-32维）: 对每个token做MD5，取哈希值决定向量位置和符号
        2. 手工特征向量（后32维）: _engineered_features提取的路由特征

        哈希策略: 用MD5前4字节作为向量索引，第5字节决定符号(+/-)，
        使相似文本的token重叠率高时，向量更接近。

        Args:
            normalized_query: 归一化后的查询文本
            images: Base64图片列表
            image_tags: 图片标签列表
            product_candidates: 候选手册名称列表

        Returns:
            归一化的特征向量
        """
        input_text = self.build_input_text(
            normalized_query,
            images=images,
            image_tags=image_tags,
            product_candidates=product_candidates,
        )
        vector = np.zeros(self.dim, dtype=np.float32)

        # 哈希嵌入（前dim-32维）
        for token in self._tokenize(input_text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            # 前4字节转整数取模作为索引（保留后32位给手工特征）
            index = int.from_bytes(digest[:4], "little") % (self.dim - 32)
            # 第5字节奇偶性决定正负符号
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        # 拼接手工特征向量（后32维）
        engineered = self._engineered_features(input_text)
        vector[-len(engineered):] = engineered

        # L2归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def _engineered_features(self, text: str) -> np.ndarray:
        """
        提取 32 维手工特征向量，用于增强哈希嵌入的语义捕获能力。

        特征维度设计说明（共 32 维）:
        - features[0]: service 意图关键词命中率（归一化到 [0,1]）
        - features[1]: manual 意图关键词命中率
        - features[2]: mixed 意图关键词命中率
        - features[3]: 是否有图片（0/1 二值特征）
        - features[4]: 英文词项密度（反映产品型号、技术术语）
        - features[5]: 中文字符密度
        - features[6]: 是否包含图示类词汇（说明书/图示/配图等）
        - features[7]: 是否包含售后类词汇（退款/退货/发票等）
        - features[8]: 是否包含操作类词汇（如何/怎么/步骤等）
        - features[9]: 查询文本长度（归一化到 [0,1]，最长按 180 字符计）
        - features[10-31]: 预留给未来的扩展特征（目前全为 0）

        命中率计算: min(命中数 / 基准数, 1.0)，防止极端值破坏归一化。

        Args:
            text: 预处理后的输入文本

        Returns:
            32 维特征向量（numpy float32 数组）
        """
        features = np.zeros(32, dtype=np.float32)
        lowered = text.lower()

        service_hits = sum(1 for token in self.SERVICE_HINTS if token in lowered)
        manual_hits = sum(1 for token in self.MANUAL_HINTS if token in lowered)
        mixed_hits = sum(1 for token in self.MIXED_HINTS if token in lowered)
        english_terms = len(re.findall(r"[a-z]{3,}", lowered))
        chinese_terms = len(re.findall(r"[\u4e00-\u9fff]", lowered))

        # 命中率归一化，防止极端别名集合导致分值爆炸
        features[0] = min(1.0, service_hits / 6.0)
        features[1] = min(1.0, manual_hits / 6.0)
        features[2] = min(1.0, mixed_hits / 5.0)
        features[3] = 1.0 if "[has_image]" in lowered else 0.0
        features[4] = min(1.0, english_terms / 12.0)
        features[5] = min(1.0, chinese_terms / 40.0)
        features[6] = 1.0 if any(token in lowered for token in ("说明书", "图示", "配图", "电子版", "纸质版") if token in self.MANUAL_HINTS) else 0.0
        features[7] = 1.0 if any(token in lowered for token in ("退款", "退货", "换货", "发票", "优惠券", "以旧换新") if token in self.SERVICE_HINTS) else 0.0
        features[8] = 1.0 if any(token in lowered for token in ("如何", "怎么", "步骤", "安装", "设置", "清洁") if token in self.MANUAL_HINTS) else 0.0
        features[9] = min(1.0, len(text) / 180.0)
        return features

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = (text or "").strip().lower()
        if not cleaned:
            return []

        tokens = re.findall(r"[a-z0-9_+-]+", cleaned)
        for segment in re.findall(r"[\u4e00-\u9fff]+", cleaned):
            if len(segment) == 1:
                tokens.append(segment)
                continue
            for idx in range(len(segment) - 1):
                tokens.append(segment[idx: idx + 2])
            for idx in range(len(segment) - 2):
                tokens.append(segment[idx: idx + 3])
        return tokens


class RouteClassifier:
    """
    ONNX CPU 路由分类器。

    使用本地ONNX Runtime进行CPU推理:
    - 支持热加载，模型缺失时自动降级规则路由
    - 与PyTorch解耦，适合生产部署

    分类结果用于DualRouteRetriever的路由裁决环节。
    """

    def __init__(self):
        self.model_dir: Path = settings.route_classifier_model_dir
        self.featureizer = RouteClassifierFeatureizer(settings.route_classifier_feature_dim)
        self.backend = settings.route_classifier_backend
        self.labels = list(ROUTE_LABELS)
        self.session = None
        self.enabled = settings.route_classifier_enabled
        self.ready = False
        self._manifest: Dict[str, Any] = {}
        self._initialized = False
        self._llm_client = None

    def initialize(self) -> None:
        """
        初始化路由分类器。

        支持的后端：
        - llm: 使用 LLM API 进行分类
        - onnx: 使用本地 ONNX 模型
        - rule: 仅使用规则路由（降级）
        """
        if self._initialized:
            return

        self._initialized = True
        if not self.enabled:
            logger.info("路由分类器已禁用，将直接回退规则路由")
            return

        if self.backend == "llm":
            self._initialize_llm()
        elif self.backend == "onnx":
            self._initialize_onnx()
        else:
            logger.info(f"路由分类器后端 {self.backend} 不支持，将回退规则路由")
            self.ready = False

    def _initialize_llm(self) -> None:
        """初始化 LLM API 后端"""
        try:
            from openai import OpenAI
            self._llm_client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
            self.ready = True
            logger.info("路由分类器 LLM 后端初始化成功")
        except Exception as exc:
            logger.warning(f"路由分类器 LLM 初始化失败: {exc}")
            self.ready = False

    def _initialize_onnx(self) -> None:
        """初始化 ONNX 模型后端"""
        manifest_path = self.model_dir / "manifest.json"
        model_path = self.model_dir / "route_classifier.onnx"
        if not manifest_path.exists() or not model_path.exists():
            logger.warning(f"路由分类器模型缺失: {self.model_dir}")
            return

        try:
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.labels = self._manifest.get("labels", self.labels)
            self.featureizer = RouteClassifierFeatureizer(
                int(self._manifest.get("feature_dim", settings.route_classifier_feature_dim))
            )

            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self.ready = True
            logger.info(f"路由分类器 ONNX 后端加载成功: labels={self.labels}")
        except Exception as exc:
            logger.warning(f"路由分类器 ONNX 初始化失败: {exc}")
            self.session = None
            self.ready = False

    def predict(
        self,
        normalized_query: str,
        images: Optional[List[str]] = None,
        image_tags: Optional[List[str]] = None,
        product_candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        对query进行路由分类预测。

        支持的后端：
        - llm: 使用 LLM API 进行分类
        - onnx: 使用本地 ONNX 模型
        - rule: 仅使用规则路由（返回 available=False）

        Args:
            normalized_query: 归一化后的查询文本
            images: Base64图片列表
            image_tags: 图片标签列表
            product_candidates: 候选手册名称列表

        Returns:
            包含available/label/confidence/probs的预测结果字典
        """
        if not self._initialized:
            self.initialize()

        input_text = self.featureizer.build_input_text(
            normalized_query,
            images=images,
            image_tags=image_tags,
            product_candidates=product_candidates,
        )

        # LLM 后端
        if self.backend == "llm" and self.ready:
            return self._predict_llm(normalized_query, input_text)

        # ONNX 后端
        if self.backend == "onnx" and self.ready and self.session is not None:
            feature_vector = self.featureizer.encode(
                normalized_query,
                images=images,
                image_tags=image_tags,
                product_candidates=product_candidates,
            )
            return self._predict_onnx(input_text, feature_vector)

        # 降级：规则路由
        return {
            "available": False,
            "label": None,
            "confidence": 0.0,
            "probs": {label: 0.0 for label in self.labels},
            "input_text": input_text,
            "backend": self.backend,
            "fallback_reason": "classifier_unavailable",
        }

    def _predict_llm(
        self,
        normalized_query: str,
        input_text: str,
    ) -> Dict[str, Any]:
        """使用 LLM API 进行分类"""
        prompt = f"""分析以下用户问题，判断其类型：

问题：{normalized_query}

请从以下三个类别中选择一个：
- service：客服服务类问题（如退款、投诉、发票、物流、安装、售后等）
- manual：产品使用类问题（如说明书、操作步骤、部件功能、设置方法等）
- mixed：混合问题（同时涉及客服服务和产品使用）

只输出一个词：service、manual 或 mixed。"""

        try:
            response = self._llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            result_raw = response.choices[0].message.content
            if result_raw is None:
                raise ValueError("LLM returned empty response")
            result = result_raw.strip().lower()

            # 解析结果
            if "service" in result and "manual" not in result:
                label = "service"
            elif "manual" in result and "service" not in result:
                label = "manual"
            elif "mixed" in result:
                label = "mixed"
            else:
                # 解析失败，使用规则
                return {
                    "available": False,
                    "label": None,
                    "confidence": 0.0,
                    "probs": {label: 0.0 for label in self.labels},
                    "input_text": input_text,
                    "backend": self.backend,
                    "fallback_reason": "llm_parse_error",
                }

            return {
                "available": True,
                "label": label,
                "confidence": 0.8,
                "probs": {label: 0.8 if label == l else 0.1 for l in self.labels},
                "input_text": input_text,
                "backend": self.backend,
                "fallback_reason": "",
            }
        except Exception as exc:
            logger.warning(f"LLM 分类失败: {exc}")
            return {
                "available": False,
                "label": None,
                "confidence": 0.0,
                "probs": {label: 0.0 for label in self.labels},
                "input_text": input_text,
                "backend": self.backend,
                "fallback_reason": "llm_error",
            }

    def _predict_onnx(self, input_text: str, feature_vector: np.ndarray) -> Dict[str, Any]:
        """使用 ONNX 模型进行分类"""
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: feature_vector.reshape(1, -1).astype(np.float32)})
            probs = np.asarray(outputs[0])[0]
            best_idx = int(np.argmax(probs))
            best_label = self.labels[best_idx]
            probability_map = {
                label: round(float(score), 6)
                for label, score in zip(self.labels, probs.tolist())
            }
            return {
                "available": True,
                "label": best_label,
                "confidence": round(float(probs[best_idx]), 6),
                "probs": probability_map,
                "input_text": input_text,
                "backend": self.backend,
                "fallback_reason": "",
            }
        except Exception as exc:
            logger.warning(f"路由分类器 ONNX 推理失败: {exc}")
            return {
                "available": False,
                "label": None,
                "confidence": 0.0,
                "probs": {label: 0.0 for label in self.labels},
                "input_text": input_text,
                "backend": self.backend,
                "fallback_reason": "onnx_error",
            }


_route_classifier: Optional[RouteClassifier] = None


def get_route_classifier() -> RouteClassifier:
    global _route_classifier
    if _route_classifier is None:
        _route_classifier = RouteClassifier()
    return _route_classifier


def reset_route_classifier() -> None:
    """重置全局路由分类器单例，便于测试和阈值扫描。"""
    global _route_classifier
    _route_classifier = None


# 模块级导出：将类属性暴露为模块常量，方便 dual_route_retriever 等模块直接导入使用
SERVICE_HINTS = RouteClassifierFeatureizer.SERVICE_HINTS
MANUAL_HINTS = RouteClassifierFeatureizer.MANUAL_HINTS
MIXED_HINTS = RouteClassifierFeatureizer.MIXED_HINTS
