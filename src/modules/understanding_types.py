"""
多模态理解结果数据结构

定义 MultimodalUnderstandingResult 供所有理解相关模块使用，
避免 vlm_understanding.py 与 multimodal_understanding.py 之间的循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(eq=False, unsafe_hash=True)
class ProductCandidate:
    """
    带来源和置信度的产品候选。

    字段：
    - name: 标准产品名（与 PRODUCT_ALIAS_MAP 对应）
    - score: 置信度 [0.0, 1.0]，由高到低排列
    - source: 来源，"text_alias" | "image_tag" | "history" | "vlm"
    """
    name: str
    score: float = 1.0
    source: str = "text_alias"

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))

    @classmethod
    def from_name(cls, name: str, score: float = 1.0, source: str = "text_alias") -> "ProductCandidate":
        """从产品名构造候选，便于从旧代码迁移。"""
        return cls(name=name, score=score, source=source)


@dataclass
class MultimodalUnderstandingResult:
    """多模态理解结构化输出"""
    normalized_query: str = ""
    language: str = "zh"
    image_tags: List[str] = field(default_factory=list)
    product_candidates: List[ProductCandidate] = field(default_factory=list)
    visual_intents: List[str] = field(default_factory=list)
    evidence_type: str = "unknown"
    confidence: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)
    referenced_previous_object: Optional[str] = None
    requires_history_resolution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_query": self.normalized_query,
            "language": self.language,
            "image_tags": self.image_tags,
            "product_candidates": [
                {"name": c.name, "score": c.score, "source": c.source}
                for c in self.product_candidates
            ],
            "visual_intents": self.visual_intents,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "notes": self.notes,
            "referenced_previous_object": self.referenced_previous_object,
            "requires_history_resolution": self.requires_history_resolution,
        }
