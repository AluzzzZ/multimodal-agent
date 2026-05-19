"""
多模态理解结果数据结构

定义 MultimodalUnderstandingResult 供所有理解相关模块使用，
避免 vlm_understanding.py 与 multimodal_understanding.py 之间的循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MultimodalUnderstandingResult:
    """多模态理解结构化输出"""
    normalized_query: str = ""
    language: str = "zh"
    image_tags: List[str] = field(default_factory=list)
    product_candidates: List[str] = field(default_factory=list)
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
            "product_candidates": self.product_candidates,
            "visual_intents": self.visual_intents,
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "notes": self.notes,
            "referenced_previous_object": self.referenced_previous_object,
            "requires_history_resolution": self.requires_history_resolution,
        }
