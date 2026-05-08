"""
模块初始化

采用惰性导入，避免普通命令在导入 `src.modules` 时顺带加载
torch / transformers / sentence-transformers 等重型依赖。
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "MultimodalUnderstanding": ("src.modules.multimodal_understanding", "MultimodalUnderstanding"),
    "TextParser": ("src.modules.multimodal_understanding", "TextParser"),
    "ImageParser": ("src.modules.multimodal_understanding", "ImageParser"),
    "get_multimodal_understanding": ("src.modules.multimodal_understanding", "get_multimodal_understanding"),
    "RAGEngine": ("src.modules.rag_engine", "RAGEngine"),
    "KnowledgeBase": ("src.modules.rag_engine", "KnowledgeBase"),
    "Document": ("src.modules.rag_engine", "Document"),
    "Reranker": ("src.modules.rag_engine", "Reranker"),
    "get_rag_engine": ("src.modules.rag_engine", "get_rag_engine"),
    "ConversationManager": ("src.modules.conversation_manager", "ConversationManager"),
    "ConversationContext": ("src.modules.conversation_manager", "ConversationContext"),
    "Message": ("src.modules.conversation_manager", "Message"),
    "DialogueState": ("src.modules.conversation_manager", "DialogueState"),
    "get_conversation_manager": ("src.modules.conversation_manager", "get_conversation_manager"),
    "HallucinationController": ("src.modules.hallucination_controller", "HallucinationController"),
    "ChainOfThoughtReasoner": ("src.modules.hallucination_controller", "ChainOfThoughtReasoner"),
    "get_hallucination_controller": ("src.modules.hallucination_controller", "get_hallucination_controller"),
    "get_cot_reasoner": ("src.modules.hallucination_controller", "get_cot_reasoner"),
    "ResponseGenerator": ("src.modules.response_generator", "ResponseGenerator"),
    "get_response_generator": ("src.modules.response_generator", "get_response_generator"),
    "DualRouteRetriever": ("src.modules.dual_route_retriever", "DualRouteRetriever"),
    "get_dual_route_retriever": ("src.modules.dual_route_retriever", "get_dual_route_retriever"),
    "RouteClassifier": ("src.modules.route_classifier", "RouteClassifier"),
    "get_route_classifier": ("src.modules.route_classifier", "get_route_classifier"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """按需加载导出对象，降低导入时的内存峰值。"""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.modules' has no attribute '{name}'")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
