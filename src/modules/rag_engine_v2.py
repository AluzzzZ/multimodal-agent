"""
RAG v2（摘要索引版）
用于加载 `knowledge_base/index_v2` 下的摘要索引与元数据，
兼容项目里现有的 `DualRouteRetriever` / `src.modules.__init__` 调用。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from loguru import logger

from config import settings
from .rag_engine import Document, RAGEngine


class SummaryDocument(Document):
    """带摘要的文档对象。"""

    def __init__(
        self,
        content: str,
        doc_id: str,
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(content=content, doc_id=doc_id, metadata=metadata)
        self.summary = summary

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["summary"] = self.summary
        return payload


class SummaryGenerator:
    """调用大模型生成摘要。"""

    def __init__(self):
        self.backend = settings.summary_backend
        self.model = settings.summary_model
        self.api_key = settings.summary_api_key or settings.dashscope_api_key
        self.base_url = settings.summary_base_url or settings.dashscope_base_url
        self.max_tokens = settings.summary_max_tokens
        self.temperature = settings.summary_temperature

    def generate_summary(self, text: str, section_title: str = "") -> str:
        if not text or not text.strip():
            return ""

        prompt = self._build_summary_prompt(text, section_title)

        if self.backend != "dashscope":
            return self._fallback_summary(text)

        try:
            import requests

            endpoint = (
                f"{(self.base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1').rstrip('/')}/"
                "services/aigc/text-generation/generation"
            )
            headers = {
                "Authorization": f"Bearer {self._get_api_key()}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": {"prompt": prompt},
                "parameters": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "result_format": "message",
                },
            }

            resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            if not resp.ok:
                logger.warning(f"摘要生成请求失败 [{resp.status_code}]: {resp.text[:200]}")
                return self._fallback_summary(text)

            data = resp.json()
            choices = data.get("output", {}).get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
        except Exception as exc:
            logger.warning(f"摘要生成异常: {exc}")

        return self._fallback_summary(text)

    def _build_summary_prompt(self, text: str, section_title: str) -> str:
        title_context = f"【章节】{section_title}\n" if section_title else ""
        return f"""你是一个手册检索摘要助手。请基于原文生成一段用于知识库召回的检索摘要，要求：

1. 只保留原文中已明确出现的信息，不要补充原文没有的新内容。
2. 优先保留型号、部件、功能、错误现象、参数、数值、单位、条件、步骤、警告。
3. 尽量保留原词、按钮名、英文术语、错误码、缩写与数值表达。
4. 删除空泛概述、背景描述、举例与重复句。
5. 摘要更像高信息密度的“检索提要”，不是长段概述。
6. 长度控制在 40-120 字。
7. 不要输出图片标记或图片编号。
8. 使用与原文相同的语言输出，禁止翻译。

{title_context}【原文】
{text[:1200]}

【摘要】"""

    def _fallback_summary(self, text: str) -> str:
        cleaned = " ".join(text.replace("\n", " ").split())
        return cleaned[:180]

    def _get_api_key(self) -> str:
        key = self.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            raise ValueError("未设置摘要模型 API Key")
        return key


class KnowledgeBaseV2:
    """基于摘要向量的知识库。"""

    def __init__(self):
        self.summary_index: Optional[faiss.Index] = None
        self.summary_documents: List[SummaryDocument] = []
        self.embedding_model = None
        self.embedding_backend: Optional[str] = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        self.embedding_backend = settings.embedding_backend
        logger.info(f"[V2] 加载嵌入后端: {self.embedding_backend}")
        base_engine = RAGEngine()
        self.embedding_model = base_engine.knowledge_base._create_embedding_model()
        self._load_index()
        self._initialized = True
        logger.info("[V2] 知识库初始化完成")

    def _load_index(self):
        index_dir = settings.knowledge_base_path / "index_v2"
        if not index_dir.exists():
            index_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[V2] 创建索引目录: {index_dir}")
            return

        index_file = index_dir / "summary_index.faiss"
        metadata_file = index_dir / "metadata_v2.json"

        if index_file.exists() and metadata_file.exists():
            try:
                self.summary_index = faiss.read_index(str(index_file))
                logger.info(f"[V2] 加载摘要索引: {self.summary_index.ntotal} 条")
            except Exception as exc:
                logger.warning(f"[V2] 摘要索引加载失败: {exc}")

            try:
                with open(metadata_file, "r", encoding="utf-8") as file:
                    metadata = json.load(file)

                if isinstance(metadata, dict):
                    raw_docs = metadata.get("texts") or metadata.get("documents") or []
                else:
                    raw_docs = metadata

                self.summary_documents = [
                    SummaryDocument(
                        content=item.get("content", ""),
                        doc_id=item.get("doc_id", ""),
                        summary=item.get("summary", ""),
                        metadata=item.get("metadata", {}),
                    )
                    for item in raw_docs
                ]
                logger.info(f"[V2] 加载摘要元数据: {len(self.summary_documents)} 条")
            except Exception as exc:
                logger.warning(f"[V2] 摘要元数据加载失败: {exc}")

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, settings.embedding_dim), dtype="float32")

        model = self.embedding_model
        if hasattr(model, "embed_documents"):
            embeddings = model.embed_documents(texts)
        elif hasattr(model, "encode"):
            embeddings = model.encode(texts)
        else:
            raise ValueError("当前 embedding model 不支持批量编码")

        array = np.array(embeddings, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self._initialized:
            self.initialize()

        if not query or self.summary_index is None or not self.summary_documents:
            return []

        search_top_k = top_k or settings.rag_top_k
        query_embedding = self._embed_texts([query])
        distances, indices = self.summary_index.search(query_embedding, search_top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.summary_documents):
                continue
            doc = self.summary_documents[idx]
            results.append(
                {
                    "content": doc.content,
                    "doc_id": doc.doc_id,
                    "summary": doc.summary,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )
        return results


class RAGEngineV2:
    """RAG v2 主引擎。"""

    def __init__(self):
        self.knowledge_base = KnowledgeBaseV2()
        self.summary_generator = SummaryGenerator()
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        self.knowledge_base.initialize()
        self._initialized = True
        logger.info("[V2] RAG引擎初始化完成")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.knowledge_base.search(query, top_k=top_k)


_rag_engine_v2: Optional[RAGEngineV2] = None


def get_rag_engine_v2() -> RAGEngineV2:
    global _rag_engine_v2
    if _rag_engine_v2 is None:
        _rag_engine_v2 = RAGEngineV2()
    return _rag_engine_v2
