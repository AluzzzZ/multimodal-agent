"""
知识库构建脚本 V2
处理手册数据并构建基于摘要索引的 RAG 知识库
支持增量构建，输出到独立的 `knowledge_base/index_v2`

设计目标:
1. 沿用现有手册解析逻辑，兼容赛题手册格式
2. 以章节/子块为单位生成摘要，供摘要向量检索使用
3. 文本原文与摘要一并写入 `metadata_v2.json`
4. 摘要向量单独写入 `summary_index.faiss`
"""

from __future__ import annotations

import ast
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.modules.rag_engine_v2 import SummaryDocument, get_rag_engine_v2


class ManualParser:
    """解析赛题格式的手册文件。"""

    @staticmethod
    def parse_manual_file(content: str) -> Tuple[str, List[str]]:
        content = content.strip()
        if not content:
            return "", []

        try:
            parsed = ast.literal_eval(content)
            if (
                isinstance(parsed, (list, tuple))
                and len(parsed) >= 2
                and isinstance(parsed[0], str)
                and isinstance(parsed[1], list)
            ):
                image_ids = [str(item) for item in parsed[1]]
                return parsed[0], image_ids
        except Exception:
            pass

        image_ids: List[str] = []
        content_only = content
        list_match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]$', content, re.DOTALL)
        if list_match:
            try:
                image_ids = ast.literal_eval(list_match.group())
                content_only = content[:list_match.start()].strip()
            except Exception:
                pass

        return content_only, image_ids

    @staticmethod
    def inject_image_ids(raw_text: str, image_ids: List[str]) -> str:
        if not raw_text:
            return ""

        image_iter = iter(image_ids)

        def replace_pic(_: re.Match) -> str:
            image_id = next(image_iter, None)
            if image_id is None:
                return "<PIC>"
            return f"<PIC>[{image_id}]"

        text = re.sub(r"<PIC>", replace_pic, raw_text)
        remaining = list(image_iter)
        if remaining:
            text = text.rstrip() + "\n" + " ".join(f"<PIC>[{img_id}]" for img_id in remaining)
        return text

    @staticmethod
    def normalize_manual_text(raw_text: str) -> str:
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]{2,}#\s*", "\n# ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def extract_sections(raw_content: str) -> List[Dict[str, str]]:
        normalized = ManualParser.normalize_manual_text(raw_content)
        if not normalized:
            return []

        candidate_pattern = re.compile(r'.?(?<=[。！？\n])\s*#\s*([^\n#]{1,80})')
        matches = list(candidate_pattern.finditer(normalized))

        if not matches:
            fallback_pattern = re.compile(r'^(?!<PIC>)#\s*([^\n#]{1,80})')
            fallback_matches = list(fallback_pattern.finditer(normalized))
            if fallback_matches:
                matches = fallback_matches
            else:
                return [{"title": "全文", "content": normalized}]

        sections: List[Dict[str, str]] = []

        if matches[0].start() > 0:
            preface = normalized[:matches[0].start()].strip()
            if preface:
                sections.append({"title": "概述", "content": preface})

        for idx, match in enumerate(matches):
            raw_title = re.sub(r"\s+", " ", match.group(1)).strip()
            if re.search(r"\.\d+\s*$", raw_title):
                continue

            title = re.sub(r"\s*[（(]图\d+[）)]\s*$", "", raw_title)
            title = re.sub(r"\s*[（(](?:PIC|pic)[）)]\s*$", "", title)
            title = re.sub(r"\s+", " ", title).strip()
            if not title:
                continue

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
            body = normalized[start:end].strip()
            section_text = f"# {title}\n{body}".strip() if body else f"# {title}"
            sections.append({"title": title, "content": section_text})

        return sections

    @staticmethod
    def extract_image_ids(text: str) -> List[str]:
        return re.findall(r"<PIC>\[([^\]]+)\]", text or "")


class KnowledgeBaseBuilderV2:
    """V2 知识库构建器。"""

    GC_INTERVAL = 5
    DEFAULT_EXCLUDED_FILES = {"汇总英文手册.txt"}

    def __init__(self, manual_dir: Optional[str] = None, include_excluded_files: bool = False):
        self.manual_dir = Path(manual_dir) if manual_dir else PROJECT_ROOT / "手册new"
        self.include_excluded_files = include_excluded_files
        self.index_dir = PROJECT_ROOT / "knowledge_base" / "index_v2"
        self.progress_file = self.index_dir / "build_progress_v2.json"
        self.metadata_file = self.index_dir / "metadata_v2.json"
        self.summary_index_file = self.index_dir / "summary_index.faiss"
        self._rag_engine_v2 = None
        self._initialized = False
        self._total_files = 0
        self._processed_files = 0

    @property
    def rag_engine_v2(self):
        if self._rag_engine_v2 is None:
            self._rag_engine_v2 = get_rag_engine_v2()
        return self._rag_engine_v2

    def initialize(self):
        self.rag_engine_v2.initialize()
        self._initialized = True
        logger.info("RAG V2 引擎初始化完成")

    def _load_progress(self) -> Dict[str, Any]:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception:
                pass
        return {"processed_files": [], "total_docs": 0, "total_chars": 0}

    def _save_progress(self, progress: Dict[str, Any]):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w", encoding="utf-8") as file:
            json.dump(progress, file, ensure_ascii=False, indent=2)

    def _load_existing_documents(self) -> List[SummaryDocument]:
        if not self.metadata_file.exists():
            return []

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            raw_docs = metadata.get("texts", []) if isinstance(metadata, dict) else metadata
            return [
                SummaryDocument(
                    content=item.get("content", ""),
                    doc_id=item.get("doc_id", ""),
                    summary=item.get("summary", ""),
                    metadata=item.get("metadata", {}),
                )
                for item in raw_docs
            ]
        except Exception as exc:
            logger.warning(f"读取现有 V2 元数据失败: {exc}")
            return []

    def _save_metadata(self, documents: List[SummaryDocument]):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "texts": [doc.to_dict() for doc in documents],
            "embedding_backend": settings.embedding_backend,
            "summary_model": settings.summary_model,
        }
        with open(self.metadata_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _embed_summaries(self, summaries: List[str]) -> np.ndarray:
        model = self.rag_engine_v2.knowledge_base.embedding_model
        if hasattr(model, "embed_documents"):
            embeddings = model.embed_documents(summaries)
        elif hasattr(model, "encode"):
            embeddings = model.encode(summaries)
        else:
            raise ValueError("当前 embedding model 不支持批量编码")

        array = np.array(embeddings, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def _save_index(self, documents: List[SummaryDocument]):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        summaries = [doc.summary or doc.content[:180] for doc in documents]
        if not summaries:
            logger.warning("没有可保存的摘要文档，跳过索引写入")
            return

        embeddings = self._embed_summaries(summaries)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.summary_index_file))
        self._save_metadata(documents)
        logger.info(f"[V2] 已保存摘要索引: {len(documents)} 条 -> {self.summary_index_file}")

    def _get_manual_files(self) -> List[Path]:
        if not self.manual_dir.exists():
            raise FileNotFoundError(f"手册目录不存在: {self.manual_dir}")

        files = sorted(self.manual_dir.glob("*.txt"))
        if self.include_excluded_files:
            return files
        return [path for path in files if path.name not in self.DEFAULT_EXCLUDED_FILES]

    def _split_long_section(self, text: str, section_title: str) -> List[str]:
        max_chars = settings.summary_max_chunk_chars
        min_chars = settings.summary_min_chunk_chars
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        if len(cleaned) <= max_chars:
            return [cleaned]

        parts: List[str] = []
        buffer = ""
        delimiters = re.split(r"(?<=[。！？.!?；;])", cleaned)
        for piece in delimiters:
            piece = piece.strip()
            if not piece:
                continue
            if not buffer:
                buffer = piece
                continue
            if len(buffer) + len(piece) <= max_chars:
                buffer += piece
            else:
                parts.append(buffer.strip())
                buffer = piece

        if buffer:
            parts.append(buffer.strip())

        merged: List[str] = []
        for part in parts:
            if merged and len(part) < min_chars:
                merged[-1] = (merged[-1] + " " + part).strip()
            else:
                merged.append(part)
        return merged

    def _build_documents_for_manual(self, manual_file: Path) -> List[SummaryDocument]:
        raw = manual_file.read_text(encoding="utf-8")
        raw_text, image_ids = ManualParser.parse_manual_file(raw)
        injected = ManualParser.inject_image_ids(raw_text, image_ids)
        sections = ManualParser.extract_sections(injected)

        documents: List[SummaryDocument] = []
        for index, section in enumerate(sections):
            title = section.get("title", "")
            content = section.get("content", "").strip()
            if not content:
                continue

            for sub_index, chunk in enumerate(self._split_long_section(content, title)):
                summary = self.rag_engine_v2.summary_generator.generate_summary(chunk, title)
                doc_id = f"manual_{manual_file.stem}_s{index}"
                if len(self._split_long_section(content, title)) > 1:
                    doc_id = f"{doc_id}_c{sub_index}"
                metadata = {
                    "source_file": manual_file.name,
                    "manual_name": manual_file.stem,
                    "section_title": title,
                    "image_ids": ManualParser.extract_image_ids(chunk),
                }
                documents.append(
                    SummaryDocument(
                        content=chunk,
                        doc_id=doc_id,
                        summary=summary,
                        metadata=metadata,
                    )
                )
        return documents

    def build(self, force_rebuild: bool = False):
        logger.info("=" * 60)
        logger.info("开始构建知识库 V2（摘要索引版）...")
        logger.info(f"手册目录: {self.manual_dir}")
        logger.info(f"摘要索引目录: {self.index_dir}")
        logger.info("=" * 60)

        self.initialize()

        manual_files = self._get_manual_files()
        self._total_files = len(manual_files)
        progress = self._load_progress()
        processed_files = set(progress.get("processed_files", []))

        documents = [] if force_rebuild else self._load_existing_documents()
        existing_doc_ids: Set[str] = {doc.doc_id for doc in documents}

        if force_rebuild:
            processed_files.clear()
            documents = []
            existing_doc_ids.clear()

        for idx, manual_file in enumerate(manual_files, start=1):
            if not force_rebuild and manual_file.name in processed_files:
                logger.info(f"[{idx}/{self._total_files}] 跳过已处理手册: {manual_file.name}")
                continue

            logger.info(f"[{idx}/{self._total_files}] 处理手册: {manual_file.name}")
            try:
                new_docs = self._build_documents_for_manual(manual_file)
                new_docs = [doc for doc in new_docs if doc.doc_id not in existing_doc_ids]
                for doc in new_docs:
                    existing_doc_ids.add(doc.doc_id)
                documents.extend(new_docs)
                processed_files.add(manual_file.name)
                self._processed_files += 1

                progress = {
                    "processed_files": sorted(processed_files),
                    "total_docs": len(documents),
                    "total_chars": sum(len(doc.content) for doc in documents),
                }
                self._save_progress(progress)

                if idx % self.GC_INTERVAL == 0:
                    gc.collect()
            except Exception as exc:
                logger.exception(f"处理手册失败: {manual_file.name} -> {exc}")

        self._save_index(documents)

        logger.info("知识库 V2 构建完成!")
        logger.info(f"  - 处理手册数: {self._processed_files}/{self._total_files}")
        logger.info(f"  - 文档数: {len(documents)}")
        logger.info(f"  - 索引位置: {self.index_dir}")


if __name__ == "__main__":
    builder = KnowledgeBaseBuilderV2()
    builder.build(force_rebuild=True)
