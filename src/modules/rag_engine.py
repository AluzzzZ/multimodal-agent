"""
RAG（检索增强生成）引擎
负责从知识库中检索相关内容
支持增量构建以节省内存
"""

import json
import hashlib
import os
import re
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator
import numpy as np
import faiss
from loguru import logger

from config import settings


class Document:
    """
    文档数据类 - 表示知识库中的单个文档片段。

    文档由内容文本、唯一标识符和元数据三部分组成。元数据中通常包含
    手册名、章节标题、图片 ID 等信息，用于后续的路由和图片关联。

    Attributes:
        content: 文档内容文本（可能包含 <PIC> 图片占位符）
        doc_id: 文档唯一标识符
        metadata: 元数据字典（可包含 manual_name、section_title、image_ids 等）
    """

    def __init__(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.doc_id = doc_id
        self.metadata = metadata or {}

    def __repr__(self):
        """返回文档的可读字符串表示（调试用）"""
        return f"Document(id={self.doc_id}, content={self.content[:100]}...)"

    def to_dict(self) -> Dict[str, Any]:
        """
        将文档转换为字典格式。

        用于序列化存储和 API 返回。

        Returns:
            包含 content、doc_id、metadata 三个字段的字典
        """
        return {
            "content": self.content,
            "doc_id": self.doc_id,
            "metadata": self.metadata
        }


class KnowledgeBase:
    """知识库管理"""

    def __init__(self):
        self.text_embeddings: Optional[faiss.Index] = None
        self.image_embeddings: Optional[faiss.Index] = None
        self.text_documents: List[Document] = []
        self.image_documents: List[Document] = []
        self.embedding_model = None
        self.embedding_backend: Optional[str] = None
        self._initialized = False
        # 混合检索：BM25 参数预计算
        # _bm25_doc_tf: 每个文档的 {token: tf} 列表（与 documents 索引对应）
        # _bm25_avgdl: 平均文档长度（字符数）
        # _bm25_n_docs: 总文档数
        # _bm25_idf: 全局 IDF 字典 {token: idf}
        self._bm25_text_doc_tf: List[Dict[str, int]] = []
        self._bm25_text_avgdl: float = 0.0
        self._bm25_text_n_docs: int = 0
        self._bm25_text_idf: Dict[str, float] = {}
        self._bm25_image_doc_tf: List[Dict[str, int]] = []
        self._bm25_image_avgdl: float = 0.0
        self._bm25_image_n_docs: int = 0
        self._bm25_image_idf: Dict[str, float] = {}
        # 向后兼容旧格式（构建索引时自动迁移时清除）
        self._text_sparse_weights: List[Dict[str, float]] = []
        self._image_sparse_weights: List[Dict[str, float]] = []

    def initialize(self):
        """初始化知识库"""
        if self._initialized:
            return

        # 加载嵌入模型
        try:
            self.embedding_backend = settings.embedding_backend
            logger.info(f"加载嵌入后端: {self.embedding_backend}")
            self.embedding_model = self._create_embedding_model()
            logger.info("嵌入模型加载成功")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise

        # 尝试加载已有的索引
        self._load_index()

        self._initialized = True
        logger.info("知识库初始化完成")

    def _create_embedding_model(self):
        """根据配置创建嵌入模型。"""
        if settings.embedding_backend == "hashing":
            logger.info(
                f"使用轻量哈希嵌入后端: dim={settings.embedding_dim} "
                f"(适合低内存构建和Cursor终端环境)"
            )
            return HashingEmbeddingModel(settings.embedding_dim)

        if settings.embedding_backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            model_name = settings.embedding_model
            logger.info(f"加载嵌入模型: {model_name}")
            model = SentenceTransformer(model_name)

            # BGE-M3 推荐使用 mean pooling（sentence-transformers 默认已处理）
            # 若模型支持自动长文本切分（max_seq_length），encode时会自动处理
            if "bge-m3" in model_name.lower():
                logger.info(
                    "检测到 BGE-M3，将支持 dense+sparse 混合检索。"
                    "sparse_weight 通过 settings.hybrid_sparse_weight 配置。"
                )

            if settings.embedding_device == "cuda":
                model = model.to("cuda")

            # 动态更新 embedding_dim，防止配置与模型实际维度不一致
            # sentence-transformers >= 3.0 将 get_sentence_embedding_dimension 更名为 get_embedding_dimension
            try:
                actual_dim = model.get_embedding_dimension()
            except AttributeError:
                actual_dim = model.get_sentence_embedding_dimension()
            if actual_dim and actual_dim != settings.embedding_dim:
                logger.info(
                    f"模型实际维度 {actual_dim} 与配置 {settings.embedding_dim} 不一致，"
                    f"自动更新为 {actual_dim}"
                )
                settings.embedding_dim = actual_dim

            return model

        if settings.embedding_backend == "transformers":
            # 直接使用 transformers 库，避免 sentence_transformers 兼容性问题
            model_name = settings.embedding_model
            logger.info(f"加载 transformers 嵌入模型: {model_name}")
            model = TransformersEmbeddingModel(
                model_name=model_name,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                max_seq_length=settings.max_seq_length
            )

            # 延迟加载模型，获取实际维度
            actual_dim = model.get_sentence_embedding_dimension()
            if actual_dim != settings.embedding_dim:
                logger.info(
                    f"模型实际维度 {actual_dim} 与配置 {settings.embedding_dim} 不一致，"
                    f"自动更新为 {actual_dim}"
                )
                settings.embedding_dim = actual_dim

            return model

        if settings.embedding_backend == "dashscope":
            logger.info(
                f"使用百炼 dashscope embedding 后端: {settings.embedding_model} "
                f"(API: {settings.dashscope_base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1'})"
            )
            return DashScopeEmbeddingModel(
                model_name=settings.embedding_model,
                api_key=settings.dashscope_api_key,
                api_base=settings.dashscope_base_url,
                batch_size=settings.embedding_batch_size,
            )

        raise ValueError(f"不支持的 embedding_backend: {settings.embedding_backend}")

    def _load_index(self):
        """加载已有的索引文件"""
        index_dir = settings.index_path
        if not index_dir.exists():
            index_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建索引目录: {index_dir}")
            return

        # 加载文本索引
        text_index_file = index_dir / settings.text_index_file
        if text_index_file.exists():
            try:
                self.text_embeddings = faiss.read_index(str(text_index_file))
                logger.info(f"加载文本索引: {self.text_embeddings.ntotal} 个向量")
            except Exception as e:
                logger.warning(f"文本索引加载失败: {e}")

        # 加载图片索引
        image_index_file = index_dir / settings.image_index_file
        if image_index_file.exists():
            try:
                self.image_embeddings = faiss.read_index(str(image_index_file))
                logger.info(f"加载图片索引: {self.image_embeddings.ntotal} 个向量")
            except Exception as e:
                logger.warning(f"图片索引加载失败: {e}")

        # 加载元数据
        metadata_file = index_dir / settings.metadata_file
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.text_documents = [Document(**d) for d in metadata.get("texts", [])]
                    self.image_documents = [Document(**d) for d in metadata.get("images", [])]
                    saved_backend = metadata.get("embedding_backend")
                    if saved_backend and saved_backend != settings.embedding_backend:
                        logger.warning(
                            f"当前 embedding_backend={settings.embedding_backend} "
                            f"与索引记录的 {saved_backend} 不一致，必要时请重建索引"
                        )
                    # 加载 BM25 参数（用于混合检索）
                    if settings.enable_hybrid_retrieval:
                        self._bm25_text_doc_tf = metadata.get("bm25_text_doc_tf", [])
                        self._bm25_text_avgdl = metadata.get("bm25_text_avgdl", 0.0)
                        self._bm25_text_n_docs = metadata.get("bm25_text_n_docs", 0)
                        self._bm25_text_idf = metadata.get("bm25_text_idf", {})
                        self._bm25_image_doc_tf = metadata.get("bm25_image_doc_tf", [])
                        self._bm25_image_avgdl = metadata.get("bm25_image_avgdl", 0.0)
                        self._bm25_image_n_docs = metadata.get("bm25_image_n_docs", 0)
                        self._bm25_image_idf = metadata.get("bm25_image_idf", {})
                        logger.info(
                            f"加载 BM25 参数: {len(self._bm25_text_doc_tf)} 文本, "
                            f"{len(self._bm25_image_doc_tf)} 图片"
                        )
                    # 向后兼容旧格式 sparse_weights（构建新索引后自动迁移）
                    if not self._bm25_text_doc_tf:
                        self._text_sparse_weights = metadata.get("text_sparse_weights", [])
                        self._image_sparse_weights = metadata.get("image_sparse_weights", [])
                logger.info(f"加载元数据: {len(self.text_documents)} 文本, {len(self.image_documents)} 图片")
            except Exception as e:
                logger.warning(f"元数据加载失败: {e}")

    def add_documents(self, documents: List[Document], doc_type: str = "text"):
        """
        添加文档到知识库

        Args:
            documents: 文档列表
            doc_type: 文档类型 ("text" 或 "image")
        """
        if not self._initialized:
            self.initialize()

        if doc_type == "text":
            self._add_text_documents(documents)
        else:
            self._add_image_documents(documents)

    def _compute_bm25_params(
        self,
        texts: List[str],
    ) -> Tuple[List[Dict[str, int]], float, int, Dict[str, float]]:
        """
        计算并返回标准 BM25 的全局参数。

        预计算内容:
        1. 每个文档的 token->TF 字典列表（与 texts 索引对应）
        2. 平均文档长度（按字符数）
        3. 全局文档数
        4. 每个 token 的 IDF 值

        Args:
            texts: 文本列表

        Returns:
            (doc_tf_list, avgdl, n_docs, idf_dict)
        """
        import math
        from collections import Counter

        n_docs = len(texts)
        if n_docs == 0:
            return [], 0.0, 0, {}

        doc_tf_list: List[Dict[str, int]] = []
        doc_lens: List[int] = []
        df: Counter = Counter()

        for text in texts:
            tokens = self._tokenize_for_sparse(text)
            tf = Counter(tokens)
            doc_tf_list.append(dict(tf))
            doc_lens.append(len(text))
            for t in set(tokens):
                df[t] += 1

        avgdl = sum(doc_lens) / n_docs

        # 标准 IDF 公式: log((N - df + 0.5) / (df + 0.5))
        idf_dict: Dict[str, float] = {}
        for token, doc_freq in df.items():
            idf_dict[token] = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

        return doc_tf_list, avgdl, n_docs, idf_dict

    def _tokenize_for_sparse(self, text: str) -> List[str]:
        """分词：支持中文（jieba）和英文/数字。"""
        import jieba
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        tokens = jieba.lcut(cleaned.lower())
        return [t for t in tokens if t.strip() and len(t) > 1]

    def _compute_query_bm25_scores(
        self,
        query: str,
        doc_tf_list: List[Dict[str, int]],
        avgdl: float,
        idf_dict: Dict[str, float],
    ) -> List[float]:
        """
        计算查询在各文档上的标准 BM25 分数。

        BM25 公式:
            score = sum_{q in query_tokens} IDF(q) * (tf * (k1+1)) / (tf + k1*(1-b+b*|D|/avgdl))

        Args:
            query: 查询文本
            doc_tf_list: 预计算的文档 TF 字典列表
            avgdl: 平均文档长度
            idf_dict: 预计算的 IDF 字典

        Returns:
            每个文档的 BM25 分数列表
        """
        import jieba
        k1 = settings.bm25_k1
        b = settings.bm25_b

        query_tokens = jieba.lcut(query.lower())
        query_tokens = [t for t in query_tokens if t.strip() and len(t) > 1]

        scores: List[float] = []
        for doc_tf in doc_tf_list:
            score = 0.0
            for q in query_tokens:
                if q in doc_tf:
                    tf = doc_tf[q]
                    idf = idf_dict.get(q, 0.0)
                    doc_len = sum(doc_tf.values())
                    len_norm = k1 * (1 - b + b * doc_len / avgdl) if avgdl > 0 else k1
                    score += idf * (tf * (k1 + 1)) / (tf + len_norm)
            scores.append(score)
        return scores

    def _add_text_documents(self, documents: List[Document]):
        """添加文本文档"""
        if not documents:
            return

        # 生成嵌入
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # 归一化
        faiss.normalize_L2(embeddings)

        # 创建或更新索引
        dimension = embeddings.shape[1]

        if self.text_embeddings is None:
            self.text_embeddings = faiss.IndexFlatIP(dimension)

        self.text_embeddings.add(embeddings.astype('float32'))
        self.text_documents.extend(documents)

        # 预计算 BM25 参数（用于混合检索）
        if settings.enable_hybrid_retrieval:
            doc_tf, avgdl, n_docs, idf = self._compute_bm25_params(texts)
            # 存储原始文档频率（df），用于加载时统一计算 IDF
            # 这里顺便计算 IDF 并存下来（增量近似：对数域加权平均）
            if self._bm25_text_n_docs > 0:
                old_n = self._bm25_text_n_docs
                # 对已有 token 做 IDF 增量加权
                self._bm25_text_idf = {
                    t: (
                        self._bm25_text_idf.get(t, 0.0) * old_n
                        + idf.get(t, 0.0) * n_docs
                    ) / (old_n + n_docs)
                    for t in set(self._bm25_text_idf) | set(idf)
                }
                total_docs = old_n + n_docs
                self._bm25_text_avgdl = (self._bm25_text_avgdl * old_n + avgdl * n_docs) / total_docs
            else:
                self._bm25_text_idf = idf
                self._bm25_text_avgdl = avgdl
            self._bm25_text_n_docs += n_docs
            self._bm25_text_doc_tf.extend(doc_tf)

        logger.debug(f"添加 {len(documents)} 个文本文档到知识库 (总计: {self.text_embeddings.ntotal})")

    def _add_image_documents(self, documents: List[Document]):
        """添加图片文档"""
        if not documents:
            return

        # 图片使用描述文本生成嵌入
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        if self.image_embeddings is None:
            self.image_embeddings = faiss.IndexFlatIP(dimension)

        self.image_embeddings.add(embeddings.astype('float32'))
        self.image_documents.extend(documents)

        # 预计算 BM25 参数（用于混合检索）
        if settings.enable_hybrid_retrieval:
            doc_tf, avgdl, n_docs, idf = self._compute_bm25_params(texts)
            if self._bm25_image_n_docs > 0:
                old_n = self._bm25_image_n_docs
                self._bm25_image_idf = {
                    t: (
                        self._bm25_image_idf.get(t, 0.0) * old_n
                        + idf.get(t, 0.0) * n_docs
                    ) / (old_n + n_docs)
                    for t in set(self._bm25_image_idf) | set(idf)
                }
                total_docs = old_n + n_docs
                self._bm25_image_avgdl = (self._bm25_image_avgdl * old_n + avgdl * n_docs) / total_docs
            else:
                self._bm25_image_idf = idf
                self._bm25_image_avgdl = avgdl
            self._bm25_image_n_docs += n_docs
            self._bm25_image_doc_tf.extend(doc_tf)

        logger.debug(f"添加 {len(documents)} 个图片文档到知识库")

    def add_documents_incremental(
        self,
        documents: List[Document],
        doc_type: str = "text",
        save_after_batch: bool = True
    ):
        """
        增量添加文档（适合大批量导入，节省内存）

        Args:
            documents: 文档列表
            doc_type: 文档类型 ("text" 或 "image")
            save_after_batch: 每批次后是否保存索引
        """
        if not self._initialized:
            self.initialize()

        batch_size = settings.embedding_batch_size * 4  # 每批处理的文档数
        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            self.add_documents(batch, doc_type)

            # 定期释放内存
            if i % (batch_size * 4) == 0 and i > 0:
                gc.collect()
                logger.debug(f"增量处理进度: {min(i + batch_size, total)}/{total}")

        if save_after_batch:
            self.save_index()

    def save_index(self):
        """保存索引到磁盘"""
        index_dir = settings.index_path
        index_dir.mkdir(parents=True, exist_ok=True)

        # 保存文本索引
        if self.text_embeddings is not None:
            text_index_file = index_dir / settings.text_index_file
            faiss.write_index(self.text_embeddings, str(text_index_file))
            logger.info(f"保存文本索引: {text_index_file} ({self.text_embeddings.ntotal} 条)")

        # 保存图片索引
        if self.image_embeddings is not None:
            image_index_file = index_dir / settings.image_index_file
            faiss.write_index(self.image_embeddings, str(image_index_file))
            logger.info(f"保存图片索引: {image_index_file}")

        # 保存元数据
        metadata_file = index_dir / settings.metadata_file
        metadata = {
            "embedding_backend": settings.embedding_backend,
            "embedding_dim": settings.embedding_dim,  # 动态更新后的实际维度
            "enable_hybrid_retrieval": settings.enable_hybrid_retrieval,
            "texts": [doc.to_dict() for doc in self.text_documents],
            "images": [doc.to_dict() for doc in self.image_documents],
        }
        # 持久化 BM25 参数（用于混合检索）
        if settings.enable_hybrid_retrieval:
            # doc_tf 过大时，只存储 IDF 约简 token（权重 >= 1.0 的高频 token）
            metadata["bm25_text_doc_tf"] = self._bm25_text_doc_tf
            metadata["bm25_text_avgdl"] = self._bm25_text_avgdl
            metadata["bm25_text_n_docs"] = self._bm25_text_n_docs
            metadata["bm25_text_idf"] = self._bm25_text_idf
            metadata["bm25_image_doc_tf"] = self._bm25_image_doc_tf
            metadata["bm25_image_avgdl"] = self._bm25_image_avgdl
            metadata["bm25_image_n_docs"] = self._bm25_image_n_docs
            metadata["bm25_image_idf"] = self._bm25_image_idf
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"保存元数据: {metadata_file}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_type: str = "all"
    ) -> List[Tuple[Document, float]]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量
            doc_type: 文档类型 ("text", "image", "all")

        Returns:
            (文档, 相似度分数) 列表
        """
        if not self._initialized:
            self.initialize()

        if top_k is None:
            top_k = settings.rag_top_k

        results = []

        # 查询文本索引
        if doc_type in ("text", "all") and self.text_embeddings is not None:
            text_results = self._search_index(
                query,
                self.text_embeddings,
                self.text_documents,
                top_k,
                doc_tf=self._bm25_text_doc_tf if settings.enable_hybrid_retrieval else None,
                avgdl=self._bm25_text_avgdl if settings.enable_hybrid_retrieval else 0.0,
                idf=self._bm25_text_idf if settings.enable_hybrid_retrieval else None,
            )
            results.extend(text_results)

        # 查询图片索引
        if doc_type in ("image", "all") and self.image_embeddings is not None:
            image_results = self._search_index(
                query,
                self.image_embeddings,
                self.image_documents,
                top_k,
                doc_tf=self._bm25_image_doc_tf if settings.enable_hybrid_retrieval else None,
                avgdl=self._bm25_image_avgdl if settings.enable_hybrid_retrieval else 0.0,
                idf=self._bm25_image_idf if settings.enable_hybrid_retrieval else None,
            )
            results.extend(image_results)

        # 按分数排序并去重
        results = self._deduplicate_and_sort(results)

        return results[:top_k]

    def _search_index(
        self,
        query: str,
        index: faiss.Index,
        documents: List[Document],
        top_k: int,
        doc_tf: Optional[List[Dict[str, int]]] = None,
        avgdl: float = 0.0,
        idf: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        在单个 FAISS 索引中检索，支持 dense+BM25 混合模式。

        检索流程:
        1. 将查询文本编码为向量并归一化（dense）
        2. 在 FAISS 索引中搜索 top_k*2 个候选
        3. 若启用混合检索，用 BM25 公式计算 sparse 分数并加权融合
        4. 过滤低于分数阈值的文档
        5. 返回 (文档, 分数) 元组列表

        Args:
            query: 查询文本
            index: FAISS 索引对象
            documents: 与索引对应的文档列表
            top_k: 请求的返回数量
            doc_tf: 预计算的文档 token->TF 字典列表
            avgdl: 预计算的平均文档长度
            idf: 预计算的 token->IDF 字典

        Returns:
            符合条件的 (文档, 分数) 列表
        """
        if index.ntotal == 0:
            return []

        search_k = min(settings.rag_rerank_candidate_k, index.ntotal)

        # --- Dense 检索 ---
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores_np, indices_np = index.search(
            query_embedding.astype('float32'),
            search_k
        )
        scores_list = scores_np[0].tolist()
        indices_list = indices_np[0].tolist()

        # --- BM25 Sparse 分数 ---
        raw_bm25_scores: List[float] = [0.0] * len(indices_list)
        if settings.enable_hybrid_retrieval and doc_tf and idf and avgdl > 0:
            raw_bm25_scores = self._compute_query_bm25_scores(
                query, doc_tf, avgdl, idf
            )
            # BM25 分数范围可能很大，先归一化到 [0, 1]
            max_bm = max(raw_bm25_scores) if raw_bm25_scores else 1.0
            if max_bm > 0:
                raw_bm25_scores = [s / max_bm for s in raw_bm25_scores]

        # --- 分数融合 ---
        results = []
        alpha = 1.0 - settings.hybrid_sparse_weight  # dense 权重
        beta = settings.hybrid_sparse_weight          # sparse 权重

        for i, idx in enumerate(indices_list):
            dense_score = float(scores_list[i])
            bm25_score = raw_bm25_scores[idx] if (doc_tf and 0 <= idx < len(raw_bm25_scores)) else 0.0
            final_score = alpha * dense_score + beta * bm25_score

            if idx >= 0 and idx < len(documents) and final_score >= settings.rag_score_threshold:
                results.append((documents[idx], final_score))

        return results

    def _deduplicate_and_sort(
        self,
        results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        对检索结果去重并按相关性分数降序排列。

        去重策略: 按doc_id去重，保留首次出现的结果。
        由于初步检索时已返回top_k*2的候选，跨文本和图片索引
        可能出现同一文档的重复命中，需要在此步骤合并。

        Args:
            results: 初步检索结果列表[(文档, 分数)]

        Returns:
            去重并排序后的结果列表
        """
        seen_ids = set()
        unique_results = []

        for doc, score in results:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                unique_results.append((doc, score))

        # 按相关性分数降序排列
        return sorted(unique_results, key=lambda x: x[1], reverse=True)

    def clear(self):
        """清空知识库（用于重建）"""
        self.text_embeddings = None
        self.image_embeddings = None
        self.text_documents = []
        self.image_documents = []
        # 清空 BM25 参数
        self._bm25_text_doc_tf = []
        self._bm25_text_avgdl = 0.0
        self._bm25_text_n_docs = 0
        self._bm25_text_idf = {}
        self._bm25_image_doc_tf = []
        self._bm25_image_avgdl = 0.0
        self._bm25_image_n_docs = 0
        self._bm25_image_idf = {}
        # 向后兼容旧格式
        self._text_sparse_weights = []
        self._image_sparse_weights = []
        gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息（用于健康检查和调试）。

        Returns:
            包含 text_count、image_count、index_total、embedding_backend、
            embedding_dim、enable_hybrid_retrieval 的字典
        """
        return {
            "text_count": len(self.text_documents) if self.text_documents else 0,
            "image_count": len(self.image_documents) if self.image_documents else 0,
            "index_total": self.text_embeddings.ntotal if self.text_embeddings else 0,
            "embedding_backend": self.embedding_backend,
            "embedding_dim": settings.embedding_dim,
            "enable_hybrid_retrieval": settings.enable_hybrid_retrieval,
            "sparse_weights_count": len(self._text_sparse_weights),
        }


class Reranker:
    """重排序器 - 对检索结果进行精细排序"""

    def __init__(self):
        self.model = None
        self._initialized = False

    def initialize(self):
        """初始化重排序模型"""
        if self._initialized:
            return

        if not settings.rag_enable_reranker:
            self._initialized = True
            return

        if settings.reranker_backend == "dashscope":
            logger.info(
                f"使用百炼 dashscope reranker: {settings.reranker_model} "
                f"(API: {settings.dashscope_base_url or 'https://dashscope.aliyuncs.com'})"
            )
            self._initialized = True
            return

        if settings.reranker_backend == "cross_encoder":
            try:
                logger.info(f"加载 cross_encoder 重排序模型: {settings.reranker_model}")
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(settings.reranker_model)
                logger.info("cross_encoder 重排序模型加载成功")
            except Exception as e:
                logger.warning(f"cross_encoder 重排序模型加载失败: {e}")
                self.model = None
        else:
            logger.warning(f"不支持的 reranker_backend: {settings.reranker_backend}")

        self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: (文档, 原分数) 列表
            top_k: 返回数量

        Returns:
            (文档, 新分数) 列表
        """
        if not self._initialized:
            self.initialize()

        if not documents:
            return []

        if self.model is None and settings.reranker_backend != "dashscope":
            return documents[:top_k]

        try:
            if settings.reranker_backend == "dashscope":
                scores = self._rerank_dashscope(query, [doc for doc, _ in documents])
            else:
                sentence_pairs = [
                    (query, doc.content) for doc, _ in documents
                ]
                scores = self.model.predict(sentence_pairs)

            scored_documents = [
                (doc, float(score))
                for (doc, _), score in zip(documents, scores)
            ]
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            return scored_documents[:top_k]

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]

    def _rerank_dashscope(
        self,
        query: str,
        documents: List[Document],
    ) -> List[float]:
        """
        通过百炼 rerank API 计算 query 与各文档的相关性分数。

        百炼 rerank API 支持批量传入 (query, document) 对，返回相关性分数列表。
        分数越高表示相关性越强。
        """
        import os
        import requests

        api_key = self._get_api_key()
        api_base = settings.dashscope_base_url or "https://dashscope.aliyuncs.com"
        endpoint = f"{api_base.rstrip('/')}/api/v1/services/rerank/text-rerank/text-rerank"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": settings.reranker_model,
            "input": {
                "query": {"text": query},
                "documents": [{"text": doc.content} for doc in documents],
            },
            "parameters": {
                "top_n": len(documents),
                "return_documents": False,
            },
        }

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        if not resp.ok:
            logger.error(f"百炼 rerank 请求失败 [{resp.status_code}]: {resp.text}")
            resp.raise_for_status()
        data = resp.json()

        scores = [0.0] * len(documents)
        for result in data.get("output", {}).get("results", []):
            doc_idx = int(result.get("index", result.get("document_index", -1)))
            if 0 <= doc_idx < len(documents):
                scores[doc_idx] = float(result.get("relevance_score", 0.0))

        if not scores and "error" in data:
            raise RuntimeError(f"百炼 rerank API 错误: {data.get('error')}")

        return scores

    @staticmethod
    def _get_api_key() -> str:
        """从 settings 或环境变量获取 API key。"""
        key = settings.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not key:
            raise ValueError("未设置 DASHSCOPE_API_KEY，请检查 .env 配置。")
        return key


class RAGEngine:
    """RAG引擎主类"""

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.reranker = Reranker()
        self._initialized = False

    def initialize(self):
        """初始化RAG引擎（不包括reranker，按需初始化）"""
        self.knowledge_base.initialize()
        # Reranker 延迟到第一次检索时才加载，避免构建知识库时占用额外内存
        self._initialized = True
        logger.info("RAG引擎初始化完成（reranker 将延迟加载）")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        检索相关内容

        Args:
            query: 查询文本
            top_k: 返回数量
            use_rerank: 是否使用重排序

        Returns:
            检索结果列表
        """
        if not self._initialized:
            self.initialize()

        if top_k is None:
            top_k = settings.rag_top_k

        # 初步检索：候选数由 rag_rerank_candidate_k 控制
        results = self.knowledge_base.retrieve(query, settings.rag_rerank_candidate_k if use_rerank else top_k)

        # 重排序（首次使用时延迟加载reranker）
        if use_rerank:
            results = self.reranker.rerank(query, results, top_k)

        # 格式化输出
        formatted_results = []
        for doc, score in results:
            result = {
                "content": doc.content,
                "doc_id": doc.doc_id,
                "relevance_score": score,
                "metadata": doc.metadata
            }

            # 处理图片ID
            if "<PIC>" in doc.content:
                result["has_image"] = True
                # 提取图片ID
                import re
                image_ids = re.findall(r'\[([^\]]+)\]', doc.content)
                result["image_ids"] = image_ids
            else:
                result["has_image"] = False
                result["image_ids"] = []

            formatted_results.append(result)

        return formatted_results

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        doc_type: str = "text"
    ):
        """
        添加文档到知识库

        Args:
            documents: 文档列表，每个文档包含 content, doc_id, metadata
            doc_type: 文档类型
        """
        if not self._initialized:
            self.initialize()

        docs = [
            Document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc.get("metadata", {})
            )
            for doc in documents
        ]

        self.knowledge_base.add_documents(docs, doc_type)

    def add_documents_incremental(
        self,
        documents: List[Dict[str, Any]],
        doc_type: str = "text"
    ):
        """
        增量添加文档（节省内存）

        Args:
            documents: 文档列表，每个文档包含 content, doc_id, metadata
            doc_type: 文档类型
        """
        if not self._initialized:
            self.initialize()

        docs = [
            Document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc.get("metadata", {})
            )
            for doc in documents
        ]

        self.knowledge_base.add_documents_incremental(docs, doc_type)

    def save_knowledge_base(self):
        """保存知识库"""
        if not self._initialized:
            self.initialize()

        self.knowledge_base.save_index()
        logger.info("知识库保存完成")

    def rebuild(self):
        """清空并重建知识库"""
        self.knowledge_base.clear()
        logger.info("知识库已清空，准备重建")

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.knowledge_base.get_stats()


class HashingEmbeddingModel:
    """超轻量文本哈希嵌入。

    用于低内存环境下的知识库构建与检索，避免导入 torch / transformers。
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        del batch_size, show_progress_bar
        vectors = [self._encode_text(text) for text in texts]
        matrix = np.vstack(vectors).astype("float32")
        return matrix if convert_to_numpy else matrix.tolist()

    def _encode_text(self, text: str) -> np.ndarray:
        """
        将文本编码为固定维度的哈希向量。

        算法原理(基于Sign Random Projection):
        1. 将文本分词得到token集合
        2. 对每个token计算MD5哈希值，取前4字节作为索引(模dim)
        3. 取第5字节的奇偶性决定正负符号(+1/-1)
        4. 在向量对应索引位置累加符号值
        5. 最后对向量做L2归一化

        该算法与torch/transformers无关，仅需hashlib，
        适合Cursor终端等低内存、无CUDA的环境。

        理论上: 相似文本会共享更多token，从而在向量空间中距离更近。
        向量归一化后可用内积(IP)直接计算余弦相似度。

        Args:
            text: 输入文本

        Returns:
            维度为self.dim的归一化向量(numpy数组)
        """
        tokens = self._tokenize(text)
        vector = np.zeros(self.dim, dtype=np.float32)

        if not tokens:
            return vector

        for token in tokens:
            # MD5哈希前4字节转整数作为向量索引（模dim避免越界）
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            # 第5字节奇偶性决定正负符号
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            # 累加到向量（多个token可能映射到同一索引）
            vector[index] += sign

        # L2归一化，使向量可直接用于余弦相似度计算
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector

    def get_sentence_embedding_dimension(self) -> int:
        """返回嵌入向量维度，供 KnowledgeBase 动态感知。"""
        return self.dim

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = (text or "").strip().lower()
        if not cleaned:
            return []

        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", cleaned)


class TransformersEmbeddingModel:
    """
    基于 transformers 库的嵌入模型。

    直接使用 transformers.AutoModel 进行编码，兼容 sentence-transformers 模型。
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        batch_size: int = 8,
        max_seq_length: int = 256
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._tokenizer = None
        self._model = None
        self._dim = None

    def _lazy_init(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import AutoTokenizer, AutoModel
        import torch

        logger.info(f"加载 transformers 模型: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)

        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to("cuda")

        self._model.eval()
        self._dim = self._model.config.hidden_size
        logger.info(f"模型加载完成，维度: {self._dim}")

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """Mean pooling - 取 token embeddings 的加权平均"""
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """将文本列表编码为嵌入向量"""
        self._lazy_init()

        import torch
        from tqdm import tqdm

        if batch_size is None:
            batch_size = self.batch_size

        all_embeddings = []
        texts = [t or "" for t in texts]

        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for i in iterator:
                batch = texts[i:i + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )

                if self.device == "cuda":
                    encoded = {k: v.cuda() for k, v in encoded.items()}

                model_output = self._model(**encoded)
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

                if self.device == "cuda":
                    embeddings = embeddings.cpu()

                all_embeddings.append(embeddings.numpy())

        return np.vstack(all_embeddings).astype("float32")

    def get_sentence_embedding_dimension(self) -> int:
        """返回嵌入向量维度"""
        self._lazy_init()
        return self._dim


class DashScopeEmbeddingModel:
    """
    百炼 dashscope API 嵌入模型。

    通过 HTTP API 调用 dashscope text-embedding-v3，避免本地加载模型。
    支持批量请求，自动分批避免超限。
    """

    def __init__(
        self,
        model_name: str = "text-embedding-v3",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.batch_size = batch_size
        self._dim: Optional[int] = None

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        if batch_size is None:
            batch_size = self.batch_size

        texts = [t or "" for t in texts]
        all_embeddings: List[np.ndarray] = []

        import os
        import requests
        from tqdm import tqdm

        headers = {
            "Authorization": f"Bearer {self.api_key or os.environ.get('DASHSCOPE_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.api_base.rstrip('/')}/compatible-mode/v1/embeddings"

        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Embedding")

        for start in iterator:
            batch = texts[start: start + batch_size]
            # 百炼 text-embedding-v3 批量上限为 10 条/请求
            api_batch_size = 10
            if len(batch) > api_batch_size:
                # 进一步拆分超出限制的部分
                for sub_start in range(0, len(batch), api_batch_size):
                    sub_batch = batch[sub_start: sub_start + api_batch_size]
                    self._call_embedding_api(endpoint, headers, sub_batch, all_embeddings)
            else:
                self._call_embedding_api(endpoint, headers, batch, all_embeddings)

        result = np.vstack(all_embeddings).astype("float32")
        return result

    def _call_embedding_api(
        self,
        endpoint: str,
        headers: Dict[str, str],
        batch: List[str],
        all_embeddings: List[np.ndarray],
    ):
        """调用百炼 embedding API 并追加结果。"""
        import requests
        payload = {
            "model": self.model_name,
            "input": batch,
        }
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        for item in data["data"]:
            vec = np.array(item["embedding"], dtype="float32")
            if self._dim is None:
                self._dim = len(vec)
            all_embeddings.append(vec)

    def get_sentence_embedding_dimension(self) -> int:
        """返回嵌入向量维度，首次请求后自动推断。"""
        if self._dim is None:
            raise RuntimeError(
                "维度未确定，请先调用 encode() 一次以自动推断维度。"
                "百炼 API 需实际请求后才知道输出维度。"
            )
        return self._dim


# 全局实例
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """获取RAG引擎实例"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def reset_rag_engine():
    """重置RAG引擎（用于重新初始化）"""
    global _rag_engine
    _rag_engine = None
