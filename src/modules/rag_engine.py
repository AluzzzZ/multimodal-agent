"""
RAG（检索增强生成）引擎
负责从知识库中检索相关内容
支持增量构建以节省内存
"""

import json
import hashlib
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
        # 混合检索：预计算的 sparse 权重（token -> 权重映射），格式为 List[Dict[str, float]]
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
                    # 加载 sparse 权重
                    if settings.enable_hybrid_retrieval:
                        self._text_sparse_weights = metadata.get("text_sparse_weights", [])
                        self._image_sparse_weights = metadata.get("image_sparse_weights", [])
                        logger.info(
                            f"加载 sparse 权重: {len(self._text_sparse_weights)} 文本, "
                            f"{len(self._image_sparse_weights)} 图片"
                        )
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

    def _compute_sparse_weights(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        计算文本集合的 sparse（词权重）表示，模拟 BM25 思想。

        算法：对每个文本分词，统计 token 频率（TF），并用归一化频率作为权重。
        检索时对查询同样分词，计算与文档的交集 token 权重和作为 sparse 分数。

        这使得 BGE-M3 等模型在不具备原生 sparse 输出时，
        也能与 dense 分数加权融合，提升关键词命中场景（如"退货""退款"）的召回。

        Args:
            texts: 文本列表

        Returns:
            每个文本的 token->权重 字典列表
        """
        from collections import Counter
        import math

        # 统计全局文档频率（IDF 近似）
        all_tokens_per_doc: List[List[str]] = []
        doc_token_sets: List[set] = []
        for text in texts:
            tokens = self._tokenize_for_sparse(text)
            all_tokens_per_doc.append(tokens)
            doc_token_sets.append(set(tokens))

        # 计算 IDF（简化版：log(N / df)）
        n_docs = len(texts)
        df: Counter = Counter()
        for token_set in doc_token_sets:
            for t in token_set:
                df[t] += 1

        weights_list: List[Dict[str, float]] = []
        for tokens in all_tokens_per_doc:
            tf = Counter(tokens)
            doc_len = len(tokens)
            weights: Dict[str, float] = {}
            for token, freq in tf.items():
                # TF-IDF 简化权重
                idf = math.log((n_docs + 1) / (df[token] + 1)) + 1
                # 归一化 TF（防止长文档主导）
                norm_tf = freq / math.sqrt(doc_len) if doc_len > 0 else 0
                weights[token] = norm_tf * idf
            weights_list.append(weights)

        return weights_list

    def _tokenize_for_sparse(self, text: str) -> List[str]:
        """分词：支持中文（jieba）和英文/数字。"""
        import jieba
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        tokens = jieba.lcut(cleaned.lower())
        return [t for t in tokens if t.strip() and len(t) > 1]

    def _compute_query_sparse_scores(
        self,
        query: str,
        documents: List[Document],
        sparse_weights: List[Dict[str, float]]
    ) -> List[float]:
        """
        计算查询在各文档上的 sparse 分数。

        分数 = 查询 token 在文档权重字典中的权重之和（交集权重累加）。
        """
        import jieba
        query_tokens = set(self._tokenize_for_sparse(query))
        scores: List[float] = []
        for weights in sparse_weights:
            score = sum(weights.get(t, 0.0) for t in query_tokens)
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

        # 预计算 sparse 权重（用于混合检索）
        if settings.enable_hybrid_retrieval:
            sparse_weights = self._compute_sparse_weights(texts)
            self._text_sparse_weights.extend(sparse_weights)

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

        # 预计算 sparse 权重
        if settings.enable_hybrid_retrieval:
            sparse_weights = self._compute_sparse_weights(texts)
            self._image_sparse_weights.extend(sparse_weights)

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
            # 持久化 sparse 权重（仅保留高权重 token，节省空间）
            "text_sparse_weights": [
                {k: v for k, v in w.items() if v > 0.1}
                for w in self._text_sparse_weights
            ] if settings.enable_hybrid_retrieval else [],
            "image_sparse_weights": [
                {k: v for k, v in w.items() if v > 0.1}
                for w in self._image_sparse_weights
            ] if settings.enable_hybrid_retrieval else [],
        }
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
                sparse_weights=self._text_sparse_weights if settings.enable_hybrid_retrieval else None
            )
            results.extend(text_results)

        # 查询图片索引
        if doc_type in ("image", "all") and self.image_embeddings is not None:
            image_results = self._search_index(
                query,
                self.image_embeddings,
                self.image_documents,
                top_k,
                sparse_weights=self._image_sparse_weights if settings.enable_hybrid_retrieval else None
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
        sparse_weights: Optional[List[Dict[str, float]]] = None
    ) -> List[Tuple[Document, float]]:
        """
        在单个FAISS索引中检索最相关的文档，支持dense+sparse混合模式。

        检索流程:
        1. 将查询文本编码为向量并归一化（dense）
        2. 在FAISS索引中搜索top_k*2个候选
        3. 若启用混合检索，计算sparse分数并加权融合
        4. 过滤低于分数阈值的文档
        5. 返回(文档, 分数)元组列表

        Args:
            query: 查询文本
            index: FAISS索引对象
            documents: 与索引对应的文档列表
            top_k: 请求的返回数量
            sparse_weights: 预计算的文档sparse权重列表（可选）

        Returns:
            符合条件的(文档, 分数)列表
        """
        if index.ntotal == 0:
            return []

        # 搜索: 多取候选给 reranker，上限由 rag_rerank_candidate_k 控制
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

        # --- Sparse 分数（可选） ---
        sparse_scores: List[float] = [0.0] * len(scores_list)
        if settings.enable_hybrid_retrieval and sparse_weights:
            raw_sparse = self._compute_query_sparse_scores(query, documents, sparse_weights)
            # Min-Max 归一化到 [0, 1]
            max_s = max(raw_sparse) if raw_sparse else 1.0
            if max_s > 0:
                sparse_scores = [s / max_s for s in raw_sparse]

        # --- 分数融合 ---
        results = []
        alpha = 1.0 - settings.hybrid_sparse_weight  # dense 权重
        beta = settings.hybrid_sparse_weight          # sparse 权重

        for i, idx in enumerate(indices_list):
            dense_score = float(scores_list[i])
            # 修复：按真实文档索引（idx）取 sparse 分数，而非按候选位置（i）
            sparse_score = sparse_scores[idx] if sparse_weights and 0 <= idx < len(sparse_scores) else 0.0

            # 混合分数：alpha * dense + beta * sparse
            final_score = alpha * dense_score + beta * sparse_score

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

        try:
            logger.info(f"加载重排序模型: {settings.reranker_model}")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(settings.reranker_model)
            logger.info("重排序模型加载成功")
        except Exception as e:
            logger.warning(f"重排序模型加载失败: {e}")
            self.model = None

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

        if self.model is None or not documents:
            return documents[:top_k]

        try:
            # 准备句子对
            sentence_pairs = [
                (query, doc.content) for doc, _ in documents
            ]

            # 获取重排序分数
            scores = self.model.predict(sentence_pairs)

            # 组合并排序
            scored_documents = [
                (doc, float(score))
                for (doc, _), score in zip(documents, scores)
            ]

            scored_documents.sort(key=lambda x: x[1], reverse=True)

            return scored_documents[:top_k]

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]


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
