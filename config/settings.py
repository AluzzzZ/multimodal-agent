"""
多模态客服智能体 - 配置文件
包含所有可配置的参数
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """应用配置"""
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "多模态客服智能体API"
    api_version: str = "1.0.0"
    
    # LLM配置
    llm_provider: str = "openai"  # openai, local, anthropic
    llm_model: str = "gpt-4-vision-preview"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    
    # Embedding配置
    embedding_backend: str = "sentence_transformer"  # hashing, sentence_transformer
    embedding_model: str = "BAAI/bge-m3"  # 中文+多语言最优: BAAI/bge-m3; 轻量备选: moka-ai/m3e-base; 旧模型: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    embedding_device: str = "cpu"  # cpu, cuda
    embedding_batch_size: int = 8
    embedding_dim: int = 1024  # BGE-M3为1024维; m3e-base为768维; MiniLM为384维

    # 多模态模型配置
    enable_vision_model: bool = False
    vision_model: str = "openai/clip-vit-large-patch14"
    vision_processor: str = "openai/clip-vit-large-patch14"
    
    chunk_size: int = 500
    chunk_overlap: int = 50

    # RAG配置（优化后）
    rag_top_k: int = 8  # 提高 top_k，增加候选给 reranker
    rag_score_threshold: float = 0.35  # 降低阈值，增加召回
    rag_rerank_candidate_k: int = 12  # rerank 候选上限，减少不必要的重排计算
    rag_enable_reranker: bool = True  # 启用BGE-M3配套的多语言重排序模型
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # 多语言交叉编码器，支持中文重排序

    # 知识库配置
    knowledge_base_path: Path = PROJECT_ROOT / "knowledge_base"
    index_path: Path = PROJECT_ROOT / "knowledge_base" / "index"
    text_index_file: str = "text_index.faiss"
    image_index_file: str = "image_index.faiss"
    metadata_file: str = "metadata.json"
    route_kb_path: Path = PROJECT_ROOT / "knowledge_base" / "routes"
    service_route_kb_file: str = "service_route_kb.json"
    service_policy_seed_path: Path = PROJECT_ROOT / "knowledge_base" / "seeds"
    service_policy_seed_file: str = "service_policy_seed.json"
    route_classifier_data_path: Path = PROJECT_ROOT / "knowledge_base" / "route_classifier"
    route_classifier_model_dir: Path = PROJECT_ROOT / "knowledge_base" / "route_classifier" / "model"

    # 双路检索配置
    route_service_top_k: int = 4
    route_manual_top_k: int = 5
    route_service_keyword_weight: float = 0.35
    route_example_similarity_weight: float = 0.65
    route_service_threshold: float = 0.38  # 降低阈值，让更多问题走 Service 路
    route_manual_threshold: float = 0.35  # 提高阈值，减少误路由到 Manual
    route_mixed_gap_threshold: float = 0.08  # 缩小差距，更容易触发 mixed
    mixed_manual_support_threshold: float = 0.68
    route_manual_candidate_top_k: int = 2
    route_manual_broad_top_k: int = 24
    route_classifier_enabled: bool = True
    route_classifier_backend: str = "onnx"
    route_classifier_high_threshold: float = 0.82
    route_classifier_low_threshold: float = 0.46
    route_classifier_use_image_tags: bool = True
    route_classifier_feature_dim: int = 512

    # 混合检索配置
    enable_hybrid_retrieval: bool = False  # 启用dense+sparse混合检索（BGE-M3内置支持）
    hybrid_sparse_weight: float = 0.3  # sparse分数权重，dense_weight = 1 - sparse_weight
    
    # 会话配置
    max_conversation_history: int = 10
    session_timeout: int = 3600  # 秒
    
    # 幻觉抑制配置
    hallucination_detection_enabled: bool = True
    hallucination_confidence_threshold: float = 0.7
    enable_cot_reasoning: bool = True
    
    # 图片配置
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    supported_image_formats: List[str] = ["jpg", "jpeg", "png", "webp", "bmp"]
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Path = PROJECT_ROOT / "logs" / "app.log"

    # 启动与资源配置
    eager_init_rag: bool = False
    eager_init_multimodal: bool = False
    eager_init_response_generator: bool = False

    # 评测配置
    evaluation_progress_interval: int = 20  # 评测进度打印间隔（题数）
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


def update_settings(**kwargs):
    """更新配置"""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
