"""
多模态客服智能体 - 配置文件
包含所有可配置的参数
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
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
    api_token: Optional[str] = Field(default=None, validation_alias="API_TOKEN")
    
    # LLM配置
    llm_provider: str = "openai"  # openai, local, anthropic
    llm_model: str = "deepseek-v4-pro"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = "https://cloud.infini-ai.com/maas/v1"
    
    # Embedding配置（支持本地模型或云端API）
    embedding_backend: str = "dashscope"  # dashscope | transformers | hashing | sentence_transformer
    embedding_model: str = "text-embedding-v3"  # dashscope: text-embedding-v3 | 本地: models/bge-m3
    embedding_api_key: Optional[str] = None  # 百炼 API Key
    embedding_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_device: str = "cpu"  # cpu, cuda（仅本地模型使用）
    embedding_batch_size: int = 10
    embedding_dim: int = 1024  # text-embedding-v3 为 1024 维
    max_seq_length: int = 2048  # 最大序列长度

    # 视觉大模型配置（外部 API）
    vision_llm_provider: str = "openai"  # openai, anthropic, google
    vision_llm_api_key: Optional[str] = None
    vision_llm_base_url: Optional[str] = "https://cloud.infini-ai.com/maas/v1"
    vision_llm_model: str = "qwen3-vl-235b-a22b-thinking"  # 视觉理解模型
    vision_llm_enabled: bool = True  # 启用视觉大模型 API

    # VLM 结构化理解配置
    multimodal_backend: str = "hybrid"  # rule | vlm | hybrid
    # rule: 只用规则链（原有逻辑）
    # vlm: 只用 VLM 结构化输出
    # hybrid: VLM 优先，失败回退规则链
    multimodal_hybrid_stage: str = "product_candidates"  # observe | product_candidates | image_tags | evidence_type | full
    # observe: 只记录 VLM 结果，不影响主链
    # product_candidates: 仅让 VLM 影响候选产品
    # image_tags: 让 VLM 影响候选产品 + 图片标签 + 视觉意图
    # evidence_type: 再让 VLM 影响证据类型
    # full: VLM 结果作为主输出，规则做补全

    vlm_max_images: int = 1          # VLM 单次调用最多图片数
    vlm_timeout_seconds: float = 10.0 # VLM API 超时（秒）
    vlm_max_tokens: int = 512        # VLM 输出最大 token 数
    vlm_temperature: float = 0.1      # VLM temperature（低以保证格式稳定）
    vlm_debug_log: bool = False      # VLM raw output 写日志（默认关闭以省日志量）
    
    chunk_size: int = 500
    chunk_overlap: int = 50

    # RAG配置（优化后）
    rag_top_k: int = 5  # 检索候选数
    rag_score_threshold: float = 0.35
    rag_rerank_candidate_k: int = 8  # rerank 候选上限
    rag_enable_reranker: bool = True

    # Reranker配置（支持本地模型或云端API）
    reranker_backend: str = "dashscope"  # dashscope | local
    reranker_model: str = "qwen3-vl-rerank"  # dashscope: qwen3-vl-rerank | local: models/bge-reranker-v2-m3
    reranker_api_key: Optional[str] = None  # 百炼 API Key
    reranker_api_base: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank"

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
    route_service_threshold: float = 0.30  # 降低阈值，让更多问题走 Service 路
    route_manual_threshold: float = 0.35  # 提高阈值，减少误路由到 Manual
    route_mixed_gap_threshold: float = 0.08  # 缩小差距，更容易触发 mixed
    mixed_manual_support_threshold: float = 0.68
    route_manual_candidate_top_k: int = 2
    route_manual_broad_top_k: int = 24
    route_classifier_enabled: bool = True
    route_classifier_backend: str = "llm"  # llm | onnx | rule（llm 使用 LLM API 判断）
    route_classifier_high_threshold: float = 0.82
    route_classifier_low_threshold: float = 0.46
    route_classifier_use_image_tags: bool = True
    route_classifier_feature_dim: int = 512

    # Manual 路由局部召回配置
    route_manual_local_recall_enabled: bool = True  # 启用局部召回：在候选手册内做向量检索
    route_manual_local_recall_min_score: float = 0.18  # 局部召回最低触发分数
    route_manual_local_recall_strong_score: float = 0.72  # 强触发分数（alias得分 >= 此值时跳过 broad 检索）
    route_manual_local_recall_min_gap: float = 0.30  # 触发 gap（alias 最高分与次高分的最小差值）
    route_manual_local_recall_max_manuals: int = 3  # 最多参与局部召回的手册数

    # 路由分类器置信度边距（用于宽松/严格模式）
    route_classifier_low_margin: float = 0.08  # 低置信度边界
    route_classifier_high_margin: float = 0.12  # 高置信度边界

    # 混合检索配置
    enable_hybrid_retrieval: bool = True  # 启用dense+sparse混合检索（BGE-M3内置支持）
    hybrid_sparse_weight: float = 0.3  # sparse分数权重，dense_weight = 1 - sparse_weight

    # 拼写纠错配置（用于 dual_route_retriever 检索前）
    spell_correction_enabled: bool = False
    
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
