"""
src 模块初始化

保持极简，避免离线脚本仅导入 `src.modules.rag_engine` 时，
被 `from .modules import *` 连带拉起整套重型依赖。
"""

__version__ = "1.0.0"
__author__ = "Multimodal Agent Team"

__all__ = ["__version__", "__author__"]
