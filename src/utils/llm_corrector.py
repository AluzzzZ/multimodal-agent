"""
拼写纠错模块（存根实现）。

职责：
1. 对用户查询文本进行拼写纠正，提升检索召回率。

当前为最小化存根，仅保证链路不报错。
后续可接入 jieba 纠错或调用小模型实现。
"""

from __future__ import annotations

from typing import Optional

from config import settings
from loguru import logger


class SpellCorrector:
    """
    拼写纠错器。

    当前实现为透传（pass-through），仅打印调试日志。
    后续应接入真实纠错逻辑。
    """

    def __init__(self):
        self.enabled = getattr(settings, "spell_correction_enabled", False)
        if not self.enabled:
            logger.debug("[SpellCorrector] 拼写纠错已禁用")

    def correct(self, text: str) -> str:
        """
        对输入文本进行拼写纠正。

        Args:
            text: 原始用户查询

        Returns:
            纠正后的文本（当前为透传）
        """
        if not self.enabled:
            return text
        if not text:
            return text
        # TODO: 接入真实纠错逻辑，例如：
        #   1. jieba 纠错
        #   2. 小模型（qwen2.5-0.5b-instruct）纠错
        # 当前直接返回原文
        return text


_corrector: Optional[SpellCorrector] = None


def get_spell_corrector() -> SpellCorrector:
    """获取全局拼写纠正器单例。"""
    global _corrector
    if _corrector is None:
        _corrector = SpellCorrector()
    return _corrector


def reset_spell_corrector() -> None:
    """重置全局拼写纠正器单例。"""
    global _corrector
    _corrector = None
