"""
多轮对话管理模块
负责管理会话历史、上下文维护

核心概念:
- 会话(Session): 一次完整的对话上下文,包含多条消息
- 消息(Message): 对话中的单条记录,可以是用户或助手
- 上下文(Context): 对话历史和当前状态的聚合

功能特性:
- 多会话管理: 支持同时维护多个用户会话
- 历史记录: 支持获取最近N轮对话作为上下文
- 会话超时: 自动标记超时会话为expired状态
- 用户索引: 支持按用户ID查询其所有会话
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from config import settings


@dataclass
class Message:
    """
    消息数据类 - 表示对话中的单条消息

    Attributes:
        role: 角色标识, "user"(用户) 或 "assistant"(助手)
        content: 消息文本内容
        images: 消息中包含的图片列表(Base64格式)
        timestamp: 消息时间戳
        metadata: 额外的元数据信息(如置信度、来源等)
    """
    role: str  # "user" or "assistant"
    content: str
    images: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式,便于序列化和API返回"""
        return {
            "role": self.role,
            "content": self.content,
            "images": self.images,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class ConversationContext:
    """
    对话上下文 - 管理单个会话的所有状态

    核心职责:
    - 存储会话消息历史
    - 追踪会话状态(active/completed/expired)
    - 管理用户信息
    - 自动裁剪超长历史

    会话超时机制:
    - 基于last_active时间戳判断
    - 超过session_timeout秒未活动视为过期
    """
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    user_info: Dict[str, Any] = field(default_factory=dict)
    state: str = "active"  # active, completed, expired

    def add_message(self, message: Message):
        """
        添加消息到会话

        自动更新last_active时间戳
        当消息数量超过max_conversation_history*2时,
        自动裁剪最早的半量消息,保留最近的上下文
        """
        self.messages.append(message)
        self.last_active = time.time()

        # 限制历史长度,避免内存溢出
        # 保留最近max_conversation_history*2条消息
        if len(self.messages) > settings.max_conversation_history * 2:
            self.messages = self.messages[-settings.max_conversation_history * 2:]

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取对话历史

        Args:
            limit: 可选,限制返回最近N条消息

        Returns:
            消息字典列表
        """
        # 如果指定limit,返回最近的limit条;否则返回全部
        messages = self.messages[-limit:] if limit else self.messages
        return [msg.to_dict() for msg in messages]

    def get_recent_context(self, last_n: int = 5) -> str:
        """
        获取最近N轮对话的文本摘要

        用于填充LLM的上下文窗口,保持对话连贯性

        Args:
            last_n: 最近的N条消息

        Returns:
            格式化的上下文字符串
        """
        recent = self.messages[-last_n:] if self.messages else []

        context_parts = []
        for msg in recent:
            role = "用户" if msg.role == "user" else "助手"
            context_parts.append(f"{role}: {msg.content}")

        return "\n".join(context_parts)


class ConversationManager:
    """
    对话管理器 - 全局会话状态管理器

    核心职责:
    - 多会话存储与查询
    - 会话生命周期管理(创建/获取/删除)
    - 用户-会话索引维护
    - 过期会话清理

    存储结构:
    - _conversations: {session_id: ConversationContext}  主存储
    - _user_sessions: {user_id: [session_ids]}  用户索引
    """

    def __init__(self):
        # 主会话存储: session_id -> ConversationContext
        self._conversations: Dict[str, ConversationContext] = {}
        # 用户会话索引: user_id -> [session_ids]
        # 一个用户可能有多个会话(跨设备/多话题)
        self._user_sessions: Dict[str, List[str]] = defaultdict(list)
        # 会话清理回调函数(可选)
        self._cleanup_callback: Optional[Callable] = None
        self._initialized = False

    def initialize(self):
        """
        初始化对话管理器

        当前实现为轻量级内存存储,
        生产环境可替换为Redis等持久化方案
        """
        logger.info("对话管理器初始化完成")
        self._initialized = True

    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        创建新会话

        为每个会话生成全局唯一UUID,
        并建立用户-会话索引(如果提供user_id)

        Args:
            user_id: 可选的用户标识

        Returns:
            新创建的session_id
        """
        # 生成全局唯一会话ID
        session_id = str(uuid.uuid4())

        # 创建会话上下文
        context = ConversationContext(session_id=session_id)

        # 建立用户索引(支持一个用户多个会话)
        if user_id:
            context.user_info["user_id"] = user_id
            self._user_sessions[user_id].append(session_id)

        # 存入主存储
        self._conversations[session_id] = context

        logger.info(f"创建新会话: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """
        获取会话上下文

        自动检查会话是否超时:
        - 如果超时,标记为expired状态
        - 如果正常,更新last_active

        Args:
            session_id: 会话ID

        Returns:
            ConversationContext 或 None(会话不存在)
        """
        context = self._conversations.get(session_id)

        if context:
            # 检查会话是否超时
            if time.time() - context.last_active > settings.session_timeout:
                context.state = "expired"
                logger.info(f"会话已过期: {session_id}")
            else:
                # 延长会话活跃时间
                context.last_active = time.time()

        return context

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        images: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加消息到指定会话

        Args:
            session_id: 目标会话ID
            role: 角色标识, "user" 或 "assistant"
            content: 消息文本
            images: 可选的图片列表
            metadata: 可选的元数据

        Returns:
            是否成功添加
        """
        context = self.get_session(session_id)

        if context is None:
            logger.warning(f"会话不存在: {session_id}")
            return False

        # 创建消息对象
        message = Message(
            role=role,
            content=content,
            images=images or [],
            metadata=metadata or {}
        )

        # 添加到会话
        context.add_message(message)

        logger.debug(f"添加消息到会话 {session_id}: {role}")
        return True

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取对话历史

        用于RAG检索时的上下文增强

        Args:
            session_id: 会话ID
            limit: 可选,限制返回条数

        Returns:
            消息字典列表
        """
        context = self.get_session(session_id)

        if context is None:
            return []

        return context.get_history(limit)

    def get_context_summary(self, session_id: str, last_n: int = 5) -> str:
        """
        获取上下文文本摘要

        用于填充LLM的上下文窗口

        Args:
            session_id: 会话ID
            last_n: 最近的N轮对话

        Returns:
            格式化的上下文字符串
        """
        context = self.get_session(session_id)

        if context is None:
            return ""

        return context.get_recent_context(last_n)

    def clear_session(self, session_id: str) -> bool:
        """
        删除指定会话

        Args:
            session_id: 要删除的会话ID

        Returns:
            是否成功删除
        """
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.info(f"清除会话: {session_id}")
            return True

        return False

    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """
        列出用户的所有会话

        Args:
            user_id: 可选,指定用户ID
                   如果为None,返回所有会话

        Returns:
            session_id列表
        """
        if user_id:
            return self._user_sessions.get(user_id, [])

        return list(self._conversations.keys())

    def cleanup_expired_sessions(self) -> int:
        """
        清理所有过期会话

        遍历所有会话,删除超过session_timeout未活动的会话

        Returns:
            清理的会话数量
        """
        current_time = time.time()
        expired_ids = []

        # 识别过期会话
        for session_id, context in self._conversations.items():
            if current_time - context.last_active > settings.session_timeout:
                expired_ids.append(session_id)

        # 删除过期会话
        for session_id in expired_ids:
            self.clear_session(session_id)

        if expired_ids:
            logger.info(f"清理了 {len(expired_ids)} 个过期会话")

        return len(expired_ids)

    def set_cleanup_callback(self, callback: Callable):
        """
        设置会话清理时的回调函数

        可用于触发持久化、日志记录等操作

        Args:
            callback: 清理时调用的回调函数
        """
        self._cleanup_callback = callback


class DialogueState:
    """
    对话状态跟踪器 - 管理多问题回答进度

    用于跟踪复杂多问题场景下:
    - 哪些问题已回答
    - 当前回答到哪个问题
    - 是否所有问题都已回答
    """

    def __init__(self):
        # 待回答的问题列表
        self.pending_questions: List[str] = []
        # 当前问题索引(0-based)
        self.current_question_index: int = 0
        # 已回答的问题列表
        self.answered_questions: List[str] = []
        # 是否为多问题场景
        self.is_multi_question: bool = False

    def set_questions(self, questions: List[str]):
        """
        设置待回答的问题列表

        Args:
            questions: 问题列表
        """
        self.pending_questions = questions
        self.is_multi_question = len(questions) > 1
        self.current_question_index = 0
        self.answered_questions = []

    def mark_answered(self, question: str):
        """
        标记问题已回答

        Args:
            question: 已回答的问题文本
        """
        if question in self.pending_questions:
            self.answered_questions.append(question)
            self.current_question_index += 1

    def get_next_question(self) -> Optional[str]:
        """
        获取下一个待回答的问题

        Returns:
            下一个问题,或None(所有问题已回答)
        """
        if self.current_question_index < len(self.pending_questions):
            return self.pending_questions[self.current_question_index]
        return None

    def is_complete(self) -> bool:
        """检查是否所有问题都已回答"""
        return len(self.answered_questions) >= len(self.pending_questions)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "pending_questions": self.pending_questions,
            "current_question_index": self.current_question_index,
            "answered_questions": self.answered_questions,
            "is_multi_question": self.is_multi_question,
            "is_complete": self.is_complete()
        }


# 全局单例实例 - 延迟初始化
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """
    获取对话管理器单例实例

    使用全局单例模式确保所有请求共享同一会话状态

    Returns:
        ConversationManager实例
    """
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
