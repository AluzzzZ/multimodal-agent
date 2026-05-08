"""
文本处理工具
"""

import re
import unicodedata
from typing import List, Tuple, Optional
from loguru import logger


class TextProcessor:
    """文本处理器"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除控制字符
        text = ''.join(
            char for char in text
            if unicodedata.category(char)[0] != 'C' or char in '\n\t'
        )
        
        # 规范化
        text = unicodedata.normalize('NFKC', text)
        
        return text.strip()
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        分句
        
        Args:
            text: 文本
            
        Returns:
            句子列表
        """
        # 按常见分隔符分句
        pattern = r'[。！？\n]+'
        sentences = re.split(pattern, text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 文本
            top_n: 返回数量
            
        Returns:
            关键词列表
        """
        # 简单实现：基于词频
        # 移除停用词
        stopwords = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '么'
        }
        
        # 提取中文词（简单按字符）
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        
        # 统计词频
        word_freq = {}
        for word in words:
            if len(word) >= 2 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 排序
        sorted_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [w for w, _ in sorted_words[:top_n]]
    
    @staticmethod
    def truncate_text(
        text: str,
        max_length: int = 500,
        suffix: str = "..."
    ) -> str:
        """
        截断文本
        
        Args:
            text: 文本
            max_length: 最大长度
            suffix: 后缀
            
        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """
        提取数字
        
        Args:
            text: 文本
            
        Returns:
            数字列表
        """
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        
        return [float(m) for m in matches if m]
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        规范化空白字符
        
        Args:
            text: 文本
            
        Returns:
            规范化后的文本
        """
        # 将所有空白字符替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class QueryProcessor:
    """查询处理器"""

    ENGLISH_PHRASE_MAP = {
        "first time to use": "首次使用",
        "before first use": "首次使用前",
        "approval label": "批准标签",
        "emission control certificate": "排放控制证书",
        "battery conversion feature": "电池转换功能",
        "over temperature warning": "过热警告",
        "sound system": "音响系统",
        "storage compartments": "储物舱",
        "wet items": "湿物品",
        "battery compartment": "电池舱",
        "anchor light": "锚灯",
        "jet wash": "冲洗功能",
        "water supply button": "供水按钮",
        "bimini top": "遮阳篷",
        "upright position": "竖立位置",
        "engine oil level": "发动机机油油位",
        "bilge pump": "舱底泵",
        "cooling system": "冷却系统",
        "energy saving mode": "节能模式",
        "factory settings": "出厂设置",
        "coffee machine": "咖啡机",
        "coffee maker": "咖啡机",
        "airfryer": "空气炸锅",
        "air fryer": "空气炸锅",
        "boat": "摩托艇",
        "ship": "摩托艇",
        "sailing": "航行",
        "cross wakes and swells": "穿越尾流和涌浪",
        "turn on or off": "打开或关闭",
    }

    ENGLISH_TOKEN_MAP = {
        "how": "如何",
        "what": "什么",
        "when": "当",
        "before": "之前",
        "after": "之后",
        "use": "使用",
        "install": "安装",
        "remove": "拆卸",
        "store": "存放",
        "find": "找到",
        "check": "检查",
        "replace": "更换",
        "flush": "冲洗",
        "start": "启动",
        "turn": "转向",
        "load": "装载",
        "open": "打开",
        "close": "关闭",
        "listen": "收听",
        "music": "音乐",
        "phone": "手机",
        "engine": "发动机",
        "fuse": "保险丝",
        "warning": "警告",
        "label": "标签",
        "sailing": "航行",
        "machine": "机器",
        "system": "系统",
        "water": "水",
        "mode": "模式",
        "default": "默认",
        "setting": "设置",
        "reset": "重置",
        "clean": "清洁",
        "maintenance": "维护",
    }

    ENGLISH_STOPWORDS = {
        "a", "an", "the", "of", "to", "is", "are", "if", "i", "my", "on", "in",
        "at", "for", "with", "and", "or", "while", "before", "after", "when",
        "do", "does", "did", "can", "could", "should", "would", "want",
    }
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """
        扩展查询（同义词扩展）
        
        Args:
            query: 原始查询
            
        Returns:
            扩展后的查询列表
        """
        # 简单的同义词映射
        synonyms = {
            '故障': ['问题', '坏了', '出错', '异常'],
            '维修': ['修理', '检修', '维护'],
            '使用': ['用法', '怎么用', '如何操作'],
            '充电': ['供电', '电池'],
            '指示灯': ['灯', '显示灯']
        }
        
        expanded = [query]
        
        for word, syns in synonyms.items():
            if word in query:
                for syn in syns:
                    new_query = query.replace(word, syn)
                    if new_query not in expanded:
                        expanded.append(new_query)
        
        return expanded
    
    @staticmethod
    def extract_entities(query: str) -> dict:
        """
        提取实体
        
        Args:
            query: 查询文本
            
        Returns:
            实体字典
        """
        entities = {
            "product_models": [],  # 产品型号
            "problems": [],        # 问题描述
            "actions": []          # 操作意图
        }
        
        # 提取型号（常见模式）
        model_patterns = [
            r'DCB\d{3}',  # DCB107, DCB112等
            r'[A-Z]{2,}\d{2,}',  # 其他型号格式
            r'[A-Z]\d{3}[A-Z]?'   # 变体格式
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, query)
            entities["product_models"].extend(matches)
        
        # 提取问题关键词
        problem_keywords = ['故障', '问题', '坏了', '不工作', '错误', '异常']
        for keyword in problem_keywords:
            if keyword in query:
                entities["problems"].append(keyword)
        
        # 提取操作意图
        action_keywords = {
            '维修': ['维修', '修理', '检修'],
            '更换': ['更换', '替换', '换'],
            '查询': ['查询', '了解', '查看'],
            '使用': ['使用', '操作', '怎么用']
        }
        
        for action, keywords in action_keywords.items():
            if any(kw in query for kw in keywords):
                entities["actions"].append(action)
        
        return entities
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        检测语言
        
        Args:
            text: 文本
            
        Returns:
            语言代码 (zh, en, mixed)
        """
        # 统计中英文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total = chinese_chars + english_chars
        
        if total == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total
        
        if chinese_ratio > 0.7:
            return 'zh'
        elif english_chars / total > 0.7:
            return 'en'
        else:
            return 'mixed'

    @classmethod
    def normalize_query_for_retrieval(cls, query: str) -> dict:
        """
        将英文/混合问题归一化为更适合中文知识库检索的查询。

        归一化流程(英文/混合语言时):
        1. 短语映射: 用ENGLISH_PHRASE_MAP将完整英文短语替换为中文
           (按长度降序匹配避免短词优先截断)
        2. 单词映射: 用ENGLISH_TOKEN_MAP将英文单词替换为中文关键词
        3. 停用词过滤: 移除ENGLISH_STOPWORDS中的常见虚词
        4. 英文保留: 若仍保留3字符以上英文词，拼接回query，
           使MANUAL_ALIAS_SEEDS别名命中机制仍能工作

        例如:
        "how to turn on the boat engine" ->
        translated_query: "如何 打开 摩托艇 发动机"
        normalized_query: "如何 打开 摩托艇 发动机 boat engine"

        返回:
            {
              "language": "en/zh/mixed/unknown",
              "normalized_query": "...",  # 用于检索的最终query
              "translated_query": "...",   # 纯翻译结果
              "translation_applied": bool, # 是否进行了翻译
            }
        """
        query = TextProcessor.clean_text(query)
        language = cls.detect_language(query)
        if language not in {"en", "mixed"}:
            return {
                "language": language,
                "normalized_query": query,
                "translated_query": "",
                "translation_applied": False,
            }

        lowered = query.lower()
        translated = lowered
        # 短语映射: 按长度降序排列避免短词截断（如"first time to use"先于"first"匹配）
        for src, dst in sorted(cls.ENGLISH_PHRASE_MAP.items(), key=lambda item: len(item[0]), reverse=True):
            translated = re.sub(rf"\b{re.escape(src)}\b", dst, translated)

        # 分词并处理
        tokenized = re.findall(r"[a-z0-9_+-]+|[\u4e00-\u9fff]+", translated)
        normalized_tokens: List[str] = []
        for token in tokenized:
            if token in cls.ENGLISH_STOPWORDS:
                continue
            normalized_tokens.append(cls.ENGLISH_TOKEN_MAP.get(token, token))

        translated_query = " ".join(normalized_tokens)
        translated_query = re.sub(r"\s+", " ", translated_query).strip()

        # 若仍保留较多英文词，则拼接原词方便后续别名命中
        english_tail = " ".join(re.findall(r"[a-z]{3,}", lowered))
        normalized_query = translated_query
        if english_tail and english_tail not in translated_query:
            normalized_query = f"{translated_query} {english_tail}".strip()

        return {
            "language": language,
            "normalized_query": normalized_query,
            "translated_query": translated_query,
            "translation_applied": translated_query != query and bool(translated_query),
        }


def format_answer(
    content: str,
    include_images: bool = True,
    structure: str = "plain"
) -> str:
    """
    格式化回答
    
    Args:
        content: 回答内容
        include_images: 是否包含图片标记
        structure: 结构类型 (plain, numbered, markdown)
        
    Returns:
        格式化后的文本
    """
    if structure == "plain":
        return content
    
    if structure == "numbered":
        # 添加编号
        lines = content.split('\n')
        numbered_lines = []
        
        for i, line in enumerate(lines, 1):
            if line.strip():
                numbered_lines.append(f"{i}. {line}")
            else:
                numbered_lines.append(line)
        
        return '\n'.join(numbered_lines)
    
    if structure == "markdown":
        # 转换为Markdown格式
        # 标题处理
        content = re.sub(r'^([^#\n].*)$', r'## \1', content, flags=re.MULTILINE)
        
        # 列表处理
        content = re.sub(r'^(\d+)\.\s*(.*)$', r'1. \2', content, flags=re.MULTILINE)
        
        return content
    
    return content


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的字符级Jaccard相似度。

    算法: 将两个文本转换为字符集合，计算交并比。
    该方法是词项相似度的简化替代，不依赖分词器，
    适合快速估计两段文本的表面重合度。

    注意: 与_lexical_similarity不同，此函数基于字符(非词项)集合，
    因此对短文本和专有名词的差异较敏感。

    Args:
        text1: 文本1
        text2: 文本2

    Returns:
        0-1范围内的Jaccard相似度分数
    """
    # 转为小写字符集合
    set1 = set(text1.lower())
    set2 = set(text2.lower())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union
