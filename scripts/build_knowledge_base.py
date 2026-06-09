"""
知识库构建脚本
处理手册数据并构建RAG知识库
支持增量构建以节省内存

内存优化策略:
1. 逐文件处理：每个手册文件单独处理，避免一次性加载所有数据
2. 增量索引：每处理完一个文件即更新FAISS索引
3. 定期GC：处理过程中定期调用垃圾回收
4. 进度保存：支持断点续传，跳过已处理的手册
"""

import json
import re
import ast
import gc
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from loguru import logger

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, update_settings


class ManualParser:
    """解析赛题格式的手册文件"""

    @staticmethod
    def parse_manual_file(content: str) -> Tuple[str, List[str]]:
        """
        解析真实手册文件。

        当前手册主格式为:
        [
            "<包含 # 标题 与 <PIC> 占位符 的整本手册文本>",
            ["img_1", "img_2", ...]
        ]
        """
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

        # 兼容旧格式回退
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
        """
        按出现顺序将图片ID绑定到 <PIC> 占位符上。
        例如:
        <PIC> -> <PIC>[Manual16_51]
        """
        if not raw_text:
            return ""

        image_iter = iter(image_ids)

        def replace_pic(_: re.Match) -> str:
            image_id = next(image_iter, None)
            if image_id is None:
                return "<PIC>"
            return f"<PIC>[{image_id}]"

        text = re.sub(r"<PIC>", replace_pic, raw_text)

        # 极少数情况下图片ID数量多于占位符，保留在文末，避免丢图
        remaining = list(image_iter)
        if remaining:
            text = text.rstrip() + "\n" + " ".join(f"<PIC>[{img_id}]" for img_id in remaining)

        return text

    @staticmethod
    def normalize_manual_text(raw_text: str) -> str:
        """规范化手册文本，便于按章节切分。"""
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r'[ \t]{2,}#\s*', '\n# ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def extract_sections(raw_content: str) -> List[Dict[str, str]]:
        """
        按标题切分章节，返回带标题的结构化段落。

        通用标题识别策略（适用于各种领域的手册）:
        - 行首 # 标题
        - 句号/感叹号/问号后换行的 # 标题
        - <PIC> 后换行的 # 标题

        过滤策略（防止误判）:
        - 排除正文中间出现的 #（如 "规定详见 # 消音器"）
        - 排除目录项（如 ".12 启动与停机"）
        - 对 # 后紧跟正文的情况，按标点截断标题和正文
        """
        normalized = ManualParser.normalize_manual_text(raw_content)
        if not normalized:
            return []

        # 找所有 # 候选位置，同时记录 # 前面1个字符
        candidate_pattern = re.compile(r'.?(?<=[。！？\n])\s*#\s*([^\n#]{1,80})')
        matches = list(candidate_pattern.finditer(normalized))

        if not matches:
            # 兜底：尝试纯行首 #（适用于单行文件开头标题）
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
            raw_title = re.sub(r'\s+', ' ', match.group(1)).strip()
            # 排除目录项模式：标题以 .数字 结尾
            if re.search(r'\.\d+\s*$', raw_title):
                continue

            # 处理标题和正文粘连：# 标题 正文
            # 按第一个句末标点或列表编号截断
            stop_markers = [
                '。', '！', '？', '；', '：',
                '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.',
                '1、', '2、', '3、', '4、', '5、', '6、', '7、', '8、', '9、',
                '1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)',
            ]
            split_pos = None
            for marker in stop_markers:
                pos = raw_title.find(marker)
                if pos > 3 and (split_pos is None or pos < split_pos):
                    split_pos = pos

            if split_pos is not None:
                title = raw_title[:split_pos].strip()
                inline_body = raw_title[split_pos:].strip()
            else:
                title = raw_title
                inline_body = ''

            title = re.sub(r'\s*[（(]图\d+[）)]\s*$', '', title)
            title = re.sub(r'\s*[（(](?:PIC|pic)[）)]\s*$', '', title)
            title = re.sub(r'\s+', ' ', title).strip()
            if not title:
                continue

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
            body = normalized[start:end].strip()
            if inline_body:
                body = (inline_body + '\n' + body).strip()

            section_text = f"# {title}\n{body}".strip() if body else f"# {title}"
            sections.append({"title": title, "content": section_text})

        return sections

    @staticmethod
    def is_english_manual(manual_name: str, text: str) -> bool:
        """粗略判断是否为英文手册。"""
        if "英文手册" in (manual_name or ""):
            return True
        sample = (text or "")[:2000]
        if not sample:
            return False
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', sample))
        english_words = re.findall(r'[A-Za-z]{3,}', sample)
        return chinese_count < 20 and len(english_words) >= 30

    @staticmethod
    def _clean_english_line(line: str) -> str:
        line = re.sub(r'\s+', ' ', (line or '').strip())
        line = re.sub(r'\b(before|after|to|the|a|an)([A-Za-z]{2,})\b', r'\1 \2', line, flags=re.IGNORECASE)
        return line.strip(' -:;,.')

    @staticmethod
    def extract_english_sections(raw_content: str) -> List[Dict[str, str]]:
        """英文手册单独章节切分，避免标题/正文粘连。"""
        normalized = ManualParser.normalize_manual_text(raw_content)
        if not normalized:
            return []

        lines = [line.strip() for line in normalized.split('\n')]
        sections: List[Dict[str, str]] = []
        current_title = "概述"
        current_body: List[str] = []

        def flush_section():
            nonlocal current_body, current_title
            body_lines = [line for line in current_body if line and line != '#']
            body = '\n'.join(body_lines).strip()
            if current_title == "概述" and not body:
                current_body = []
                return
            content = f"# {current_title}\n{body}".strip() if body else f"# {current_title}"
            sections.append({"title": current_title, "content": content})
            current_body = []

        known_titles = {
            'important', 'danger', 'warning', 'caution', 'note', 'introduction',
            'general description', 'before first use', 'the nutriu app', 'preparing for use',
            'using the appliance', 'food table', 'airfrying', 'cleaning', 'cleaning table',
            'storage', 'recycling', 'troubleshooting', 'guarantee and support',
            'declaration of conformity', 'factory reset', 'device compatibility',
            'software updates', 'table of contents'
        }

        for raw_line in lines:
            line = ManualParser._clean_english_line(raw_line)
            if not line:
                continue

            line = re.sub(r'^#\s*', '', line).strip()
            lower = line.lower()

            is_heading = False
            heading = line
            body_part = ""

            if lower in known_titles:
                is_heading = True
            elif re.fullmatch(r'[A-Z][A-Za-z/&()\- ]{1,50}', line) and len(line.split()) <= 6:
                is_heading = True
            else:
                for title in sorted(known_titles, key=len, reverse=True):
                    if lower.startswith(title + ' '):
                        is_heading = True
                        heading = line[:len(title)].strip()
                        body_part = line[len(title):].strip()
                        break

            if is_heading:
                if current_body or current_title != "概述":
                    flush_section()
                current_title = heading.title() if heading.islower() else heading
                if body_part:
                    current_body.append(body_part)
                continue

            current_body.append(line)

        if current_body or current_title != "概述":
            flush_section()

        return sections or [{"title": "全文", "content": normalized}]


    @staticmethod
    def extract_image_ids(text: str) -> List[str]:
        """提取当前文本块内绑定的图片ID。"""
        return re.findall(r'<PIC>\[([^\]]+)\]', text or "")


class KnowledgeBaseBuilder:
    """知识库构建器 - 内存友好版本（增强分块）"""

    # 内存优化配置
    BATCH_EMBED_SIZE = 16  # 每批嵌入处理的文档数
    GC_INTERVAL = 5  # 每处理N个文件后执行GC
    SAVE_INTERVAL = 5  # 每处理N个文件后执行一次完整保存
    DEFAULT_EXCLUDED_FILES = {"汇总英文手册.txt"}
    PIC_CONTEXT_PREFIX = 60  # 图片占位符前保留的字符数
    PIC_CONTEXT_SUFFIX = 40  # 图片占位符后保留的字符数

    def __init__(self, manual_dir: str = None, include_excluded_files: bool = False):
        if manual_dir:
            self.manual_dir = Path(manual_dir)
        else:
            self.manual_dir = PROJECT_ROOT / "手册new"

        self.include_excluded_files = include_excluded_files
        self.index_dir = PROJECT_ROOT / "knowledge_base" / "index_v1"
        self.progress_file = self.index_dir / "build_progress.json"

        # 从 settings 读取 chunk 参数，建议在 config.py 中设置 chunk_size=400, chunk_overlap=50
        self.CHUNK_SIZE = settings.chunk_size
        self.CHUNK_OVERLAP = settings.chunk_overlap

        # RAG引擎延迟初始化
        self._rag_engine = None
        self._initialized = False

        # 统计
        self._total_files = 0
        self._processed_files = 0
        self._total_docs = 0
        self._total_chars = 0

    @property
    def rag_engine(self):
        """延迟加载RAG引擎"""
        if self._rag_engine is None:
            from src.modules.rag_engine import get_rag_engine, reset_rag_engine
            if self._initialized:
                reset_rag_engine()
            self._rag_engine = get_rag_engine()
        return self._rag_engine

    def initialize(self):
        """初始化RAG引擎"""
        self.rag_engine.initialize()
        self._initialized = True
        logger.info("RAG引擎初始化完成")

    def _load_progress(self) -> Dict[str, Any]:
        """加载构建进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"processed_files": [], "total_docs": 0, "total_chars": 0}

    def _save_progress(self, progress: Dict[str, Any]):
        """保存构建进度"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _get_existing_doc_ids(self) -> Set[str]:
        """获取已存在的文档ID"""
        existing_ids = set()
        metadata_file = self.index_dir / settings.metadata_file
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    for doc in metadata.get("texts", []):
                        existing_ids.add(doc["doc_id"])
                    for doc in metadata.get("images", []):
                        existing_ids.add(doc["doc_id"])
            except Exception:
                pass
        return existing_ids

    def _chunk_text(self, text: str, section_title: str = "") -> List[str]:
        """
        将长文本切分为语义边界感知的文本块（增强版）。

        切分策略（优先级从高到低）:
        1. 若文本长度<=CHUNK_SIZE，直接返回
        2. 优先按段落边界（空行）切分
        3. 其次按列表项边界（•、、数字编号、字母编号等）切分
        4. 最后按句子边界切分（中文：。！？，英文：. ? !）
        5. 连续窗口间有CHUNK_OVERLAP重叠

        语言检测规则:
        - 有中文 → 主要使用中文分隔符（。！？\n）
        - 有英文句号 → 加入英文分隔符（. ? ! \n Step等）

        防止死循环机制: 若CHUNK_OVERLAP过大导致next_start<=previous_start，
        强制将游标推进到end位置。
        """
        text = text.strip()
        if not text:
            return []

        title_prefix = f"{section_title}\n" if section_title else ""

        if len(text) <= self.CHUNK_SIZE:
            return [title_prefix + text]

        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        has_english_period = '.' in text

        # --- 增强的列表项分隔符（覆盖更多格式）---
        list_seps = [
            # 标准项目符号
            '\n• ', '\n· ', '\n– ', '\n— ', '\n ', '\n- ',
            # 数字编号（中英文）
            '\n1. ', '\n2. ', '\n3. ', '\n4. ', '\n5. ',
            '\n6. ', '\n7. ', '\n8. ', '\n9. ', '\n0. ',
            '\n1、', '\n2、', '\n3、', '\n4、', '\n5、',
            '\n1)', '\n2)', '\n3)', '\n4)', '\n5)',
            '\n(1)', '\n(2)', '\n(3)', '\n(4)', '\n(5)',
            '\n1）', '\n2）', '\n3）', '\n4）', '\n5）',
            # 字母编号（英文常见）
            '\na) ', '\nb) ', '\nc) ', '\nd) ', '\ne) ',
            '\nA) ', '\nB) ', '\nC) ', '\nD) ', '\nE) ',
            '\na. ', '\nb. ', '\nc. ', '\nd. ', '\ne. ',
            '\nA. ', '\nB. ', '\nC. ', '\nD. ', '\nE. ',
            # 英文步骤关键词
            '\nStep ', '\nstep ', '\nNote: ', '\nTIP: ',
        ]
        en_sentence_seps = ['. ', '.\n', '? ', '!\n', '?\n', '!? ', '! ']
        zh_sentence_seps = ['。', '！', '？']
        paragraph_sep = '\n\n'

        if has_chinese and has_english_period:
            separators = [paragraph_sep] + list_seps + zh_sentence_seps + en_sentence_seps
        elif has_chinese:
            separators = [paragraph_sep] + list_seps + zh_sentence_seps
        else:
            separators = [paragraph_sep] + list_seps + en_sentence_seps

        chunks = []
        start = 0
        text_len = len(text)
        last_normalized_chunk = ""

        while start < text_len:
            previous_start = start
            end = start + self.CHUNK_SIZE

            if end < text_len:
                best_boundary = -1
                for sep in separators:
                    if sep == paragraph_sep:
                        boundary = text.rfind('\n\n', start, end)
                        if boundary > start:
                            best_boundary = boundary + 2
                            break
                    else:
                        boundary = text.rfind(sep, start, end)
                        if boundary > start:
                            best_boundary = max(best_boundary, boundary + len(sep))
                if best_boundary > start:
                    end = best_boundary

            chunk = text[start:end].strip()
            normalized_chunk = re.sub(r'\s+', ' ', chunk)
            if chunk:
                if normalized_chunk != last_normalized_chunk:
                    chunks.append(title_prefix + chunk)
                    last_normalized_chunk = normalized_chunk

            if end >= text_len:
                break

            remaining_len = text_len - end
            if remaining_len < max(40, self.CHUNK_OVERLAP // 2):
                tail_chunk = text[max(0, end - self.CHUNK_OVERLAP):].strip()
                normalized_tail = re.sub(r'\s+', ' ', tail_chunk)
                if tail_chunk and normalized_tail != last_normalized_chunk:
                    chunks.append(title_prefix + tail_chunk)
                break

            next_start = max(end - self.CHUNK_OVERLAP, previous_start + 1)
            if next_start <= previous_start:
                next_start = end
            start = next_start

        return chunks

    def _chunk_english_text(self, text: str, section_title: str = "") -> List[str]:
        """英文手册单独分块：按段落/句子聚合，避免滑窗碎片。"""
        text = text.strip()
        if not text:
            return []

        title_prefix = f"{section_title}\n" if section_title else ""
        body = re.sub(r'^#\s*[^\n]+\n?', '', text).strip() if text.startswith('#') else text
        if not body:
            return [title_prefix + text]

        paragraphs = [p.strip() for p in re.split(r'\n{2,}', body) if p.strip()]
        if not paragraphs:
            paragraphs = [body]

        chunks: List[str] = []
        current = ""

        def flush_current():
            nonlocal current
            current = current.strip()
            if current:
                chunks.append(title_prefix + current)
            current = ""

        for para in paragraphs:
            para = re.sub(r'\s+', ' ', para).strip()
            if not para:
                continue

            if len(para) > self.CHUNK_SIZE:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    candidate = f"{current} {sentence}".strip() if current else sentence
                    if len(candidate) <= self.CHUNK_SIZE:
                        current = candidate
                    else:
                        flush_current()
                        if len(sentence) <= self.CHUNK_SIZE:
                            current = sentence
                        else:
                            for i in range(0, len(sentence), self.CHUNK_SIZE):
                                part = sentence[i:i + self.CHUNK_SIZE].strip()
                                if part:
                                    chunks.append(title_prefix + part)
            else:
                candidate = f"{current}\n\n{para}".strip() if current else para
                if len(candidate) <= self.CHUNK_SIZE:
                    current = candidate
                else:
                    flush_current()
                    current = para

        flush_current()
        return chunks or [title_prefix + body]


    def _process_single_manual(
        self,
        manual_file: Path,
        existing_ids: Set[str]
    ) -> Tuple[List[Dict], int, bool]:
        """
        处理单个手册文件
        返回: (文档列表, 字符数, 是否成功)
        """
        documents = []
        total_chars = 0

        try:
            with open(manual_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            manual_name = manual_file.stem
            manual_text, manual_image_ids = ManualParser.parse_manual_file(raw_content)
            bound_text = ManualParser.inject_image_ids(manual_text, manual_image_ids)
            is_english_manual = ManualParser.is_english_manual(manual_name, bound_text)
            sections = (
                ManualParser.extract_english_sections(bound_text)
                if is_english_manual
                else ManualParser.extract_sections(bound_text)
            )

            for i, section in enumerate(sections):
                content = section["content"].strip()
                section_title = section["title"]
                if not content:
                    continue

                chunks = (
                    self._chunk_english_text(content, section_title)
                    if is_english_manual
                    else self._chunk_text(content, section_title)
                )

                for j, chunk in enumerate(chunks):
                    doc_id = f"manual_{manual_name}_s{i}_c{j}"

                    if doc_id in existing_ids:
                        continue

                    chunk_image_ids = ManualParser.extract_image_ids(chunk)

                    doc = {
                        "content": chunk,
                        "doc_id": doc_id,
                        "metadata": {
                            "source_file": manual_file.name,
                            "manual_name": manual_name,
                            "section_title": section_title,
                            "section_index": i,
                            "chunk_index": j,
                            "route": "manual",
                            "type": "manual",
                            "image_ids": chunk_image_ids,
                            "image_count": len(chunk_image_ids)
                        }
                    }
                    documents.append(doc)
                    total_chars += len(chunk)

            logger.info(f"处理手册 {manual_file.name}: {len(sections)} 个段落 -> {len(documents)} 个新文档")
            return documents, total_chars, True

        except Exception as e:
            logger.error(f"处理手册失败 {manual_file}: {type(e).__name__}: {e!r}")
            return [], 0, False

    def build(self, force_rebuild: bool = False):
        """
        构建知识库 - 内存优化版本

        策略:
        1. 逐文件处理，每个手册文件独立处理
        2. 增量添加到FAISS索引
        3. 定期执行垃圾回收
        4. 支持断点续传
        """
        logger.info("=" * 60)
        logger.info("开始构建知识库 (内存优化模式)...")
        logger.info(f"手册目录: {self.manual_dir}")
        logger.info(f"Embedding后端: {settings.embedding_backend}")
        logger.info(f"分块大小: {self.CHUNK_SIZE}, 重叠: {self.CHUNK_OVERLAP}")
        logger.info("=" * 60)

        self.initialize()

        if not self.manual_dir.exists():
            logger.error(f"手册目录不存在: {self.manual_dir}")
            return

        txt_files = sorted(self.manual_dir.glob("*.txt"))
        if not self.include_excluded_files:
            txt_files = [
                path for path in txt_files
                if path.name not in self.DEFAULT_EXCLUDED_FILES
            ]
            if self.DEFAULT_EXCLUDED_FILES:
                logger.info(f"默认排除低相关度手册: {', '.join(sorted(self.DEFAULT_EXCLUDED_FILES))}")
        if not txt_files:
            logger.error(f"未找到txt手册文件: {self.manual_dir}")
            return

        self._total_files = len(txt_files)

        if force_rebuild:
            logger.info("强制重建：清空现有知识库...")
            self.rag_engine.rebuild()
            progress = {"processed_files": [], "total_docs": 0, "total_chars": 0}
            existing_ids: Set[str] = set()
        else:
            progress = self._load_progress()
            existing_ids = self._get_existing_doc_ids()
            logger.info(f"发现 {len(existing_ids)} 个已存在文档，将跳过重复")

        processed_files = set(progress.get("processed_files", []))
        self._total_docs = progress.get("total_docs", 0)
        self._total_chars = progress.get("total_chars", 0)

        new_docs = 0
        new_chars = 0

        for idx, txt_file in enumerate(txt_files):
            file_name = txt_file.name
            if file_name in processed_files:
                logger.info(f"[{idx+1}/{self._total_files}] 跳过已处理: {file_name}")
                continue

            logger.info(f"[{idx+1}/{self._total_files}] 处理中: {file_name}")
            self._processed_files = idx + 1

            docs, chars, success = self._process_single_manual(txt_file, existing_ids)

            if docs:
                self.rag_engine.add_documents(docs, doc_type="text")
                existing_ids.update(doc["doc_id"] for doc in docs)
                new_docs += len(docs)
                new_chars += chars
                self._total_docs += len(docs)
                self._total_chars += chars

            if success:
                processed_files.add(file_name)
                progress = {
                    "processed_files": list(processed_files),
                    "total_docs": self._total_docs,
                    "total_chars": self._total_chars
                }
                self._save_progress(progress)

            # 定期GC
            if (idx + 1) % self.GC_INTERVAL == 0:
                gc.collect()
                logger.debug(f"内存清理完成，当前文档数: {self._total_docs}")

            # 定期保存
            if (idx + 1) % self.SAVE_INTERVAL == 0 or (idx + 1) == self._total_files:
                self.rag_engine.save_knowledge_base()
                logger.debug(f"[保存] 已保存知识库（已处理 {idx + 1}/{self._total_files} 个文件）")

        gc.collect()

        logger.info("=" * 60)
        logger.info(f"知识库构建完成!")
        logger.info(f"  - 处理手册数: {self._processed_files}/{self._total_files}")
        logger.info(f"  - 新增文档数: {new_docs}")
        logger.info(f"  - 总文档数: {self._total_docs}")
        logger.info(f"  - 总字符数: {self._total_chars}")
        logger.info(f"  - 索引位置: {self.index_dir}")
        logger.info("=" * 60)

    def add_sample_data(self):
        """添加赛题示例相关的数据（仅用于 debug / 演示，正式知识库不应包含）"""
        logger.warning(
            "[add_sample_data] 正在添加赛题示例数据！"
            "这些数据仅用于 debug / 演示，正式提交前请勿混入正式知识库。"
        )
        logger.info("添加赛题示例数据...")

        self.initialize()

        sample_docs = [
            {
                "content": "DCB107、DCB112电池组充电中<PIC>[drill0_04]\n\n当电池组正在充电时，指示灯会闪烁，表示充电进行中。",
                "doc_id": "drill_battery_charging",
                "metadata": {"category": "battery", "models": ["DCB107", "DCB112"], "type": "sample", "route": "manual"}
            },
            {
                "content": "电池组已充满<PIC>[drill0_05]\n\n当指示灯常亮不再闪烁时，表示电池已完全充满。",
                "doc_id": "drill_battery_full",
                "metadata": {"category": "battery", "models": ["DCB107", "DCB112"], "type": "sample", "route": "manual"}
            },
            {
                "content": "过热/过冷延迟<PIC>[drill0_06]\n\n当电池温度过高或过低时，充电器会进入延迟模式，指示灯慢闪。",
                "doc_id": "drill_battery_temp",
                "metadata": {"category": "battery", "models": ["DCB107", "DCB112"], "type": "sample", "route": "manual"}
            },
            {
                "content": "表带尺寸\n\n表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。<PIC>[Manual16_51]",
                "doc_id": "watch_band_size",
                "metadata": {"category": "accessory", "product": "fitness_tracker", "type": "sample", "route": "manual"}
            },
            {
                "content": "环境条件<PIC>[Manual16_52]\n\n本产品适用于一般室内外环境，请避免极端温度和湿度。",
                "doc_id": "watch_env_conditions",
                "metadata": {"category": "specification", "product": "fitness_tracker", "type": "sample", "route": "manual"}
            },
            {
                "content": "配送范围\n\n我们的商品支持送到大部分乡镇哦，具体能否送达，取决于您的收货地址，您可以告诉我详细的收货地址，我帮您查询。",
                "doc_id": "shipping_area",
                "metadata": {"category": "shipping", "type": "sample", "route": "service"}
            },
            {
                "content": "运费说明\n\n送到乡镇一般不需要额外加运费，和市区运费一致。",
                "doc_id": "shipping_fee",
                "metadata": {"category": "shipping", "type": "sample", "route": "service"}
            },
            {
                "content": "配送时效\n\n正常情况下，下单后48小时发货，乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。",
                "doc_id": "shipping_time",
                "metadata": {"category": "shipping", "type": "sample", "route": "service"}
            },
            {
                "content": "待揽收状态\n\n您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，我们会催促快递方尽快上门。",
                "doc_id": "logistics_pending",
                "metadata": {"category": "logistics", "type": "sample", "route": "service"}
            },
            {
                "content": "维修质保\n\n您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，且是上次维修不彻底导致的，属于我们的维修失误，支持免费重新维修，并延长维修质保期。请您提供维修单号、商品故障描述，我们立即安排专业维修人员处理。",
                "doc_id": "repair_warranty",
                "metadata": {"category": "after_sales", "type": "sample", "route": "service"}
            },
        ]

        self.rag_engine.add_documents(sample_docs, doc_type="text")
        self.rag_engine.save_knowledge_base()
        logger.info(f"添加了 {len(sample_docs)} 个赛题示例文档")

    def show_stats(self):
        """显示知识库统计信息"""
        print("\n" + "=" * 60)
        print("知识库统计信息")
        print("=" * 60)

        if self.index_dir.exists():
            for f in self.index_dir.iterdir():
                if f.is_file():
                    size = f.stat().st_size
                    if size > 1024 * 1024:
                        print(f"  {f.name}: {size / 1024 / 1024:.2f} MB")
                    elif size > 1024:
                        print(f"  {f.name}: {size / 1024:.2f} KB")
                    else:
                        print(f"  {f.name}: {size} B")

        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                print(f"\n  构建进度:")
                print(f"    - 已处理手册: {len(progress.get('processed_files', []))}")
                print(f"    - 总文档数: {progress.get('total_docs', 0)}")
                print(f"    - 总字符数: {progress.get('total_chars', 0):,}")
            except Exception:
                pass

        print("  " + "-" * 40)
        print(f"  手册目录: {self.manual_dir}")
        if self.manual_dir.exists():
            txt_files = list(self.manual_dir.glob("*.txt"))
            print(f"  手册数量: {len(txt_files)}")
            total_size = sum(f.stat().st_size for f in txt_files)
            print(f"  总大小: {total_size / 1024 / 1024:.2f} MB")
            if len(txt_files) > 5:
                print("  前5个手册:")
                for tf in txt_files[:5]:
                    size = tf.stat().st_size
                    print(f"    - {tf.name}: {size / 1024:.1f} KB")
                print(f"    ... 还有 {len(txt_files) - 5} 个")
            else:
                for tf in txt_files:
                    size = tf.stat().st_size
                    print(f"    - {tf.name}: {size / 1024:.1f} KB")
        else:
            print("  (手册目录不存在)")
        print("=" * 60 + "\n")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="知识库构建工具 (内存优化版，增强分块)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m scripts.build_knowledge_base --mode build
  python -m scripts.build_knowledge_base --mode build --force
  python -m scripts.build_knowledge_base --mode sample
  python -m scripts.build_knowledge_base --mode stats
        """
    )
    parser.add_argument(
        "--mode",
        choices=["build", "sample", "stats"],
        default="stats",
        help="build=增量构建索引, sample=示例数据, stats=显示统计"
    )
    parser.add_argument(
        "--manual-dir",
        default=None,
        help="手册文件目录"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建索引（清空现有数据）"
    )
    parser.add_argument(
        "--backend",
        choices=["hashing", "sentence_transformer"],
        default=None,
        help="嵌入后端，hashing更省内存，适合Cursor终端和低内存环境"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="计算设备，CUDA不可用时自动回退到CPU"
    )
    parser.add_argument(
        "--include-excluded-files",
        action="store_true",
        help="包含默认排除的低相关度手册（如英文汇总手册）"
    )

    args = parser.parse_args()

    if args.backend or args.device:
        new_settings = {}
        if args.backend:
            new_settings["embedding_backend"] = args.backend
            if args.backend == "hashing":
                new_settings["enable_vision_model"] = False
                new_settings["embedding_batch_size"] = min(settings.embedding_batch_size, 8)
        if args.device:
            new_settings["embedding_device"] = args.device
        update_settings(**new_settings)
        logger.info(f"本次构建使用 embedding_backend={settings.embedding_backend}, embedding_device={settings.embedding_device}")

    builder = KnowledgeBaseBuilder(
        args.manual_dir,
        include_excluded_files=args.include_excluded_files
    )

    if args.mode == "stats":
        builder.show_stats()
    elif args.mode == "build":
        builder.build(force_rebuild=args.force)
    elif args.mode == "sample":
        builder.add_sample_data()
        print("\n示例数据添加完成!")
        print(f"索引文件位置: {PROJECT_ROOT / 'knowledge_base' / 'index'}")


if __name__ == "__main__":
    main()