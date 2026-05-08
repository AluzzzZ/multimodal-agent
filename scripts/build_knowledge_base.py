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
        当前手册中大量标题出现在:
        - 行首: # 标题
        - 占位图后: <PIC>[id]  # 标题
        """
        normalized = ManualParser.normalize_manual_text(raw_content)
        if not normalized:
            return []

        pattern = re.compile(r'(?:(?<=\n)|^)\s*#\s*([^\n#]{1,80})')
        matches = list(pattern.finditer(normalized))
        if not matches:
            return [{"title": "全文", "content": normalized}]

        sections: List[Dict[str, str]] = []

        if matches[0].start() > 0:
            preface = normalized[:matches[0].start()].strip()
            if preface:
                sections.append({"title": "概述", "content": preface})

        for idx, match in enumerate(matches):
            title = re.sub(r'\s+', ' ', match.group(1)).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
            body = normalized[start:end].strip()
            section_text = f"# {title}\n{body}".strip() if body else f"# {title}"
            sections.append({"title": title, "content": section_text})

        return sections

    @staticmethod
    def extract_image_ids(text: str) -> List[str]:
        """提取当前文本块内绑定的图片ID。"""
        return re.findall(r'<PIC>\[([^\]]+)\]', text or "")


class KnowledgeBaseBuilder:
    """知识库构建器 - 内存友好版本"""

    # 内存优化配置
    CHUNK_SIZE = 400  # 文本分块大小（字符数）
    CHUNK_OVERLAP = 80  # 重叠大小（增大以保留更多跨chunk上下文）
    BATCH_EMBED_SIZE = 16  # 每批嵌入处理的文档数
    GC_INTERVAL = 5  # 每处理N个文件后执行GC
    DEFAULT_EXCLUDED_FILES = {"汇总英文手册.txt"}
    # 图片上下文扩展配置
    PIC_CONTEXT_PREFIX = 60  # 图片占位符前保留的字符数
    PIC_CONTEXT_SUFFIX = 40  # 图片占位符后保留的字符数

    def __init__(self, manual_dir: str = None, include_excluded_files: bool = False):
        if manual_dir:
            self.manual_dir = Path(manual_dir)
        else:
            self.manual_dir = PROJECT_ROOT / "手册"

        self.include_excluded_files = include_excluded_files
        self.index_dir = PROJECT_ROOT / "knowledge_base" / "index"
        self.progress_file = self.index_dir / "build_progress.json"

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
            # 如果已初始化过，先重置
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

        # 从metadata中读取已存在的ID
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

    def _chunk_text(self, text: str) -> List[str]:
        """
        将长文本切分为固定大小的文本块。

        切分策略:
        1. 若文本长度<=CHUNK_SIZE，直接返回
        2. 否则按CHUNK_SIZE划分窗口，优先在句子边界(。！？
)截断
        3. 连续窗口间有CHUNK_OVERLAP重叠，保证跨句子边界的上下文不丢失

        防止死循环机制: 若CHUNK_OVERLAP过大导致next_start<=previous_start，
        强制将游标推进到end位置，确保每次循环都有进展。

        Returns:
            切分后的文本块列表
        """
        text = text.strip()
        if len(text) <= self.CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            previous_start = start
            end = start + self.CHUNK_SIZE

            # 优先在句子边界处截断，保持句子完整性
            if end < text_len:
                for sep in ['。', '！', '？', '\n']:
                    boundary = text.rfind(sep, start, end)
                    if boundary > start:
                        end = boundary + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break

            # 计算下一窗口起点，确保有重叠且游标始终前进
            next_start = max(end - self.CHUNK_OVERLAP, previous_start + 1)
            if next_start <= previous_start:
                next_start = end
            start = next_start

        return chunks

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
            sections = ManualParser.extract_sections(bound_text)

            for i, section in enumerate(sections):
                content = section["content"].strip()
                section_title = section["title"]
                if not content:
                    continue

                # 分块处理。每个chunk保留本地图片绑定关系，避免整章图片污染所有块。
                chunks = self._chunk_text(content)

                for j, chunk in enumerate(chunks):
                    doc_id = f"manual_{manual_name}_s{i}_c{j}"

                    # 跳过已存在的文档
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
        logger.info("=" * 60)

        # 初始化RAG引擎
        self.initialize()

        # 检查手册目录
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
                logger.info(
                    f"默认排除低相关度手册: {', '.join(sorted(self.DEFAULT_EXCLUDED_FILES))}"
                )
        if not txt_files:
            logger.error(f"未找到txt手册文件: {self.manual_dir}")
            return

        self._total_files = len(txt_files)

        # 如果强制重建，先清空
        if force_rebuild:
            logger.info("强制重建：清空现有知识库...")
            self.rag_engine.rebuild()
            progress = {"processed_files": [], "total_docs": 0, "total_chars": 0}
            existing_ids: Set[str] = set()
        else:
            # 加载进度
            progress = self._load_progress()
            existing_ids = self._get_existing_doc_ids()
            logger.info(f"发现 {len(existing_ids)} 个已存在文档，将跳过重复")

        # 获取已处理的文件列表
        processed_files = set(progress.get("processed_files", []))
        self._total_docs = progress.get("total_docs", 0)
        self._total_chars = progress.get("total_chars", 0)

        # 统计
        new_docs = 0
        new_chars = 0

        for idx, txt_file in enumerate(txt_files):
            file_name = txt_file.name

            # 跳过已处理的文件
            if file_name in processed_files:
                logger.info(f"[{idx+1}/{self._total_files}] 跳过已处理: {file_name}")
                continue

            logger.info(f"[{idx+1}/{self._total_files}] 处理中: {file_name}")
            self._processed_files = idx + 1

            # 处理单个手册
            docs, chars, success = self._process_single_manual(txt_file, existing_ids)

            if docs:
                # 增量添加文档
                self.rag_engine.add_documents(docs, doc_type="text")
                self.rag_engine.save_knowledge_base()
                existing_ids.update(doc["doc_id"] for doc in docs)

                new_docs += len(docs)
                new_chars += chars
                self._total_docs += len(docs)
                self._total_chars += chars

            # 仅在处理成功时更新进度，避免失败文件被错误标记为已完成
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

        # 最终保存
        self.rag_engine.save_knowledge_base()
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
        """添加赛题示例相关的数据"""
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

        # 索引文件信息
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

        # 构建进度
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

        # 手册文件信息
        print(f"  手册目录: {self.manual_dir}")
        if self.manual_dir.exists():
            txt_files = list(self.manual_dir.glob("*.txt"))
            print(f"  手册数量: {len(txt_files)}")
            total_size = sum(f.stat().st_size for f in txt_files)
            print(f"  总大小: {total_size / 1024 / 1024:.2f} MB")

            # 显示前5个手册
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
        description="知识库构建工具 (内存优化版)",
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
        "--include-excluded-files",
        action="store_true",
        help="包含默认排除的低相关度手册（如英文汇总手册）"
    )

    args = parser.parse_args()

    if args.backend:
        new_settings = {"embedding_backend": args.backend}
        if args.backend == "hashing":
            new_settings["enable_vision_model"] = False
            new_settings["embedding_batch_size"] = min(settings.embedding_batch_size, 8)
        update_settings(**new_settings)
        logger.info(f"本次构建使用 embedding_backend={settings.embedding_backend}")

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
