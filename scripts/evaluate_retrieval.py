"""
公开题检索评测脚本（增强版）

新增功能：
1. AB 对比模式：支持在同题上对比新旧两种 embedding 模型的 Top-K 命中率
2. 分层召回率指标：按 route（manual/service/mixed）分组统计召回率
3. 综合评分：融合 Top1 分数、关键词覆盖、召回率的多维评估

用途：
1. 批量读取 question_public.csv
2. 调用当前知识库执行 Top-K 检索
3. 生成逐题检索明细、逐题摘要和总览报告
4. 可选：对比两种 embedding 配置的检索效果

说明：
- 当前 question_public.csv 不包含标准答案，因此本脚本评估的是"召回画像"和"命中可疑度"，
  不是严格准确率。
- 评测结果适合快速排查：
  1. 哪些问题几乎没有命中知识库
  2. 哪些问题被错误手册持续吸走
  3. 哪些问题虽然有召回，但关键词覆盖度很低
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, update_settings
from src.modules.rag_engine import get_rag_engine, reset_rag_engine


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "knowledge_base" / "evaluation"
DEFAULT_QUESTION_FILE = PROJECT_ROOT / "question_public.csv"

# 客服意图关键词，用于自动推断 route 标签
SERVICE_KEYWORDS = [
    "退货", "换货", "退款", "取消订单", "发票", "开票", "物流",
    "快递", "运费", "揽收", "补寄", "包装破损", "少发", "污渍",
    "损坏", "维修", "质保", "保修", "投诉", "赔偿", "安装",
    "上门", "试用装", "说明书", "电子版", "纸质版",
]

MANUAL_KEYWORDS = [
    "功能", "使用", "操作", "步骤", "设置", "连接", "充电",
    "电池", "屏幕", "按钮", "规格", "参数", "尺寸", "重量",
    "材质", "配件", "清洁", "保养", "故障", "问题", "怎么",
]

STOPWORDS = {
    "请问", "一下", "你们", "我们", "商品", "这个", "那个",
    "什么", "怎么", "如何", "可以", "是否", "一下子", "一下呢",
    "一下吗", "吗", "呢", "呀", "啊", "了", "的", "和", "与",
    "或", "及", "再", "还", "有", "被", "把", "在", "是", "我",
    "要", "想", "给", "后", "前",
}


@dataclass
class QuestionRecord:
    """单条公开题记录。"""
    question_id: str
    raw_question: str
    normalized_question: str
    inferred_route: str = ""  # manual / service / mixed / unknown
    keywords: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评估公开题检索效果（增强版）")
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTION_FILE),
        help="公开题 CSV 文件路径，默认使用 question_public.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="评测输出目录",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="每题导出的 Top-K 检索结果数量",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评测前 N 题，0 表示全部题目",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="评测时启用 rerank",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["hashing", "sentence_transformer"],
        default=None,
        help="覆盖当前嵌入后端配置",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
        help="检索内容预览长度",
    )
    # AB 对比模式
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="AB 对比基线模型名称（如 BAAI/bge-m3），留空则不启用 AB 对比",
    )
    parser.add_argument(
        "--baseline-output-dir",
        default=None,
        help="基线模型评测结果目录（若不指定则用 <output-dir>/baseline）",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="启用 dense+sparse 混合检索",
    )
    return parser.parse_args()


def normalize_question(raw_question: str) -> str:
    """清洗公开题中的多行/多引号格式。"""
    text = (raw_question or "").strip()
    if not text:
        return ""

    text = text.replace('""', '"')
    text = re.sub(r'^\s*"', "", text)
    text = re.sub(r'"\s*$', "", text)
    text = re.sub(r'"\s*,\s*"', "\n", text)
    text = text.replace('",\n"', "\n")
    text = text.replace('","', "\n")
    text = text.replace("\\n", "\n")

    lines = []
    for line in text.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def infer_route(normalized_question: str) -> str:
    """
    根据问题文本中的关键词自动推断路由类型。

    判断规则：
    - service 关键词命中 > 0 且 manual 关键词命中 = 0 -> service
    - manual 关键词命中 > 0 且 service 关键词命中 = 0 -> manual
    - 两者都有命中 -> mixed
    - 都没有命中 -> unknown
    """
    q_lower = normalized_question.lower()
    service_hits = sum(1 for kw in SERVICE_KEYWORDS if kw in q_lower)
    manual_hits = sum(1 for kw in MANUAL_KEYWORDS if kw in q_lower)

    if service_hits > 0 and manual_hits == 0:
        return "service"
    if manual_hits > 0 and service_hits == 0:
        return "manual"
    if service_hits > 0 and manual_hits > 0:
        return "mixed"
    return "unknown"


def load_questions(csv_path: Path, limit: int = 0) -> List[QuestionRecord]:
    """加载公开题并推断每题的路由类型。"""
    records: List[QuestionRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = normalize_question(row["question"])
            keywords = extract_keywords(normalized)
            inferred = infer_route(normalized)
            record = QuestionRecord(
                question_id=str(row["id"]).strip(),
                raw_question=row["question"],
                normalized_question=normalized,
                inferred_route=inferred,
                keywords=keywords,
            )
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def extract_keywords(text: str) -> List[str]:
    """提取问题文本中的有效关键词（去停用词、去重、保持顺序）。"""
    candidates = re.findall(r"[A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}", text or "")
    keywords: List[str] = []
    for token in candidates:
        token = token.strip().lower()
        if not token or token in STOPWORDS:
            continue
        keywords.append(token)
    return list(dict.fromkeys(keywords))


def clean_preview(text: str, preview_chars: int) -> str:
    """截取预览文本，去除多余空白。"""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:preview_chars]


def ratio(numerator: int, denominator: int) -> float:
    """安全除法。"""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def summarize_hit(top1_score: float, coverage: float, result_count: int) -> str:
    """
    综合 Top1 分数和关键词覆盖度，判断单题命中质量。

    质量等级：
    - empty: 无检索结果
    - high: Top1 >= 0.72 且 coverage >= 0.35
    - medium: Top1 >= 0.58 且 coverage >= 0.18
    - low: 其余情况
    """
    if result_count == 0:
        return "empty"
    if top1_score >= 0.72 and coverage >= 0.35:
        return "high"
    if top1_score >= 0.58 and coverage >= 0.18:
        return "medium"
    return "low"


def compute_recall_at_k(
    results: List[Any],
    k: int,
    min_score: float = 0.0,
) -> float:
    """
    计算 Top-K 召回率。

    定义：有至少一条结果且分数 >= min_score 视为"召回成功"。
    """
    if not results:
        return 0.0
    valid = [r for r in results[:k] if r.get("relevance_score", 0) >= min_score]
    return 1.0 if valid else 0.0


def run_retrieval(
    questions: Sequence[QuestionRecord],
    top_k: int,
    use_rerank: bool,
    preview_chars: int,
    enable_hybrid: bool = False,
) -> Dict[str, Any]:
    """
    对公开题执行检索并收集评测数据。

    新增字段：
    - recall@5: 是否在 Top-5 中有有效召回
    - recall@top3: 是否在 Top-3 中有有效召回
    - inferred_route: 自动推断的路由类型
    - dense_score / sparse_score: 当启用混合检索时分别记录
    """
    reset_rag_engine()

    # 显式设置混合检索开关：无论 .env 默认值是什么，评测时严格按参数控制
    update_settings(enable_hybrid_retrieval=enable_hybrid)

    rag = get_rag_engine()
    rag.initialize()

    detail_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    manual_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    # 按路由类型分组统计
    route_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "total": 0, "high": 0, "medium": 0, "low": 0, "empty": 0,
        "recalled": 0, "recalled_top3": 0,
    })

    for question in questions:
        results = rag.retrieve(question.normalized_question, top_k=top_k, use_rerank=use_rerank)
        keywords = question.keywords
        joined_context = "\n".join(item.get("content", "") for item in results).lower()
        matched_keywords = [kw for kw in keywords if kw in joined_context]
        keyword_coverage = ratio(len(matched_keywords), len(keywords))

        top1_score = results[0]["relevance_score"] if results else 0.0
        avg_score = statistics.fmean(item["relevance_score"] for item in results) if results else 0.0
        manual_names = [
            str(item.get("metadata", {}).get("manual_name", "")).strip()
            for item in results
            if item.get("metadata", {}).get("manual_name")
        ]
        for name in manual_names:
            manual_counter[name] += 1

        label = summarize_hit(top1_score, keyword_coverage, len(results))
        label_counter[label] += 1

        # 召回率计算
        recall_5 = compute_recall_at_k(results, 5)
        recall_3 = compute_recall_at_k(results, 3)

        # 更新路由分组统计
        route = question.inferred_route
        route_stats[route]["total"] += 1
        route_stats[route][label] += 1
        route_stats[route]["recalled"] += int(recall_5)
        route_stats[route]["recalled_top3"] += int(recall_3)

        question_rows.append({
            "id": question.question_id,
            "question": question.normalized_question,
            "keywords": "|".join(keywords),
            "matched_keywords": "|".join(matched_keywords),
            "keyword_coverage": round(keyword_coverage, 4),
            "result_count": len(results),
            "top1_score": round(top1_score, 4),
            "avg_score": round(avg_score, 4),
            "recall@5": round(recall_5, 4),
            "recall@3": round(recall_3, 4),
            "inferred_route": route,
            "top_manuals": "|".join(manual_names[:3]),
            "top_image_ids": "|".join(results[0].get("image_ids", [])[:5]) if results else "",
            "hit_quality": label,
        })

        for rank, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})
            detail_rows.append({
                "id": question.question_id,
                "rank": rank,
                "question": question.normalized_question,
                "score": round(item.get("relevance_score", 0.0), 6),
                "manual_name": metadata.get("manual_name", ""),
                "source_file": metadata.get("source_file", ""),
                "section_title": metadata.get("section_title", ""),
                "section_index": metadata.get("section_index", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "doc_id": item.get("doc_id", ""),
                "image_ids": "|".join(item.get("image_ids", [])),
                "preview": clean_preview(item.get("content", ""), preview_chars),
            })

    summary = build_summary(
        question_rows, detail_rows, manual_counter, label_counter,
        route_stats, use_rerank, top_k,
    )
    return {
        "summary": summary,
        "questions": question_rows,
        "details": detail_rows,
    }


def build_summary(
    question_rows: Sequence[Dict[str, Any]],
    detail_rows: Sequence[Dict[str, Any]],
    manual_counter: Counter[str],
    label_counter: Counter[str],
    route_stats: Dict[str, Dict[str, int]],
    use_rerank: bool,
    top_k: int,
) -> Dict[str, Any]:
    """构建汇总报告，包含分层召回率指标。"""
    top1_scores = [float(row["top1_score"]) for row in question_rows]
    coverages = [float(row["keyword_coverage"]) for row in question_rows]
    recall5_list = [float(row["recall@5"]) for row in question_rows]
    recall3_list = [float(row["recall@3"]) for row in question_rows]

    # 低命中示例
    low_examples = [
        {
            "id": row["id"],
            "question": row["question"],
            "top1_score": row["top1_score"],
            "keyword_coverage": row["keyword_coverage"],
            "recall@5": row["recall@5"],
            "top_manuals": row["top_manuals"],
            "inferred_route": row["inferred_route"],
        }
        for row in question_rows
        if row["hit_quality"] == "low"
    ][:20]

    # 分层召回率报告
    stratified_recall: Dict[str, Dict[str, Any]] = {}
    for route, stats in sorted(route_stats.items()):
        total = stats["total"]
        if total > 0:
            stratified_recall[route] = {
                "count": total,
                "recall@5": round(statistics.fmean([
                    float(row["recall@5"]) for row in question_rows
                    if row["inferred_route"] == route
                ]), 4),
                "recall@3": round(statistics.fmean([
                    float(row["recall@3"]) for row in question_rows
                    if row["inferred_route"] == route
                ]), 4),
                "avg_top1_score": round(statistics.fmean([
                    float(row["top1_score"]) for row in question_rows
                    if row["inferred_route"] == route
                ]), 4),
                "avg_keyword_coverage": round(statistics.fmean([
                    float(row["keyword_coverage"]) for row in question_rows
                    if row["inferred_route"] == route
                ]), 4),
                "high_ratio": round(stats["high"] / total, 4),
                "medium_ratio": round(stats["medium"] / total, 4),
                "low_ratio": round(stats["low"] / total, 4),
                "empty_ratio": round(stats["empty"] / total, 4),
            }

    return {
        "question_count": len(question_rows),
        "detail_row_count": len(detail_rows),
        "top_k": top_k,
        "use_rerank": use_rerank,
        # 整体指标
        "avg_top1_score": round(statistics.fmean(top1_scores), 4) if top1_scores else 0.0,
        "median_top1_score": round(statistics.median(top1_scores), 4) if top1_scores else 0.0,
        "avg_keyword_coverage": round(statistics.fmean(coverages), 4) if coverages else 0.0,
        "avg_recall@5": round(statistics.fmean(recall5_list), 4) if recall5_list else 0.0,
        "avg_recall@3": round(statistics.fmean(recall3_list), 4) if recall3_list else 0.0,
        # 命中质量分布
        "hit_quality_distribution": dict(label_counter),
        # 分层召回率
        "stratified_recall": stratified_recall,
        # 高频命中手册
        "top_manuals": manual_counter.most_common(20),
        # 低命中示例
        "low_hit_examples": low_examples,
    }


def compare_results(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    对比两套评测结果，返回差值表。

    用于判断新模型（BGE-M3）相比旧模型（MiniLM）的改进幅度。
    正值表示新模型更好。
    """
    current_sum = current["summary"]
    baseline_sum = baseline["summary"]

    def safe_diff(key: str) -> Any:
        c = current_sum.get(key, 0)
        b = baseline_sum.get(key, 0)
        if isinstance(c, float) and isinstance(b, float):
            return round(c - b, 4)
        return None

    comparison = {
        "avg_top1_score": safe_diff("avg_top1_score"),
        "median_top1_score": safe_diff("median_top1_score"),
        "avg_keyword_coverage": safe_diff("avg_keyword_coverage"),
        "avg_recall@5": safe_diff("avg_recall@5"),
        "avg_recall@3": safe_diff("avg_recall@3"),
    }

    # 分层对比
    comparison["stratified_recall"] = {}
    current_routes = current_sum.get("stratified_recall", {})
    baseline_routes = baseline_sum.get("stratified_recall", {})
    all_routes = set(current_routes.keys()) | set(baseline_routes.keys())
    for route in sorted(all_routes):
        c_recall5 = current_routes.get(route, {}).get("recall@5", 0)
        b_recall5 = baseline_routes.get(route, {}).get("recall@5", 0)
        c_recall3 = current_routes.get(route, {}).get("recall@3", 0)
        b_recall3 = baseline_routes.get(route, {}).get("recall@3", 0)
        comparison["stratified_recall"][route] = {
            "recall@5": round(c_recall5 - b_recall5, 4),
            "recall@3": round(c_recall3 - b_recall3, 4),
        }

    # 命中质量分布对比
    comparison["hit_quality_delta"] = {}
    c_dist = current_sum.get("hit_quality_distribution", {})
    b_dist = baseline_sum.get("hit_quality_distribution", {})
    for label in ["high", "medium", "low", "empty"]:
        c_count = c_dist.get(label, 0)
        b_count = b_dist.get(label, 0)
        comparison["hit_quality_delta"][label] = c_count - b_count

    return comparison


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, payload: Dict[str, Any], comparison: Optional[Dict[str, Any]] = None) -> None:
    """生成 Markdown 评测报告，支持 AB 对比。"""
    summary = payload["summary"]
    lines = [
        "# 公开题检索评测摘要",
        "",
        f"- 题目数: {summary['question_count']}",
        f"- 明细行数: {summary['detail_row_count']}",
        f"- Top-K: {summary['top_k']}",
        f"- 是否启用 rerank: {summary['use_rerank']}",
        f"- 平均 Top1 分数: {summary['avg_top1_score']}",
        f"- Top1 分数中位数: {summary['median_top1_score']}",
        f"- 平均关键词覆盖度: {summary['avg_keyword_coverage']}",
        f"- 平均 Recall@5: {summary['avg_recall@5']}",
        f"- 平均 Recall@3: {summary['avg_recall@3']}",
        "",
        "## 命中质量分布",
        "",
    ]

    for label, count in summary["hit_quality_distribution"].items():
        lines.append(f"- {label}: {count}")

    # 分层召回率
    stratified = summary.get("stratified_recall", {})
    if stratified:
        lines.extend(["", "## 分层召回率（按路由类型）", ""])
        lines.append("| 路由 | 数量 | Recall@5 | Recall@3 | Top1 均分 | 高质量占比 | 中质量占比 | 低质量占比 |")
        lines.append("|------|------|----------|----------|-----------|------------|------------|------------|")
        for route, stats in stratified.items():
            route_label = route or "unknown"
            lines.append(
                f"| {route_label} | {stats['count']} | "
                f"{stats['recall@5']} | {stats['recall@3']} | "
                f"{stats['avg_top1_score']} | {stats['high_ratio']} | "
                f"{stats['medium_ratio']} | {stats['low_ratio']} |"
            )

    # AB 对比（若存在）
    if comparison:
        lines.extend(["", "## AB 模型对比（当前 vs 基线）", ""])
        for key, val in comparison.items():
            if key == "stratified_recall":
                continue
            if key == "hit_quality_delta":
                lines.extend(["", "### 命中质量变化", ""])
                for label, delta in val.items():
                    sign = "+" if delta > 0 else ""
                    lines.append(f"- {label}: {sign}{delta}")
                continue
            if val is not None:
                sign = "+" if val > 0 else ""
                lines.append(f"- {key}: {sign}{val}")

        if "stratified_recall" in comparison:
            lines.extend(["", "### 分层召回率变化", ""])
            lines.append("| 路由 | Recall@5 变化 | Recall@3 变化 |")
            lines.append("|------|--------------|--------------|")
            for route, delta in comparison["stratified_recall"].items():
                r5 = delta.get("recall@5", 0)
                r3 = delta.get("recall@3", 0)
                sign5 = "+" if r5 > 0 else ""
                sign3 = "+" if r3 > 0 else ""
                lines.append(f"| {route} | {sign5}{r5} | {sign3}{r3} |")

    lines.extend([
        "",
        "## 高频命中手册",
        "",
    ])

    for manual_name, count in summary["top_manuals"]:
        lines.append(f"- {manual_name}: {count}")

    lines.extend([
        "",
        "## 低命中示例（前20条）",
        "",
    ])

    for item in summary["low_hit_examples"]:
        lines.append(
            f"- id={item['id']} recall@5={item['recall@5']} "
            f"top1={item['top1_score']} coverage={item['keyword_coverage']} "
            f"route={item['inferred_route']} manuals={item['top_manuals']} "
            f"question={item['question'].replace(chr(10), ' / ')}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def run_single_eval(
    questions: List[QuestionRecord],
    top_k: int,
    use_rerank: bool,
    preview_chars: int,
    enable_hybrid: bool,
    output_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """执行一次评测并写入输出文件。"""
    payload = run_retrieval(questions, top_k, use_rerank, preview_chars, enable_hybrid)
    write_csv(output_dir / f"questions_{output_name}.csv", payload["questions"])
    write_csv(output_dir / f"details_{output_name}.csv", payload["details"])
    write_json(output_dir / f"summary_{output_name}.json", payload["summary"])
    return payload


def main() -> None:
    args = parse_args()

    if args.embedding_backend:
        update_settings(embedding_backend=args.embedding_backend)

    question_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    questions = load_questions(question_path, limit=args.limit)

    # --- 当前模型评测 ---
    print(f"[评测] 当前模型评测，题目数: {len(questions)}")
    current_payload = run_single_eval(
        questions=questions,
        top_k=args.top_k,
        use_rerank=args.rerank,
        preview_chars=args.preview_chars,
        enable_hybrid=args.hybrid,
        output_name="current",
        output_dir=output_dir,
    )

    # --- 基线模型评测（AB 对比模式）---
    comparison: Optional[Dict[str, Any]] = None
    if args.baseline_model:
        baseline_dir = Path(args.baseline_output_dir) if args.baseline_output_dir else output_dir / "baseline"
        ensure_dir(baseline_dir)

        # 保存当前模型结果
        save_current = {
            "summary": current_payload["summary"],
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "current_summary.json", save_current)

        print(f"[AB对比] 加载基线模型配置: {args.baseline_model}")
        print(f"[AB对比] 基线结果将写入: {baseline_dir}")

        # 临时切换 embedding 模型
        old_model = settings.embedding_model
        update_settings(embedding_model=args.baseline_model)

        baseline_payload = run_single_eval(
            questions=questions,
            top_k=args.top_k,
            use_rerank=args.rerank,
            preview_chars=args.preview_chars,
            enable_hybrid=False,  # 基线模型对比时关闭混合检索
            output_name="baseline",
            output_dir=baseline_dir,
        )

        # 恢复原模型
        update_settings(embedding_model=old_model)

        # 计算对比
        comparison = compare_results(current_payload, baseline_payload)
        write_json(output_dir / "ab_comparison.json", comparison)

        print("\n=== AB 对比结果 ===")
        for key, val in comparison.items():
            if key in ("stratified_recall", "hit_quality_delta"):
                continue
            if val is not None:
                sign = "+" if val > 0 else ""
                print(f"  {key}: {sign}{val}")
        if comparison.get("hit_quality_delta"):
            print("  命中质量变化:")
            for label, delta in comparison["hit_quality_delta"].items():
                sign = "+" if delta > 0 else ""
                print(f"    {label}: {sign}{delta}")

    # 写最终 Markdown 报告
    write_markdown(output_dir / "retrieval_summary.md", current_payload, comparison)

    # 输出汇总
    summary = current_payload["summary"]
    print(f"\n=== 评测完成 ===")
    print(f"输出目录: {output_dir}")
    print(f"题目数: {summary['question_count']}")
    print(f"平均 Top1 分数: {summary['avg_top1_score']}")
    print(f"平均 Recall@5: {summary['avg_recall@5']}")
    print(f"平均关键词覆盖度: {summary['avg_keyword_coverage']}")
    print(f"命中质量分布: {summary['hit_quality_distribution']}")

    if summary.get("stratified_recall"):
        print("\n分层召回率:")
        for route, stats in summary["stratified_recall"].items():
            print(
                f"  {route}: recall@5={stats['recall@5']} "
                f"recall@3={stats['recall@3']} "
                f"count={stats['count']}"
            )


if __name__ == "__main__":
    main()
