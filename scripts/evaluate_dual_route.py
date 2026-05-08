"""
双路检索评测脚本 - 评估 DualRouteRetriever 的效果

用途：
1. 使用 DualRouteRetriever 进行检索
2. 评估路由准确率和各类检索质量
3. 对比纯规则路由和小模型路由的效果
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, update_settings
from src.modules.dual_route_retriever import get_dual_route_retriever
from src.modules.rag_engine import reset_rag_engine


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "knowledge_base" / "evaluation"
DEFAULT_QUESTION_FILE = PROJECT_ROOT / "question_public.csv"

STOPWORDS = {
    "请问", "一下", "你们", "我们", "商品", "这个", "那个", "什么", "怎么", "如何",
    "可以", "是否", "一下子", "一下呢", "一下吗", "吗", "呢", "呀", "啊",
    "了", "的", "和", "与", "或", "及", "再", "还", "有", "被", "把", "在",
    "是", "我", "要", "想", "给", "后", "前",
}


@dataclass
class QuestionRecord:
    question_id: str
    raw_question: str
    normalized_question: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评估双路检索效果")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTION_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--embedding-backend", choices=["hashing", "sentence_transformer"], default=None)
    return parser.parse_args()


def normalize_question(raw_question: str) -> str:
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


def load_questions(csv_path: Path, limit: int = 0) -> List[QuestionRecord]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            record = QuestionRecord(
                question_id=str(row["id"]).strip(),
                raw_question=row["question"],
                normalized_question=normalize_question(row["question"]),
            )
            rows.append(record)
            if limit and len(rows) >= limit:
                break
    return rows


def run_dual_route_retrieval(
    questions: Sequence[QuestionRecord],
    top_k: int,
) -> Dict[str, Any]:
    reset_rag_engine()
    retriever = get_dual_route_retriever()
    retriever.initialize()

    route_counts = Counter()
    classifier_used_counts = Counter()
    service_results_count = 0
    manual_results_count = 0
    empty_results_count = 0
    question_rows = []

    for question in questions:
        result = retriever.retrieve(question.normalized_question)
        route = result.get("route_info", {}).get("route", "unknown")
        classifier_used = result.get("route_info", {}).get("classifier_used", False)
        service_res = result.get("service_results", [])
        manual_res = result.get("manual_results", [])
        all_res = result.get("results", [])

        route_counts[route] += 1
        if classifier_used:
            classifier_used_counts["used"] += 1
        else:
            classifier_used_counts["fallback"] += 1

        if route == "service":
            service_results_count += 1
        elif route == "manual":
            manual_results_count += 1

        if not all_res:
            empty_results_count += 1

        manual_names = []
        for item in result.get("manual_results", [])[:3]:
            name = str(item.get("metadata", {}).get("manual_name", "")).strip()
            if name:
                manual_names.append(name)

        service_titles = []
        for item in result.get("service_results", [])[:2]:
            title = str(item.get("metadata", {}).get("title", "")).strip()
            if title:
                service_titles.append(title)

        question_rows.append({
            "id": question.question_id,
            "question": question.normalized_question[:50],
            "route": route,
            "classifier_used": classifier_used,
            "classifier_label": result.get("route_info", {}).get("classifier_label", ""),
            "classifier_confidence": round(result.get("route_info", {}).get("classifier_confidence", 0), 4),
            "classifier_fallback_reason": result.get("route_info", {}).get("classifier_fallback_reason", ""),
            "service_count": len(service_res),
            "manual_count": len(manual_res),
            "total_count": len(all_res),
            "top_manuals": "|".join(manual_names[:3]),
            "top_service_titles": "|".join(service_titles[:2]),
        })

    summary = {
        "total_questions": len(questions),
        "route_distribution": dict(route_counts),
        "classifier_usage": dict(classifier_used_counts),
        "service_routed": service_results_count,
        "manual_routed": manual_results_count,
        "mixed_routed": route_counts.get("mixed", 0),
        "empty_results": empty_results_count,
    }
    return {
        "summary": summary,
        "questions": question_rows,
    }


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


def main() -> None:
    args = parse_args()
    if args.embedding_backend:
        update_settings(embedding_backend=args.embedding_backend)

    question_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    questions = load_questions(question_path, limit=args.limit)
    print(f"加载了 {len(questions)} 个问题")

    payload = run_dual_route_retrieval(questions=questions, top_k=args.top_k)

    write_json(output_dir / "dual_route_summary.json", payload["summary"])
    write_csv(output_dir / "dual_route_questions.csv", payload["questions"])

    print("\n" + "=" * 60)
    print("双路检索评测摘要")
    print("=" * 60)
    summary = payload["summary"]
    print(f"总问题数: {summary['total_questions']}")
    print(f"\n路由分布:")
    for route, count in summary["route_distribution"].items():
        pct = count / summary["total_questions"] * 100
        print(f"  {route}: {count} ({pct:.1f}%)")
    print(f"\n分类器使用情况:")
    for mode, count in summary["classifier_usage"].items():
        pct = count / summary["total_questions"] * 100
        print(f"  {mode}: {count} ({pct:.1f}%)")
    print(f"\n空结果数: {summary['empty_results']}")
    print(f"\n结果已保存到:")
    print(f"  {output_dir / 'dual_route_summary.json'}")
    print(f"  {output_dir / 'dual_route_questions.csv'}")


if __name__ == "__main__":
    main()
