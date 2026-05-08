"""
manual 路诊断脚本

用途：
1. 批量读取公开题或自定义问题文件；
2. 对每题执行真实 dual_route 路由；
3. 针对最终进入 manual 路的问题，输出候选手册、局部召回是否触发、
   最终 Top1 手册/章节/分数等诊断信息；
4. 生成 CSV / JSON / Markdown，便于人工复盘 manual 路短板。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.modules.dual_route_retriever import (  # noqa: E402
    get_dual_route_retriever,
    reset_dual_route_retriever,
)
from scripts.build_dual_route_kb import normalize_question  # noqa: E402


DEFAULT_QUESTION_FILE = PROJECT_ROOT / "question_public.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "knowledge_base" / "evaluation" / "manual_route_diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出 manual 路诊断结果")
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTION_FILE),
        help="题目 CSV 文件，默认 question_public.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="诊断输出目录",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅分析前 N 题，0 表示全部",
    )
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="将 mixed 路题目也输出到诊断结果中",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_questions(csv_path: Path, limit: int = 0) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "id": str(row["id"]).strip(),
                    "question": normalize_question(row["question"]),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Manual 路诊断摘要",
        "",
        f"- 输入题目数: {summary['input_question_count']}",
        f"- 输出诊断题数: {summary['diagnosed_question_count']}",
        f"- manual 路题数: {summary['manual_route_count']}",
        f"- mixed 路题数(纳入诊断): {summary['mixed_route_count']}",
        f"- 有候选手册题数: {summary['with_candidates_count']}",
        f"- 触发局部召回题数: {summary['local_recall_triggered_count']}",
        f"- Top1 为空题数: {summary['empty_top1_count']}",
        "",
        "## 高频 Top1 手册",
        "",
    ]
    for manual_name, count in summary["top1_manual_counter"]:
        lines.append(f"- {manual_name}: {count}")

    # 按 top1_score 升序、empty top1 优先排序后再输出，便于人工复盘低命中问题
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            0 if r["top1_manual_name"] else 1,  # 空 top1 优先排前面
            float(r["top1_score"]) if r["top1_score"] else 0.0,  # 再按分数升序
        )
    )

    lines.extend([
        "",
        "## 低命中样例（按分数升序，前20条）",
        "",
    ])
    for row in sorted_rows[:20]:
        lines.append(
            f"- id={row['id']} route={row['route']} local={row['local_recall_triggered']} "
            f"top1={row['top1_manual_name']} / {row['top1_section_title']} "
            f"score={row['top1_score']} candidates={row['manual_candidates']}"
        )
        lines.append(f"  question={row['question'].replace(chr(10), ' / ')}")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_diagnostic_row(
    retriever,
    question_id: str,
    question: str,
    include_mixed: bool,
) -> Dict[str, Any] | None:
    route_info = retriever.route_query(question)
    route = route_info.get("route", "unknown")
    if route not in ("manual", "mixed"):
        return None
    if route == "mixed" and not include_mixed:
        return None

    normalized = route_info.get("normalized_query", question)
    candidate_pairs = retriever._detect_manual_candidates(normalized)
    local_recall_triggered = retriever._should_use_local_manual_recall(candidate_pairs)
    result_payload = retriever.retrieve(question)
    manual_results = result_payload.get("manual_results", [])
    top1 = manual_results[0] if manual_results else {}
    top1_meta = top1.get("metadata", {})

    return {
        "id": question_id,
        "question": question,
        "route": route,
        "rule_route": route_info.get("rule_route", ""),
        "strong_rule_route": route_info.get("strong_rule_route", ""),
        "classifier_label": route_info.get("classifier_label", ""),
        "classifier_confidence": round(float(route_info.get("classifier_confidence", 0.0)), 4),
        "classifier_used": route_info.get("classifier_used", False),
        "manual_candidates": " | ".join(
            f"{manual_name}:{score:.1f}" for manual_name, score in candidate_pairs[:5]
        ),
        "manual_candidate_count": len(candidate_pairs),
        "local_recall_triggered": local_recall_triggered,
        "manual_result_count": len(manual_results),
        "top1_doc_id": top1.get("doc_id", ""),
        "top1_manual_name": top1_meta.get("manual_name", ""),
        "top1_section_title": top1_meta.get("section_title", ""),
        "top1_score": round(float(top1.get("relevance_score", 0.0)), 6),
        "top1_image_ids": "|".join(top1.get("image_ids", [])),
        "top3_manuals": " | ".join(
            str(item.get("metadata", {}).get("manual_name", "")).strip()
            for item in manual_results[:3]
            if item.get("metadata", {}).get("manual_name")
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    questions = load_questions(Path(args.questions), limit=args.limit)
    reset_dual_route_retriever()
    retriever = get_dual_route_retriever()
    retriever.initialize()

    rows: List[Dict[str, Any]] = []
    top1_counter: Counter[str] = Counter()
    manual_route_count = 0
    mixed_route_count = 0
    local_recall_count = 0
    candidate_count = 0
    empty_top1_count = 0

    for row in questions:
        diag = build_diagnostic_row(
            retriever=retriever,
            question_id=row["id"],
            question=row["question"],
            include_mixed=args.include_mixed,
        )
        if not diag:
            continue
        rows.append(diag)
        if diag["route"] == "manual":
            manual_route_count += 1
        if diag["route"] == "mixed":
            mixed_route_count += 1
        if diag["manual_candidate_count"] > 0:
            candidate_count += 1
        if diag["local_recall_triggered"]:
            local_recall_count += 1
        if diag["top1_manual_name"]:
            top1_counter[diag["top1_manual_name"]] += 1
        else:
            empty_top1_count += 1

    summary = {
        "input_question_count": len(questions),
        "diagnosed_question_count": len(rows),
        "manual_route_count": manual_route_count,
        "mixed_route_count": mixed_route_count,
        "with_candidates_count": candidate_count,
        "local_recall_triggered_count": local_recall_count,
        "empty_top1_count": empty_top1_count,
        "top1_manual_counter": top1_counter.most_common(20),
    }

    write_csv(output_dir / "manual_route_diagnostic.csv", rows)
    write_json(output_dir / "manual_route_diagnostic_summary.json", summary)
    write_markdown(output_dir / "manual_route_diagnostic_summary.md", summary, rows)

    print("=== Manual 路诊断完成 ===")
    print(f"输出目录: {output_dir}")
    print(f"输入题目数: {summary['input_question_count']}")
    print(f"输出诊断题数: {summary['diagnosed_question_count']}")
    print(f"manual 路题数: {summary['manual_route_count']}")
    print(f"mixed 路题数: {summary['mixed_route_count']}")
    print(f"有候选手册题数: {summary['with_candidates_count']}")
    print(f"触发局部召回题数: {summary['local_recall_triggered_count']}")
    print(f"Top1 为空题数: {summary['empty_top1_count']}")


if __name__ == "__main__":
    main()
