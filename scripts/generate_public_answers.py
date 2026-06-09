"""
批量生成公开题答案，并输出提交文件与质量画像。

目标：
1. 读取 question_public.csv，调用当前 /chat 主链路的核心回答器批量生成答案；
2. 产出符合 submission_example.csv 格式的提交文件；
3. 输出便于人工复盘的质量分析摘要与明细。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from config import settings
from src.modules.response_generator import ResponseGenerator
from scripts.build_dual_route_kb import normalize_question


def load_questions(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "id": str(row["id"]).strip(),
                    "question": normalize_question(row["question"]),
                }
            )
    return rows


def dominant_route(route_records: List[Dict[str, Any]]) -> str:
    if not route_records:
        return "unknown"
    counts = Counter(item.get("route", "manual") for item in route_records)
    return counts.most_common(1)[0][0]


def contains_numbered_structure(answer: str) -> bool:
    return any(token in answer for token in ["1.", "2.", "3.", "1、", "2、"])


def significant_terms(text: str) -> List[str]:
    import re

    text = (text or "").strip().lower()
    if not text:
        return []

    terms = set(re.findall(r"[a-z0-9_-]+", text))
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(segment) <= 4:
            terms.add(segment)
        for idx in range(len(segment) - 1):
            terms.add(segment[idx: idx + 2])
    return sorted(terms)


def keyword_coverage(question: str, answer: str) -> float:
    q_terms = [term for term in significant_terms(question) if len(term) >= 2]
    if not q_terms:
        return 0.0
    hits = sum(1 for term in q_terms if term in answer)
    return round(hits / len(q_terms), 4)


def risk_level(record: Dict[str, Any]) -> str:
    """
    评估单条回答的风险等级。

    风险等级定义:
    - high: 明显存在回答质量问题，需要重点人工复盘
    - medium: 存在潜在问题，建议抽检
    - low: 回答质量良好，可信度高

    触发high风险的条件:
    - 使用了fallback兜底回答（说明检索失败）
    - 置信度<0.55（模型对答案缺乏信心）
    - 服务型问题被主判为manual路由（路由可能出错）

    触发medium风险的条件:
    - 多问题但回答无编号结构（可能遗漏部分问题）
    - 置信度<0.72（信心不足）
    - 关键词覆盖率<0.08（回答与问题关联度低）

    Args:
        record: build_analysis_record构建的答案分析记录

    Returns:
        风险等级字符串: "high", "medium", 或 "low"
    """
    if record["used_fallback"]:
        return "high"
    if record["confidence"] < 0.55:
        return "high"
    if record["dominant_route"] == "manual" and record["service_like_question"]:
        return "high"
    if record["sub_question_count"] >= 2 and not record["has_numbered_structure"]:
        return "medium"
    if record["confidence"] < 0.72:
        return "medium"
    if record["keyword_coverage"] < 0.08:
        return "medium"
    return "low"


def is_service_like_question(question: str) -> bool:
    """
    判断问题是否属于服务型问题（需要客服介入处理）。

    服务型问题特征:
    - 涉及退款/退货/换货等售后诉求
    - 涉及发票/优惠等财务操作
    - 涉及物流/安装/维修等履约服务
    - 涉及投诉/赔偿等争议处理

    这些问题通常需要客服人工介入核实订单信息，
    而非仅靠产品手册即可回答。

    Args:
        question: 用户问题文本

    Returns:
        是否为服务型问题
    """
    service_clues = [
        "退款", "退货", "换货", "发票", "售后", "维修", "物流", "运费", "安装",
        "投诉", "赔偿", "保修", "质保", "补寄", "签收", "破损", "客服", "发货",
        "无理由", "抬头", "税号", "企业采购", "翻新", "假货", "优惠券", "以旧换新",
    ]
    return any(clue in question for clue in service_clues)


def build_analysis_record(row: Dict[str, str], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    构建单条答案的完整分析记录。

    分析维度:
    - 基础信息: id, question, answer, answer_chars
    - 路由统计: dominant_route及各路由数量
    - 置信度: 模型返回的confidence分数
    - 覆盖率: 回答对问题关键词的覆盖程度
    - 结构特征: 是否含编号、是否含图片标记
    - 分类器: 是否使用了ONNX路由分类器
    - 风险评估: 综合以上维度给出风险等级

    分析记录用于:
    1. 批量生成提交文件
    2. 质量画像统计
    3. 高风险样本人工复盘

    Args:
        row: 原始题目数据
        result: 回答生成器的返回结果

    Returns:
        包含所有分析维度的字典
    """
    answer = result["response"]
    route_records = result.get("routes", [])
    sub_question_count = len(route_records) or 1
    dominant = dominant_route(route_records)
    # 统计各路由的数量
    service_route_count = sum(1 for item in route_records if item.get("route") == "service")
    mixed_route_count = sum(1 for item in route_records if item.get("route") == "mixed")
    manual_route_count = sum(1 for item in route_records if item.get("route") == "manual")
    coverage = keyword_coverage(row["question"], answer)
    # 检测是否使用了fallback兜底回答（优先使用显式标志）
    used_fallback = result.get("used_fallback", False)
    # 统计分类器使用情况
    classifier_used_count = sum(1 for item in route_records if item.get("classifier_used"))
    classifier_labels = [item.get("classifier_label") for item in route_records if item.get("classifier_label")]
    classifier_confidences = [float(item.get("classifier_confidence", 0.0)) for item in route_records if item.get("classifier_label")]

    record = {
        "id": row["id"],
        "question": row["question"],
        "answer": answer,
        "confidence": round(float(result.get("confidence", 0.0)), 4),
        "dominant_route": dominant,
        "service_route_count": service_route_count,
        "mixed_route_count": mixed_route_count,
        "manual_route_count": manual_route_count,
        "sub_question_count": sub_question_count,
        "answer_chars": len(answer),
        "image_count": len(result.get("images", [])),
        "has_pic_marker": "<PIC>" in answer,
        "has_numbered_structure": contains_numbered_structure(answer),
        "keyword_coverage": coverage,
        "used_fallback": used_fallback,
        "service_like_question": is_service_like_question(row["question"]),
        "classifier_used_count": classifier_used_count,
        "classifier_dominant_label": Counter(classifier_labels).most_common(1)[0][0] if classifier_labels else "",
        "classifier_avg_confidence": round(statistics.mean(classifier_confidences), 4) if classifier_confidences else 0.0,
        "route_records": route_records,
    }
    record["risk_level"] = risk_level(record)
    return record


def load_detail_rows(detail_path: Path) -> List[Dict[str, Any]]:
    """读取已有明细文件，返回所有分析记录（用于续跑后重新生成完整摘要）。"""
    rows: List[Dict[str, Any]] = []
    if not detail_path.exists():
        return rows
    with detail_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["confidence"] = float(row["confidence"])
            row["keyword_coverage"] = float(row["keyword_coverage"])
            row["answer_chars"] = int(row["answer_chars"])
            row["image_count"] = int(row["image_count"])
            row["has_pic_marker"] = row["has_pic_marker"] == "True"
            row["has_numbered_structure"] = row["has_numbered_structure"] == "True"
            row["used_fallback"] = row["used_fallback"] == "True"
            row["service_like_question"] = row["service_like_question"] == "True"
            row["sub_question_count"] = int(row["sub_question_count"])
            row["service_route_count"] = int(row["service_route_count"])
            row["mixed_route_count"] = int(row["mixed_route_count"])
            row["manual_route_count"] = int(row["manual_route_count"])
            row["classifier_used_count"] = int(row["classifier_used_count"])
            row["classifier_avg_confidence"] = float(row["classifier_avg_confidence"])
            row["route_records"] = []
            row["risk_level"] = risk_level(row)
            rows.append(row)
    return rows


def write_submission_row(path: Path, record: Dict[str, Any]) -> None:
    """追加单条答案到提交文件（增量写入）。"""
    file_exists = path.exists()
    # utf-8-sig 追加时只在文件开头写一次 BOM，后续追加不会重复写 BOM
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(
            handle,
            quoting=csv.QUOTE_MINIMAL,    # 只在必要时加引号，与示例文件一致
            quotechar='"',
            doublequote=True,
            lineterminator="\n",
        )
        if not file_exists:
            writer.writerow(["id", "ret"])
        writer.writerow([record["id"], record["answer"]])


def write_detail_row(path: Path, record: Dict[str, Any]) -> None:
    """追加单条分析记录到明细文件（增量写入）。"""
    fieldnames = [
        "id", "dominant_route", "confidence", "risk_level",
        "sub_question_count", "service_route_count", "mixed_route_count",
        "manual_route_count", "answer_chars", "image_count",
        "has_pic_marker", "has_numbered_structure", "keyword_coverage",
        "used_fallback", "service_like_question",
        "classifier_used_count", "classifier_dominant_label",
        "classifier_avg_confidence", "question", "answer",
    ]
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator="\n")
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: record[key] for key in fieldnames})


def write_timing_row(path: Path, record: Dict[str, Any]) -> None:
    """追加单条计时记录到文件（增量写入）。"""
    fieldnames = ["id", "total", "step1", "step2", "step3", "step4", "step5", "step6"]
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator="\n")
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: record[key] for key in fieldnames})


def build_summary(rows: List[Dict[str, Any]], settings_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    confidences = [row["confidence"] for row in rows]
    route_counter = Counter(row["dominant_route"] for row in rows)
    risk_counter = Counter(row["risk_level"] for row in rows)
    fallback_count = sum(1 for row in rows if row["used_fallback"])
    multi_question_count = sum(1 for row in rows if row["sub_question_count"] >= 2)
    pic_count = sum(1 for row in rows if row["has_pic_marker"])
    classifier_used_count = sum(1 for row in rows if row["classifier_used_count"] > 0)
    manual_on_service_like = [
        row["id"] for row in rows
        if row["dominant_route"] == "manual" and row["service_like_question"]
    ]

    # 计时统计（如果 timing_rows 被传入）
    timing_stats = {}

    summary = {
        "settings": settings_snapshot,
        "question_count": len(rows),
        "avg_confidence": round(statistics.mean(confidences), 4) if confidences else 0.0,
        "median_confidence": round(statistics.median(confidences), 4) if confidences else 0.0,
        "avg_answer_chars": round(statistics.mean(row["answer_chars"] for row in rows), 2) if rows else 0.0,
        "avg_keyword_coverage": round(statistics.mean(row["keyword_coverage"] for row in rows), 4) if rows else 0.0,
        "route_distribution": dict(route_counter),
        "risk_distribution": dict(risk_counter),
        "fallback_count": fallback_count,
        "multi_question_count": multi_question_count,
        "answers_with_pic_marker": pic_count,
        "classifier_used_question_count": classifier_used_count,
        "manual_on_service_like_count": len(manual_on_service_like),
        "manual_on_service_like_ids": manual_on_service_like[:50],
        "high_risk_examples": [
            {
                "id": row["id"],
                "route": row["dominant_route"],
                "confidence": row["confidence"],
                "question": row["question"],
                "answer_preview": row["answer"][:180],
            }
            for row in rows
            if row["risk_level"] == "high"
        ][:20],
    }
    return summary


def build_timing_summary(timing_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """根据计时记录生成统计摘要。"""
    if not timing_rows:
        return {}

    totals = [r["total"] for r in timing_rows]
    step1s = [r["step1"] for r in timing_rows]
    step2s = [r["step2"] for r in timing_rows]
    step3s = [r["step3"] for r in timing_rows]
    step4s = [r["step4"] for r in timing_rows]
    step5s = [r["step5"] for r in timing_rows]
    step6s = [r["step6"] for r in timing_rows]

    sorted_totals = sorted(totals)

    return {
        "count": len(timing_rows),
        "total": {
            "avg": round(statistics.mean(totals), 3),
            "median": round(statistics.median(totals), 3),
            "min": round(min(totals), 3),
            "max": round(max(totals), 3),
            "p95": round(sorted_totals[int(len(sorted_totals) * 0.95)], 3),
            "p99": round(sorted_totals[int(len(sorted_totals) * 0.99)], 3),
        },
        "step1_decomposition": {"avg": round(statistics.mean(step1s), 3)},
        "step2_retrieval": {"avg": round(statistics.mean(step2s), 3)},
        "step3_context": {"avg": round(statistics.mean(step3s), 3)},
        "step4_generation": {"avg": round(statistics.mean(step4s), 3)},
        "step5_hallucination": {"avg": round(statistics.mean(step5s), 3)},
        "step6_image": {"avg": round(statistics.mean(step6s), 3)},
    }


def write_summary_md(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    high_risk_rows = [row for row in rows if row["risk_level"] == "high"][:20]
    mixed_rows = [row for row in rows if row["dominant_route"] == "mixed"][:10]

    lines = [
        "# question_public 批量答题质量摘要",
        "",
        f"- 题目数: `{summary['question_count']}`",
        f"- LLM provider: `{summary['settings']['llm_provider']}`",
        f"- 平均置信度: `{summary['avg_confidence']}`",
        f"- 中位置信度: `{summary['median_confidence']}`",
        f"- 平均回答长度: `{summary['avg_answer_chars']}`",
        f"- 平均关键词覆盖度: `{summary['avg_keyword_coverage']}`",
        f"- fallback 数: `{summary['fallback_count']}`",
        f"- 多问题题数: `{summary['multi_question_count']}`",
        f"- 含 `<PIC>` 回答数: `{summary['answers_with_pic_marker']}`",
        f"- 使用分类器的题数: `{summary['classifier_used_question_count']}`",
        f"- 服务型问题却被主判为 manual 的题数: `{summary['manual_on_service_like_count']}`",
        "",
        "## 路由分布",
    ]
    for route, count in summary["route_distribution"].items():
        lines.append(f"- {route}: `{count}`")

    lines.extend(["", "## 风险分布"])
    for risk, count in summary["risk_distribution"].items():
        lines.append(f"- {risk}: `{count}`")

    lines.extend(["", "## Mixed 示例"])
    for row in mixed_rows:
        lines.append(f"- id={row['id']} confidence={row['confidence']} question={row['question'].replace(chr(10), ' / ')}")
        lines.append(f"  answer={row['answer'][:220]}")

    lines.extend(["", "## 高风险示例"])
    for row in high_risk_rows:
        lines.append(f"- id={row['id']} route={row['dominant_route']} confidence={row['confidence']} question={row['question'].replace(chr(10), ' / ')}")
        lines.append(f"  answer={row['answer'][:220]}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成公开题答案与质量画像")
    parser.add_argument("--questions", default="question_public.csv", help="公开题 CSV 路径")
    parser.add_argument("--output-dir", default=str(Path("knowledge_base") / "evaluation_v1"), help="输出目录")
    parser.add_argument("--submission-name", default="submission_public_generated.csv", help="提交 CSV 文件名")
    parser.add_argument("--disable-llm", action="store_true", help="关闭外部 LLM，使用检索驱动兜底回答")
    parser.add_argument("--resume", action="store_true", default=True, help="启用断点续跑，自动跳过已处理的题目（默认开启）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点续跑，从头开始处理")
    args = parser.parse_args()

    if args.disable_llm:
        settings.llm_provider = "disabled"
        settings.enable_cot_reasoning = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / args.submission_name
    detail_path = output_dir / "public_answer_detail.csv"
    summary_json_path = output_dir / "public_answer_summary.json"
    summary_md_path = output_dir / "public_answer_summary.md"
    timing_path = output_dir / "timing_stats.csv"

    rows = load_questions(Path(args.questions))
    generator = ResponseGenerator()
    generator.initialize()

    existing_rows: List[Dict[str, Any]] = []
    existing_timing_rows: List[Dict[str, Any]] = []

    # 加载已有的 timing 文件（用于续跑）
    if args.resume and timing_path.exists():
        with timing_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["total"] = float(row["total"])
                row["step1"] = float(row["step1"])
                row["step2"] = float(row["step2"])
                row["step3"] = float(row["step3"])
                row["step4"] = float(row["step4"])
                row["step5"] = float(row["step5"])
                row["step6"] = float(row["step6"])
                existing_timing_rows.append(row)

    if args.resume:
        existing_rows = load_detail_rows(detail_path)
        if existing_rows:
            print(f"[断点续跑] 检测到已有 {len(existing_rows)} 条结果，将跳过已处理题目")

    done_ids: Set[str] = {row["id"] for row in existing_rows}
    timing_done_ids: Set[str] = {row["id"] for row in existing_timing_rows}
    total = len(rows)

    analysis_rows: List[Dict[str, Any]] = list(existing_rows)
    timing_rows: List[Dict[str, Any]] = list(existing_timing_rows)
    error_rows: List[Dict[str, Any]] = [row for row in existing_rows if row["risk_level"] == "high"]

    for idx, row in enumerate(rows):
        if row["id"] in done_ids:
            continue
        try:
            start_time = time.time()
            result = generator.generate(row["question"])
            elapsed = time.time() - start_time

            # 提取计时信息
            timing = result.get("timing", {})
            timing_record = {
                "id": row["id"],
                "total": elapsed,
                "step1": timing.get("step1_decomposition", 0),
                "step2": timing.get("step2_retrieval", 0),
                "step3": timing.get("step3_context", 0),
                "step4": timing.get("step4_generation", 0),
                "step5": timing.get("step5_hallucination", 0),
                "step6": timing.get("step6_image", 0),
            }
            timing_rows.append(timing_record)

            record = build_analysis_record(row, result)
            analysis_rows.append(record)
            write_submission_row(submission_path, record)
            write_detail_row(detail_path, record)
            write_timing_row(timing_path, timing_record)
            print(f"[{idx + 1}/{total}] 完成 id={row['id']} confidence={record['confidence']:.2f} 耗时={elapsed:.2f}s")
        except Exception as e:
            record = {
                "id": row["id"],
                "question": row["question"],
                "answer": f"[生成失败，请人工处理] {type(e).__name__}: {e}",
                "confidence": 0.0,
                "dominant_route": "error",
                "service_route_count": 0,
                "mixed_route_count": 0,
                "manual_route_count": 0,
                "sub_question_count": 0,
                "answer_chars": 0,
                "image_count": 0,
                "has_pic_marker": False,
                "has_numbered_structure": False,
                "keyword_coverage": 0.0,
                "used_fallback": False,
                "service_like_question": is_service_like_question(row["question"]),
                "classifier_used_count": 0,
                "classifier_dominant_label": "",
                "classifier_avg_confidence": 0.0,
                "route_records": [],
                "risk_level": "high",
            }
            error_rows.append(record)
            analysis_rows.append(record)
            write_submission_row(submission_path, record)
            write_detail_row(detail_path, record)
            print(f"[{idx + 1}/{total}] 失败 id={row['id']} ({type(e).__name__}: {e})")

    new_errors = [row for row in error_rows if row["dominant_route"] == "error"]
    if new_errors:
        print(f"[汇总] 本次新增 {len(new_errors)} 题生成失败，已追加到结果文件中")

    summary = build_summary(
        analysis_rows,
        settings_snapshot={
            "llm_provider": settings.llm_provider,
            "embedding_backend": settings.embedding_backend,
            "rag_top_k": settings.rag_top_k,
        },
    )

    # 生成计时统计摘要
    timing_summary = build_timing_summary(timing_rows)
    timing_summary_path = output_dir / "timing_summary.json"
    timing_summary_path.write_text(json.dumps(timing_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(summary_md_path, summary, analysis_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n=== 计时统计 ===")
    if timing_summary:
        print(f"总题数: {timing_summary['count']}")
        print(f"总耗时 - 平均: {timing_summary['total']['avg']:.3f}s, 中位数: {timing_summary['total']['median']:.3f}s, P95: {timing_summary['total']['p95']:.3f}s, 最大: {timing_summary['total']['max']:.3f}s")
        print(f"  Step1 问题分解: {timing_summary['step1_decomposition']['avg']:.3f}s")
        print(f"  Step2 RAG检索: {timing_summary['step2_retrieval']['avg']:.3f}s")
        print(f"  Step3 上下文构建: {timing_summary['step3_context']['avg']:.3f}s")
        print(f"  Step4 回答生成: {timing_summary['step4_generation']['avg']:.3f}s")
        print(f"  Step5 幻觉检测: {timing_summary['step5_hallucination']['avg']:.3f}s")
        print(f"  Step6 图片处理: {timing_summary['step6_image']['avg']:.3f}s")
    else:
        print("无计时数据")
    print(f"已输出提交文件: {submission_path}")
    print(f"已输出明细文件: {detail_path}")
    print(f"已输出摘要 JSON: {summary_json_path}")
    print(f"已输出摘要 MD: {summary_md_path}")
    print(f"已输出计时明细: {timing_path}")
    print(f"已输出计时摘要: {timing_summary_path}")


if __name__ == "__main__":
    main()
