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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

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
    # 检测是否使用了fallback兜底回答
    used_fallback = "抱歉，我暂时没有检索到足够的参考信息" in answer
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


def write_submission_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "ret"])
        for row in rows:
            writer.writerow([row["id"], row["answer"]])


def write_detail_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "dominant_route",
        "confidence",
        "risk_level",
        "sub_question_count",
        "service_route_count",
        "mixed_route_count",
        "manual_route_count",
        "answer_chars",
        "image_count",
        "has_pic_marker",
        "has_numbered_structure",
        "keyword_coverage",
        "used_fallback",
        "service_like_question",
        "classifier_used_count",
        "classifier_dominant_label",
        "classifier_avg_confidence",
        "question",
        "answer",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


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
    parser.add_argument("--output-dir", default=str(Path("knowledge_base") / "evaluation"), help="输出目录")
    parser.add_argument("--submission-name", default="submission_public_generated.csv", help="提交 CSV 文件名")
    parser.add_argument("--disable-llm", action="store_true", help="关闭外部 LLM，使用检索驱动兜底回答")
    args = parser.parse_args()

    if args.disable_llm:
        settings.llm_provider = "disabled"
        settings.enable_cot_reasoning = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_questions(Path(args.questions))
    generator = ResponseGenerator()
    generator.initialize()

    analysis_rows: List[Dict[str, Any]] = []
    for row in rows:
        result = generator.generate(row["question"])
        analysis_rows.append(build_analysis_record(row, result))

    submission_path = output_dir / args.submission_name
    detail_path = output_dir / "public_answer_detail.csv"
    summary_json_path = output_dir / "public_answer_summary.json"
    summary_md_path = output_dir / "public_answer_summary.md"

    write_submission_csv(submission_path, analysis_rows)
    write_detail_csv(detail_path, analysis_rows)

    summary = build_summary(
        analysis_rows,
        settings_snapshot={
            "llm_provider": settings.llm_provider,
            "embedding_backend": settings.embedding_backend,
            "rag_top_k": settings.rag_top_k,
        },
    )
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(summary_md_path, summary, analysis_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已输出提交文件: {submission_path}")
    print(f"已输出明细文件: {detail_path}")
    print(f"已输出摘要 JSON: {summary_json_path}")
    print(f"已输出摘要 MD: {summary_md_path}")


if __name__ == "__main__":
    main()
