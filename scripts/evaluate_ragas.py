"""
RAGAS 评估脚本 - 使用 RAGAS 框架评估系统质量

RAGAS (RAG Assessment) 是当前 RAG 系统评估的事实标准框架，核心指标：
- Faithfulness：回答是否忠实于检索上下文（幻觉率代理）
- Answer Relevancy：回答是否真正回应了用户问题
- Context Precision：召回的片段中，有多少是真正有用的

用途：
1. 读取 question_public.csv 中的问题
2. 对每个问题执行完整的 DualRouteRetriever 检索 + ResponseGenerator 生成
3. 将检索到的上下文（contexts）作为 RAGAS 输入
4. 输出逐题 RAGAS 指标得分、汇总报告和低分样例

说明：
- 本脚本不需要 ground_truth 答案，适合公开题无标答场景
- 多问题会被 QuestionDecomposer 拆解后逐个评估
- 使用项目中已配置的 Infini LLM 作为 RAGAS 裁判模型
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.modules.dual_route_retriever import (
    get_dual_route_retriever,
    reset_dual_route_retriever,
)
from src.modules.response_generator import ResponseGenerator


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "knowledge_base" / "evaluation"
DEFAULT_QUESTION_FILE = PROJECT_ROOT / "question_public.csv"

# RAGAS 评估模式：决定每个指标使用哪些数据字段
RAGAS_MODE_NONE = "none"       # 不跑 RAGAS（仅收集数据）
RAGAS_MODE_MINIMAL = "minimal"  # 最小模式：只用 faithfulness + answer_relevancy
RAGAS_MODE_FULL = "full"       # 完整模式：+ context_precision
RAGAS_MODE_ALL = "all"         # 全部指标（需要 ground_truth）


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class QuestionRecord:
    """单条公开题记录。"""
    question_id: str
    raw_question: str
    normalized_question: str


@dataclass
class RagasRow:
    """RAGAS 评估数据集的一行。"""
    question: str
    answer: str
    contexts: List[str]
    # ground_truth 字段可选（无标答时留空）
    ground_truth: str = ""
    question_id: str = ""
    sub_question_index: int = 0
    parent_question: str = ""  # 来自哪个原始多问题
    route: str = ""
    confidence: float = 0.0
    used_fallback: bool = False
    image_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 RAGAS 框架评估系统质量")
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTION_FILE),
        help="公开题 CSV 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="评测输出目录",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评测前 N 题，0 表示全部",
    )
    parser.add_argument(
        "--ragas-mode",
        choices=["none", "minimal", "full", "all"],
        default="minimal",
        help=(
            "RAGAS 评估模式：\n"
            "  none    - 仅收集问答数据，不调用 RAGAS（快速检查）\n"
            "  minimal - faithfulness + answer_relevancy（无标答推荐）\n"
            "  full    - + context_precision（无标答完整版）\n"
            "  all     - 全部指标（需要 ground_truth，本项目公开题不可用）\n"
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="RAGAS 裁判模型，默认使用 INFINI_MODEL_ID",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="RAGAS embedding 模型，默认使用 EMBEDDING_MODEL",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="关闭 LLM，使用检索兜底回答",
    )
    parser.add_argument(
        "--sub-question",
        action="store_true",
        help="多问题拆解后逐子问题评估（默认对原问题整体评估）",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3000,
        help="每个 context 的最大字符数（防止超长截断）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="检索召回的上下文数量上限",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="RAGAS 评估时遇错立即抛出（默认 continue，部分失败记录为 nan）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 文本处理
# ---------------------------------------------------------------------------

def normalize_question(raw_question: str) -> str:
    """清洗 CSV 中的多行/多引号格式。"""
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
    """加载公开题。"""
    records: List[QuestionRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = normalize_question(row["question"])
            record = QuestionRecord(
                question_id=str(row["id"]).strip(),
                raw_question=row["question"],
                normalized_question=normalized,
            )
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def truncate_contexts(contexts: List[str], max_chars: int = 3000) -> List[str]:
    """截断每个 context 到最大字符数，防止超长输入。"""
    truncated = []
    for ctx in contexts:
        if len(ctx) > max_chars:
            truncated.append(ctx[:max_chars] + " [...截断]")
        else:
            truncated.append(ctx)
    return truncated


# ---------------------------------------------------------------------------
# 数据收集：调用系统检索 + 生成
# ---------------------------------------------------------------------------

def collect_ragas_data(
    questions: Sequence[QuestionRecord],
    top_k: int,
    max_context_chars: int,
    sub_question_mode: bool,
    disable_llm: bool,
) -> List[RagasRow]:
    """
    对每个问题执行检索和生成，收集 RAGAS 所需的数据。

    Args:
        sub_question_mode: True 时，多问题拆解后逐个子问题收集一行
                          False 时，原问题整体收集一行
    """
    reset_dual_route_retriever()
    generator = ResponseGenerator()

    if disable_llm:
        settings.llm_provider = "disabled"
        settings.enable_cot_reasoning = False

    generator.initialize()

    rows: List[RagasRow] = []
    error_count = 0

    for idx, question in enumerate(questions):
        try:
            result = generator.generate(question.normalized_question)
            answer = result.get("response", "")
            confidence = float(result.get("confidence", 0.0))
            used_fallback = bool(result.get("used_fallback", False))
            sources: List[Dict[str, Any]] = result.get("sources", [])
            route_records = result.get("routes", [])

            # 汇总路由信息
            if route_records:
                dominant_routes = Counter(
                    r.get("route", "manual") for r in route_records if r
                )
                dominant_route = dominant_routes.most_common(1)[0][0] if dominant_routes else "manual"
            else:
                dominant_route = "unknown"

            # 收集图片 ID
            image_ids = result.get("images", [])

            if sub_question_mode and len(route_records) > 1:
                # 多问题场景：每个子问题单独收集一行
                for sq_idx, route_record in enumerate(route_records):
                    sq_text = route_record.get("question", question.normalized_question)
                    sq_sources = route_record.get("results", [])

                    # 取该子问题的检索结果作为 contexts
                    sq_contexts = truncate_contexts(
                        [src.get("content", "") for src in sq_sources[:top_k]],
                        max_context_chars,
                    )

                    rows.append(RagasRow(
                        question=sq_text,
                        answer=answer,  # 整体回答（包含该子问题的答案）
                        contexts=sq_contexts,
                        question_id=f"{question.question_id}_sq{sq_idx}",
                        sub_question_index=sq_idx,
                        parent_question=question.normalized_question,
                        route=route_record.get("route", dominant_route),
                        confidence=confidence,
                        used_fallback=used_fallback,
                        image_ids=image_ids,
                    ))
            else:
                # 整体评估：取前 top_k 个检索结果作为 contexts
                all_contexts = truncate_contexts(
                    [src.get("content", "") for src in sources[:top_k]],
                    max_context_chars,
                )

                rows.append(RagasRow(
                    question=question.normalized_question,
                    answer=answer,
                    contexts=all_contexts,
                    question_id=question.question_id,
                    route=dominant_route,
                    confidence=confidence,
                    used_fallback=used_fallback,
                    image_ids=image_ids,
                ))

        except Exception as e:
            error_count += 1
            print(f"[警告] 题目 {question.question_id} 收集失败: {type(e).__name__}: {e}")

        if (idx + 1) % 10 == 0:
            print(f"[进度] 已收集 {idx + 1}/{len(questions)} 题")

    if error_count > 0:
        print(f"[汇总] 共 {error_count} 题收集失败")

    return rows


# ---------------------------------------------------------------------------
# RAGAS 评估
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    rows: List[RagasRow],
    ragas_mode: str,
    judge_model: Optional[str],
    embedding_model: Optional[str],
    fail_fast: bool,
) -> Dict[str, Any]:
    """
    调用 RAGAS 对收集的数据进行评估。

    Args:
        rows: RagasRow 数据列表
        ragas_mode: "minimal" | "full" | "all" | "none"
        judge_model: 裁判 LLM 模型名
        embedding_model: embedding 模型名
        fail_fast: True 则遇错抛出，否则记录 nan

    Returns:
        {"scores": [{row_id: {...metrics...}}, ...], "summary": {...}}
    """
    from datasets import Dataset

    # 准备 HuggingFace Dataset
    hf_data: Dict[str, List[Any]] = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "question_id": [],
        "route": [],
        "confidence": [],
        "used_fallback": [],
    }

    for row in rows:
        hf_data["user_input"].append(row.question)
        hf_data["response"].append(row.answer)
        hf_data["retrieved_contexts"].append(row.contexts)
        hf_data["question_id"].append(row.question_id)
        hf_data["route"].append(row.route)
        hf_data["confidence"].append(row.confidence)
        hf_data["used_fallback"].append(row.used_fallback)

    dataset = Dataset.from_dict(hf_data)
    print(f"[RAGAS] 数据集准备完毕: {len(dataset)} 条")

    # 配置 LLM 裁判（使用 Infini OpenAI 兼容接口）
    judge_llm = _build_ragas_llm(judge_model)
    judge_embeddings = _build_ragas_embeddings(embedding_model)

    # 按模式决定使用哪些指标
    metrics = _build_metrics(ragas_mode, judge_llm, model_name=judge_model)

    if not metrics:
        print("[RAGAS] 模式为 none，跳过 RAGAS 评估")
        return _build_empty_scores(rows)

    # column_map 格式：{RAGAS期望的标准列名: 我们数据集的实际列名}
    # remap_column_names 会取 inverse，即从我们列名 rename 到 RAGAS 标准列名
    # RAGAS 期望: user_input, answer, contexts
    # 我们数据集: user_input, response, retrieved_contexts
    column_map = {
        "answer": "response",
        "contexts": "retrieved_contexts",
    }

    # 对于需要 ground_truth 的指标（如 context_recall），RAGAS 会自动跳过无标答的记录
    # 设置 raise_exceptions=False 让部分失败返回 nan 而不中断整批
    from ragas import evaluate

    print(f"[RAGAS] 开始评估，模式={ragas_mode}，指标={len(metrics)} 个")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        column_map=column_map,
        raise_exceptions=fail_fast,
        show_progress=True,
    )

    # 提取逐题得分
    # result._repr_dict 包含每个指标的聚合均值，result[key] 返回每行得分列表
    score_keys = list(result._repr_dict.keys())
    scores = []
    for i in range(len(rows)):
        row_scores: Dict[str, float] = {}
        for key in score_keys:
            vals = result[key]  # List[float], one per row
            val = vals[i] if i < len(vals) else float("nan")
            row_scores[key] = float(val) if val is not None and not _is_nan(val) else float("nan")
        scores.append({
            "question_id": rows[i].question_id,
            "question": rows[i].question,
            "route": rows[i].route,
            "confidence": rows[i].confidence,
            "used_fallback": rows[i].used_fallback,
            "context_count": len(rows[i].contexts),
            **row_scores,
        })

    # 构建汇总
    summary = _build_summary(score_keys, scores, rows)

    return {"scores": scores, "summary": summary, "metrics": score_keys}


def _build_ragas_llm(model_name: Optional[str]) -> Any:
    """
    构建 RAGAS LLM 裁判（使用 Infini OpenAI 兼容接口）。

    注意：max_tokens 需足够大（默认 1024），Faithfulness 指标会生成大量 statement，
    生成不足会导致 JSON 截断、InstructorRetryException。
    """
    from ragas.llms import llm_factory
    from openai import OpenAI

    model = model_name or settings.llm_model
    base_url = getattr(settings, "llm_base_url", None) or "https://cloud.infini-ai.com/maas/v1"
    api_key = getattr(settings, "llm_api_key", None) or "dummy"

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
    return llm_factory(model, client=client, max_tokens=4096)


def _build_ragas_embeddings(model_name: Optional[str]) -> Any:
    """占位函数，RAGAS 0.4 faithfulness 指标不依赖 embeddings，保留接口兼容。"""
    return None


def _build_metrics(mode: str, judge_llm: Any, model_name: Optional[str] = None) -> List[Any]:
    """
    按模式构建 RAGAS 指标列表。

    RAGAS 0.4 API 说明：
    - 旧版指标（_faithfulness._faithfulness, _answer_relevance.AnswerRelevancy）是 Metric 实例，
      直接传给 evaluate() 即可。
    - 新版 collections 指标（Faithfulness, AnswerRelevancy）是 ABC 子类，
      evaluate() 拒绝接收，故不采用。

    使用的指标：
    - faithfulness: 检查回答是否忠实于检索上下文（幻觉率代理）
    - answer_relevancy: 检查回答是否与问题相关

    关于 ContextPrecision 和 ContextRecall：
      这两个指标需要 reference（标准答案）字段，公开题无标答，不能使用。
    """
    if mode == "none":
        return []

    try:
        from ragas.metrics._faithfulness import faithfulness
        from ragas.metrics.base import Metric
    except ImportError:
        print("[错误] RAGAS 未安装，请运行: pip install ragas datasets")
        return []

    # answer_relevancy 需要 embed_query 接口，但 RAGAS 0.4 的 embedding_factory
    # 返回的是不含 embed_query 的 RAGAS 原生 OpenAIEmbeddings，无法满足。
    # 暂时只使用 faithfulness，等 RAGAS collections API 稳定后再接入 answer_relevancy。
    return [faithfulness]


def _is_nan(val: Any) -> bool:
    """判断是否为 nan（兼容 float 和 numpy nan）。"""
    try:
        import numpy as np
        return bool(np.isnan(val))
    except (TypeError, ValueError):
        return False


def _build_empty_scores(rows: List[RagasRow]) -> Dict[str, Any]:
    """ragas_mode=none 时返回空得分。"""
    scores = []
    for row in rows:
        scores.append({
            "question_id": row.question_id,
            "question": row.question,
            "route": row.route,
            "confidence": row.confidence,
            "used_fallback": row.used_fallback,
            "context_count": len(row.contexts),
        })
    return {"scores": scores, "summary": {}, "metrics": []}


def _build_summary(metric_keys: List[str], scores: List[Dict], rows: List[RagasRow]) -> Dict[str, Any]:
    """构建 RAGAS 指标汇总报告。"""
    summary: Dict[str, Any] = {}

    for key in metric_keys:
        vals = [s[key] for s in scores if key in s and not _is_nan(s[key])]
        if vals:
            summary[key] = {
                "mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "count": len(vals),
            }

    # 按路由分组
    route_groups: Dict[str, List[Dict]] = {}
    for s, row in zip(scores, rows):
        route = row.route or "unknown"
        if route not in route_groups:
            route_groups[route] = []
        route_groups[route].append(s)

    summary["by_route"] = {}
    for route, group_scores in route_groups.items():
        route_summary: Dict[str, Any] = {}
        for key in metric_keys:
            vals = [g[key] for g in group_scores if key in g and not _is_nan(g[key])]
            if vals:
                route_summary[key] = round(statistics.mean(vals), 4)
        route_summary["count"] = len(group_scores)
        summary["by_route"][route] = route_summary

    # 低分样例（每指标取最低3条）
    summary["low_score_examples"] = {}
    for key in metric_keys:
        valid = [s for s in scores if key in s and not _is_nan(s[key])]
        if valid:
            sorted_by_key = sorted(valid, key=lambda x: x[key])
            summary["low_score_examples"][key] = [
                {
                    "question_id": s["question_id"],
                    "question": s["question"],
                    "score": s[key],
                    "route": s["route"],
                }
                for s in sorted_by_key[:3]
            ]

    return summary


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, summary: Dict[str, Any], scores: List[Dict], metrics: List[str]) -> None:
    """生成 RAGAS 评估 Markdown 报告。"""
    lines = [
        "# RAGAS 评估报告",
        "",
        f"- 评估模式: `{summary.get('ragas_mode', 'N/A')}`",
        f"- 总题数: `{len(scores)}`",
        f"- 指标: `{', '.join(metrics) or '无（RAGAS mode=none）'}`",
        "",
        "## 指标汇总",
        "",
    ]

    if metrics:
        lines.append("| 指标 | 均值 | 中位数 | 最小 | 最大 | 标准差 | 有效数 |")
        lines.append("|------|------|--------|------|------|--------|--------|")
        for key in metrics:
            s = summary.get(key, {})
            if s:
                lines.append(
                    f"| {key} | {s.get('mean', 'N/A')} | {s.get('median', 'N/A')} | "
                    f"{s.get('min', 'N/A')} | {s.get('max', 'N/A')} | "
                    f"{s.get('stdev', 'N/A')} | {s.get('count', 'N/A')} |"
                )
    else:
        lines.append("*（RAGAS mode=none，仅收集数据，未运行评估）*")

    # 按路由分组
    by_route = summary.get("by_route", {})
    if by_route and metrics:
        lines.extend(["", "## 按路由分组", ""])
        lines.append("| 路由 | 数量 | " + " | ".join(metrics) + " |")
        lines.append("|------|------|" + "|".join(["------" for _ in metrics]) + "|")
        for route, rs in by_route.items():
            vals = [str(rs.get(k, "N/A")) for k in metrics]
            lines.append(f"| {route} | {rs.get('count', 0)} | " + " | ".join(vals) + " |")

    # 低分样例
    low_examples = summary.get("low_score_examples", {})
    if low_examples and metrics:
        lines.extend(["", "## 低分样例（每指标最低3条）", ""])
        for key in metrics:
            examples = low_examples.get(key, [])
            if examples:
                lines.append(f"### {key}")
                for ex in examples:
                    lines.append(
                        f"- id={ex['question_id']} route={ex['route']} "
                        f"score={ex['score']} question={ex['question'][:80]}"
                    )
                lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    question_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Step 1: 加载问题
    questions = load_questions(question_path, limit=args.limit)
    print(f"[Step 1] 加载了 {len(questions)} 个问题")

    # Step 2: 收集问答数据
    print(f"[Step 2] 开始收集检索和生成数据 (sub_question={args.sub_question})")
    ragas_rows = collect_ragas_data(
        questions=questions,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        sub_question_mode=args.sub_question,
        disable_llm=args.disable_llm,
    )
    print(f"[Step 2] 数据收集完成: {len(ragas_rows)} 条")

    # Step 3: RAGAS 评估
    print(f"[Step 3] 开始 RAGAS 评估 (mode={args.ragas_mode})")
    eval_result = run_ragas_evaluation(
        rows=ragas_rows,
        ragas_mode=args.ragas_mode,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        fail_fast=args.fail_fast,
    )

    scores = eval_result.get("scores", [])
    summary = eval_result.get("summary", {})
    metrics = eval_result.get("metrics", [])

    summary["ragas_mode"] = args.ragas_mode
    summary["total_rows"] = len(scores)
    summary["total_questions"] = len(questions)

    # Step 4: 输出文件
    scores_path = output_dir / "ragas_scores.json"
    summary_path = output_dir / "ragas_summary.json"
    md_path = output_dir / "ragas_report.md"

    write_json(scores_path, {"summary": summary, "scores": scores})
    write_json(summary_path, summary)
    write_markdown(md_path, summary, scores, metrics)

    # 打印汇总
    print("\n" + "=" * 60)
    print("RAGAS 评估完成")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"总行数: {len(scores)}")

    if metrics:
        print("\n指标均值:")
        for key in metrics:
            s = summary.get(key, {})
            if s:
                print(f"  {key}: {s.get('mean', 'N/A')}")
    else:
        print("\n（RAGAS mode=none，仅收集数据，未运行评估）")

    print(f"\n文件已保存:")
    print(f"  {scores_path}")
    print(f"  {summary_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
