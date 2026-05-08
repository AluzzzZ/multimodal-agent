"""
扫描 route classifier 的 high/low threshold 组合，评估更优路由阈值。

评估分两层：
1. 带标签测试集：accuracy / macro_f1 / service_f1 / mixed_f1
2. question_public 公开题代理指标：manual_on_service_like_count / classifier_used

默认使用当前最优候选模型目录作为输入，方便比较不同阈值下的路由仲裁效果。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, update_settings
from src.modules.dual_route_retriever import DualRouteRetriever, reset_dual_route_retriever
from src.modules.route_classifier import ROUTE_LABELS, reset_route_classifier


LABEL_TO_ID = {label: idx for idx, label in enumerate(ROUTE_LABELS)}
DEFAULT_QUESTION_FILE = PROJECT_ROOT / "question_public.csv"
DEFAULT_DATASET_FILE = PROJECT_ROOT / "knowledge_base" / "route_classifier" / "dataset" / "test.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "knowledge_base" / "evaluation" / "threshold_scan"

SERVICE_LIKE_KEYWORDS = [
    "退款", "退货", "换货", "发票", "物流", "运费", "售后", "维修", "投诉", "客服",
    "配送", "签收", "赔偿", "安装收费", "价保", "补发", "发货", "揽收",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 route classifier 阈值组合")
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "knowledge_base" / "route_classifier" / "model_ratio_03"))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_FILE))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTION_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--high-values", default="0.76,0.80,0.82,0.84,0.88")
    parser.add_argument("--low-values", default="0.30,0.36,0.42,0.46,0.50")
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    values = []
    for item in (raw or "").split(","):
        item = item.strip()
        if item:
            values.append(round(float(item), 4))
    return values


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_public_question(raw_question: str) -> str:
    text = (raw_question or "").strip()
    if not text:
        return ""
    text = text.replace('""', '"')
    text = text.replace("\\n", "\n")
    text = text.strip('"')
    parts = []
    for line in text.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            parts.append(line)
    return "\n".join(parts)


def load_public_questions(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = normalize_public_question(row.get("question", ""))
            if question:
                rows.append({"id": str(row["id"]).strip(), "question": question})
    return rows


def compute_f1_metrics(preds: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    pred_ids = [LABEL_TO_ID[item] for item in preds]
    label_ids = [LABEL_TO_ID[item] for item in labels]
    accuracy = sum(int(p == y) for p, y in zip(pred_ids, label_ids)) / max(1, len(label_ids))

    f1_by_label: Dict[str, float] = {}
    for label_name, label_id in LABEL_TO_ID.items():
        tp = sum(int(p == label_id and y == label_id) for p, y in zip(pred_ids, label_ids))
        fp = sum(int(p == label_id and y != label_id) for p, y in zip(pred_ids, label_ids))
        fn = sum(int(p != label_id and y == label_id) for p, y in zip(pred_ids, label_ids))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_by_label[label_name] = round(f1, 4)

    macro_f1 = round(sum(f1_by_label.values()) / len(ROUTE_LABELS), 4)
    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": macro_f1,
        "service_f1": f1_by_label.get("service", 0.0),
        "manual_f1": f1_by_label.get("manual", 0.0),
        "mixed_f1": f1_by_label.get("mixed", 0.0),
    }


def build_retriever(model_dir: Path, high_threshold: float, low_threshold: float) -> DualRouteRetriever:
    update_settings(
        route_classifier_model_dir=model_dir,
        route_classifier_high_threshold=high_threshold,
        route_classifier_low_threshold=low_threshold,
    )
    reset_route_classifier()
    reset_dual_route_retriever()
    retriever = DualRouteRetriever()
    retriever.initialize()
    return retriever


def evaluate_thresholds(
    model_dir: Path,
    high_threshold: float,
    low_threshold: float,
    dataset_rows: Sequence[Dict[str, Any]],
    public_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    retriever = build_retriever(model_dir, high_threshold, low_threshold)

    preds: List[str] = []
    labels: List[str] = []
    classifier_used = 0
    fallback_counts = Counter()
    for row in dataset_rows:
        text = str(row["text"]).strip()
        label = str(row["label"]).strip()
        if not text or label not in LABEL_TO_ID:
            continue
        info = retriever.route_query(text)
        preds.append(info["route"])
        labels.append(label)
        if info.get("classifier_used"):
            classifier_used += 1
        fallback_counts[info.get("classifier_fallback_reason", "")] += 1

    metrics = compute_f1_metrics(preds, labels)

    public_route_counter = Counter()
    public_classifier_used = 0
    manual_on_service_like = 0
    public_fallback_counts = Counter()
    confidences: List[float] = []

    for row in public_rows:
        info = retriever.route_query(row["question"])
        route = info["route"]
        public_route_counter[route] += 1
        confidences.append(float(info.get("classifier_confidence", 0.0)))
        if info.get("classifier_used"):
            public_classifier_used += 1
        public_fallback_counts[info.get("classifier_fallback_reason", "")] += 1
        if route == "manual" and any(keyword in row["question"] for keyword in SERVICE_LIKE_KEYWORDS):
            manual_on_service_like += 1

    return {
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "labeled_metrics": metrics,
        "labeled_classifier_used": classifier_used,
        "labeled_fallback_reasons": dict(fallback_counts),
        "public_route_distribution": dict(public_route_counter),
        "public_classifier_used": public_classifier_used,
        "public_manual_on_service_like_count": manual_on_service_like,
        "public_classifier_avg_confidence": round(statistics.mean(confidences), 4) if confidences else 0.0,
        "public_fallback_reasons": dict(public_fallback_counts),
    }


def rank_key(item: Dict[str, Any]) -> Tuple[float, float, float, int, float]:
    metrics = item["labeled_metrics"]
    return (
        metrics["macro_f1"],
        metrics["service_f1"],
        metrics["accuracy"],
        -item["public_manual_on_service_like_count"],
        item["public_classifier_avg_confidence"],
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    dataset_rows = load_jsonl(Path(args.dataset))
    public_rows = load_public_questions(Path(args.questions))
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    high_values = parse_float_list(args.high_values)
    low_values = parse_float_list(args.low_values)

    results: List[Dict[str, Any]] = []
    for high_threshold in high_values:
        for low_threshold in low_values:
            if low_threshold >= high_threshold:
                continue
            result = evaluate_thresholds(
                model_dir=model_dir,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
                dataset_rows=dataset_rows,
                public_rows=public_rows,
            )
            results.append(result)
            metrics = result["labeled_metrics"]
            print(
                f"high={high_threshold:.2f}, low={low_threshold:.2f} | "
                f"macro_f1={metrics['macro_f1']:.4f}, service_f1={metrics['service_f1']:.4f}, "
                f"public_manual_on_service_like={result['public_manual_on_service_like_count']}"
            )

    ranked = sorted(results, key=rank_key, reverse=True)
    summary = {
        "model_dir": str(model_dir),
        "top_result": ranked[0] if ranked else {},
        "top_5": ranked[:5],
        "result_count": len(ranked),
    }

    (output_dir / "threshold_scan_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "threshold_scan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n最佳组合:")
    print(json.dumps(summary["top_result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
