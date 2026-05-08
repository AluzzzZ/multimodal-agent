"""
端到端压测脚本。

目标：
1. 使用真实 FastAPI /chat 端点做文本和多模态冒烟压测；
2. 输出顺序压测与并发压测的耗时统计；
3. 默认关闭外部 LLM，使用检索驱动回答，便于本地稳定复现。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from config import settings
from src.api import app, api_config
from scripts.build_dual_route_kb import classify_intents, normalize_question


PNG_1X1_BASE64 = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jwN8AAAAASUVORK5CYII="
)


def load_sample_questions(csv_path: Path, limit_per_route: int = 8) -> Dict[str, List[str]]:
    grouped = {"service": [], "manual": []}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question = normalize_question(row["question"])
            if not question:
                continue
            route = "service" if classify_intents(question) else "manual"
            if len(grouped[route]) < limit_per_route:
                grouped[route].append(question)
            if all(len(values) >= limit_per_route for values in grouped.values()):
                break
    return grouped


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    index = int(round((len(values) - 1) * ratio))
    return values[index]


def summarize_timings(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [item["latency_seconds"] for item in records]
    success_count = sum(1 for item in records if item["status_code"] == 200 and item["code"] == 0)
    return {
        "count": len(records),
        "success_count": success_count,
        "success_rate": round(success_count / len(records), 4) if records else 0.0,
        "avg_seconds": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p50_seconds": round(percentile(latencies, 0.50), 4) if latencies else 0.0,
        "p95_seconds": round(percentile(latencies, 0.95), 4) if latencies else 0.0,
        "max_seconds": round(max(latencies), 4) if latencies else 0.0,
    }


def issue_request(client: TestClient, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    started = time.perf_counter()
    response = client.post("/chat", json=payload, headers=headers)
    elapsed = time.perf_counter() - started
    body = response.json()
    answer = body.get("data", {}).get("answer", "") if isinstance(body, dict) else ""
    return {
        "question": payload["question"],
        "latency_seconds": elapsed,
        "status_code": response.status_code,
        "code": body.get("code") if isinstance(body, dict) else None,
        "answer_preview": answer[:160],
    }


def run_sequential_benchmark(client: TestClient, payloads: List[Dict[str, Any]], headers: Dict[str, str]) -> List[Dict[str, Any]]:
    return [issue_request(client, payload, headers) for payload in payloads]


def run_concurrent_benchmark(
    client: TestClient,
    payloads: List[Dict[str, Any]],
    headers: Dict[str, str],
    workers: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(issue_request, client, payload, headers) for payload in payloads]
        for future in as_completed(futures):
            records.append(future.result())
    return records


def build_payloads(samples: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
    text_payloads = [{"question": question} for question in (samples["service"][:6] + samples["manual"][:6])]
    multimodal_payloads = [
        {
            "question": question,
            "images": [PNG_1X1_BASE64],
        }
        for question in (samples["service"][:4] + samples["manual"][:4])
    ]
    return {
        "text": text_payloads,
        "multimodal": multimodal_payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对 /chat 进行端到端压测")
    parser.add_argument("--questions", default=str(Path("question_public.csv")), help="公开题 CSV 路径")
    parser.add_argument("--output-dir", default=str(Path("knowledge_base") / "evaluation"), help="输出目录")
    parser.add_argument("--workers", type=int, default=4, help="并发压测线程数")
    parser.add_argument("--disable-llm", action="store_true", help="压测时关闭外部 LLM，使用检索兜底回答")
    args = parser.parse_args()

    if args.disable_llm:
        settings.llm_provider = "disabled"
        settings.enable_cot_reasoning = False
        settings.hallucination_detection_enabled = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_sample_questions(Path(args.questions))
    payload_groups = build_payloads(samples)
    headers = {"Authorization": f"Bearer {api_config.api_token}"}

    with TestClient(app) as client:
        sequential_text = run_sequential_benchmark(client, payload_groups["text"], headers)
        sequential_mm = run_sequential_benchmark(client, payload_groups["multimodal"], headers)
        concurrent_text = run_concurrent_benchmark(client, payload_groups["text"], headers, workers=args.workers)
        concurrent_mm = run_concurrent_benchmark(client, payload_groups["multimodal"], headers, workers=args.workers)

    summary = {
        "settings": {
            "llm_provider": settings.llm_provider,
            "workers": args.workers,
            "text_timeout_target_seconds": 20,
            "multimodal_timeout_target_seconds": 30,
        },
        "text": {
            "sequential": summarize_timings(sequential_text),
            "concurrent": summarize_timings(concurrent_text),
        },
        "multimodal": {
            "sequential": summarize_timings(sequential_mm),
            "concurrent": summarize_timings(concurrent_mm),
        },
        "samples": {
            "text": sequential_text[:3],
            "multimodal": sequential_mm[:3],
        },
    }

    json_path = output_dir / "chat_benchmark_summary.json"
    md_path = output_dir / "chat_benchmark_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# /chat 端到端压测摘要",
        "",
        f"- LLM provider: `{summary['settings']['llm_provider']}`",
        f"- workers: `{summary['settings']['workers']}`",
        f"- 文本超时目标: `{summary['settings']['text_timeout_target_seconds']}s`",
        f"- 多模态超时目标: `{summary['settings']['multimodal_timeout_target_seconds']}s`",
        "",
        "## 文本请求",
        f"- 顺序平均: `{summary['text']['sequential']['avg_seconds']}s`",
        f"- 顺序 P95: `{summary['text']['sequential']['p95_seconds']}s`",
        f"- 并发平均: `{summary['text']['concurrent']['avg_seconds']}s`",
        f"- 并发 P95: `{summary['text']['concurrent']['p95_seconds']}s`",
        "",
        "## 多模态请求",
        f"- 顺序平均: `{summary['multimodal']['sequential']['avg_seconds']}s`",
        f"- 顺序 P95: `{summary['multimodal']['sequential']['p95_seconds']}s`",
        f"- 并发平均: `{summary['multimodal']['concurrent']['avg_seconds']}s`",
        f"- 并发 P95: `{summary['multimodal']['concurrent']['p95_seconds']}s`",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已输出: {json_path}")
    print(f"已输出: {md_path}")


if __name__ == "__main__":
    main()
