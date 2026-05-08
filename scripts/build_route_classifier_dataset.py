"""
构建 route classifier 的弱标注三分类数据集。

输出：
1. train/val/test JSONL
2. 人工审查集 JSONL
3. 数据集摘要 JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from scripts.build_dual_route_kb import (
    INTENT_SPECS,
    classify_intents,
    classify_route,
    load_questions,
)
from src.utils.text_utils import QueryProcessor


LABELS = ["service", "manual", "mixed"]


def dedupe_key(text: str) -> str:
    """
    计算去重键，用于合并语义相同的样本。

    去重键 = 归一化(小写)后的检索查询文本。
    相同语义的问题会归一化到相同的key，用于去重和数据集划分。

    Args:
        text: 原始问题文本

    Returns:
        去重键字符串
    """
    normalized = QueryProcessor.normalize_query_for_retrieval(text)["normalized_query"]
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def load_manual_sections(metadata_path: Path) -> List[Dict[str, Any]]:
    if not metadata_path.exists():
        return []
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = []
    for doc in payload.get("texts", []):
        metadata = doc.get("metadata", {})
        route = metadata.get("route")
        doc_type = metadata.get("type")
        if route not in {None, "manual"} and doc_type != "manual":
            continue
        if route == "service":
            continue
        manual_name = str(metadata.get("manual_name", "")).strip()
        section_title = str(metadata.get("section_title", "")).strip()
        content = str(doc.get("content", "")).strip()
        if manual_name and section_title and content:
            rows.append(
                {
                    "manual_name": manual_name,
                    "section_title": section_title,
                    "content": content,
                }
            )
    return rows


def build_service_samples(question_rows: Sequence[Dict[str, Any]], route_kb: Dict[str, Any], seed_docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in question_rows:
        if row["route"] == "service":
            samples.append(
                {
                    "text": row["question"],
                    "label": "service",
                    "source": "question_public",
                    "confidence": 0.96,
                }
            )

    for doc in route_kb.get("service_playbook_documents", []):
        title = doc.get("metadata", {}).get("title", "")
        intent = doc.get("metadata", {}).get("intent", "")
        if title:
            samples.append(
                {
                    "text": f"请问{title}怎么处理？",
                    "label": "service",
                    "source": f"playbook:{intent}",
                    "confidence": 0.93,
                }
            )
            samples.append(
                {
                    "text": f"{title}需要提供哪些信息？",
                    "label": "service",
                    "source": f"playbook:{intent}",
                    "confidence": 0.9,
                }
            )

    for doc in seed_docs:
        title = doc["title"]
        keywords = doc.get("keywords", [])
        applies_to = doc.get("applies_to", [])
        if title:
            samples.append(
                {
                    "text": f"{title}的售后规则是什么？",
                    "label": "service",
                    "source": f"seed:{doc['intent']}",
                    "confidence": 0.95,
                }
            )
        for keyword in keywords[:3]:
            samples.append(
                {
                    "text": f"关于{keyword}，客服一般怎么处理？",
                    "label": "service",
                    "source": f"seed:{doc['intent']}",
                    "confidence": 0.88,
                }
            )
        for scene in applies_to[:1]:
            samples.append(
                {
                    "text": scene,
                    "label": "service",
                    "source": f"seed_scene:{doc['intent']}",
                    "confidence": 0.82,
                }
            )
    return samples


def build_manual_samples(question_rows: Sequence[Dict[str, Any]], manual_sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in question_rows:
        if row["route"] == "manual":
            samples.append(
                {
                    "text": row["question"],
                    "label": "manual",
                    "source": "question_public",
                    "confidence": 0.95,
                }
            )

    templates = [
        "如何{section_title}？",
        "{manual_name}的{section_title}怎么操作？",
        "{section_title}需要注意什么？",
    ]
    for section in manual_sections:
        section_title = section["section_title"].replace("#", "").strip()
        manual_name = section["manual_name"].replace("手册", "")
        if not section_title or len(section_title) < 2:
            continue
        for template in templates:
            text = template.format(section_title=section_title, manual_name=manual_name)
            samples.append(
                {
                    "text": text,
                    "label": "manual",
                    "source": f"manual_section:{manual_name}",
                    "confidence": 0.84,
                }
            )
    return samples


def build_mixed_samples(question_rows: Sequence[Dict[str, Any]], manual_sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in question_rows:
        if row["route"] == "mixed":
            samples.append(
                {
                    "text": row["question"],
                    "label": "mixed",
                    "source": "question_public",
                    "confidence": 0.97,
                }
            )

    service_clues = [
        "安装收费怎么处理",
        "售后维修流程是什么",
        "能补发纸质说明书吗",
        "需要联系客服核实哪些信息",
    ]
    manual_clues = [
        "给我对应的说明书",
        "附上图示",
        "告诉我具体操作步骤",
        "返回相关手册内容",
    ]

    for section in manual_sections[:160]:
        manual_name = section["manual_name"].replace("手册", "")
        section_title = section["section_title"].replace("#", "").strip()
        if not manual_name or not section_title:
            continue
        for service_part in service_clues[:2]:
            for manual_part in manual_clues[:2]:
                samples.append(
                    {
                        "text": f"{manual_name}{service_part}？另外请把{section_title}的{manual_part}也发我。",
                        "label": "mixed",
                        "source": f"synthetic_mixed:{manual_name}",
                        "confidence": 0.9,
                    }
                )
    return samples


def load_seed_documents(seed_path: Path, qa_seed_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """加载客服政策种子文档，包括基础政策和扩充 QA"""
    all_docs = []
    
    # 1. 加载基础政策种子
    if seed_path.exists():
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
        all_docs.extend(payload.get("documents", []))
    
    # 2. 加载扩充 QA 种子（如果存在）
    if qa_seed_path and qa_seed_path.exists():
        try:
            qa_payload = json.loads(qa_seed_path.read_text(encoding="utf-8"))
            all_docs.extend(qa_payload.get("documents", []))
        except Exception:
            pass
    
    return all_docs


def read_route_kb(route_kb_path: Path) -> Dict[str, Any]:
    return json.loads(route_kb_path.read_text(encoding="utf-8"))


def filter_high_confidence(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        if sample["confidence"] < 0.82:
            continue
        key = dedupe_key(sample["text"])
        current = deduped.get(key)
        if current is None or sample["confidence"] > current["confidence"]:
            deduped[key] = sample
    return list(deduped.values())


def split_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    将样本划分为训练/验证/测试集。

    划分策略: 基于样本去重键的MD5哈希值取模100:
    - [0, 78) -> train (78%)
    - [78, 89) -> val (11%)
    - [89, 100) -> test (11%)

    确定性划分: 同一去重键总被划分到同一集合，保证训练/验证/测试集不重叠。

    Args:
        samples: 待划分的样本列表

    Returns:
        包含train/val/test三个子列表的字典
    """
    buckets = {"train": [], "val": [], "test": []}
    for sample in samples:
        key = dedupe_key(sample["text"])
        # MD5哈希前8字符转整数后模100，决定划分
        digest = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
        if digest < 78:
            split = "train"
        elif digest < 89:
            split = "val"
        else:
            split = "test"
        buckets[split].append({**sample, "split": split})
    return buckets


def build_manual_review_set(samples: Sequence[Dict[str, Any]], size: int = 150) -> List[Dict[str, Any]]:
    """
    构建需要人工审查的样本集。

    采样策略:
    1. 每类(service/manual/mixed)随机抽取最多40条
    2. 英文或混合语言样本额外抽取30条
    3. 按去重键去重
    4. 总计不超过150条

    用途: 弱标注数据的质量验证，通过人工审查评估标注准确率。

    Args:
        samples: 所有标注样本
        size: 审查集最大容量

    Returns:
        审查集样本列表
    """
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    english_or_mixed = []
    for sample in samples:
        by_label[sample["label"]].append(sample)
        language = QueryProcessor.detect_language(sample["text"])
        if language in {"en", "mixed"}:
            english_or_mixed.append(sample)

    review: List[Dict[str, Any]] = []
    random.seed(42)
    for label in LABELS:
        label_rows = by_label[label][:]
        random.shuffle(label_rows)
        review.extend(label_rows[:40])

    random.shuffle(english_or_mixed)
    review.extend(english_or_mixed[:30])

    unique: Dict[str, Dict[str, Any]] = {}
    for row in review:
        unique.setdefault(dedupe_key(row["text"]), row)
    return list(unique.values())[:size]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(split_buckets: Dict[str, List[Dict[str, Any]]], review_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "splits": {},
        "review_size": len(review_rows),
    }
    for split_name, rows in split_buckets.items():
        summary["splits"][split_name] = {
            "count": len(rows),
            "labels": dict(Counter(row["label"] for row in rows)),
            "sources": dict(Counter(row["source"].split(":")[0] for row in rows)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 route classifier 弱标注数据集")
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "question_public.csv"))
    parser.add_argument("--route-kb", default=str(settings.route_kb_path / settings.service_route_kb_file))
    parser.add_argument("--seed", default=str(settings.service_policy_seed_path / settings.service_policy_seed_file))
    parser.add_argument("--qa-seed", default=str(settings.service_policy_seed_path / "service_qa_seed.json"))
    parser.add_argument("--metadata", default=str(settings.index_path / settings.metadata_file))
    parser.add_argument("--output-dir", default=str(settings.route_classifier_data_path / "dataset"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    question_rows = load_questions(Path(args.questions))
    route_kb = read_route_kb(Path(args.route_kb))
    qa_seed_path = Path(args.qa_seed)
    seed_docs = load_seed_documents(Path(args.seed), qa_seed_path)
    manual_sections = load_manual_sections(Path(args.metadata))

    samples = []
    samples.extend(build_service_samples(question_rows, route_kb, seed_docs))
    samples.extend(build_manual_samples(question_rows, manual_sections))
    samples.extend(build_mixed_samples(question_rows, manual_sections))
    samples = filter_high_confidence(samples)

    split_buckets = split_samples(samples)
    review_rows = build_manual_review_set(samples)
    summary = build_summary(split_buckets, review_rows)

    for split_name, rows in split_buckets.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    write_jsonl(output_dir / "manual_review.jsonl", review_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"数据集已输出到: {output_dir}")
    print(f"QA种子加载: {qa_seed_path.exists()}")


if __name__ == "__main__":
    main()
