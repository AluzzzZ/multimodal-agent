"""
数据集审计工具 - 辅助人工审查数据集标注质量

功能：
1. 识别可能存在标注问题的样本
2. 按标签和来源统计分布
3. 输出待审查的样本列表
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# 人工审查规则：识别可能存在问题的样本
QUESTIONABLE_PATTERNS = [
    # Seed 生成的问题：格式固定的模板问题
    (r"^关于.+，客服一般怎么处理\?$", "SERVICE", "seed模板问题，可能偏机械"),
    (r"^请问.+怎么处理\?$", "SERVICE", "playbook模板问题"),
    (r"^.+需要提供哪些信息\?$", "SERVICE", "playbook模板问题"),
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_sample_label(row: Dict[str, Any]) -> Dict[str, Any]:
    """检查样本标注是否可能存在问题"""
    text = row.get("text", "")
    label = row.get("label", "")
    source = row.get("source", "")

    issues = []

    # 1. 检查固定模板问题
    for pattern, expected_issue_type, reason in QUESTIONABLE_PATTERNS:
        if re.match(pattern, text.strip()):
            issues.append(f"[{expected_issue_type}] {reason}")

    # 2. 检查 seed 模板来源
    if source.startswith("seed:") and len(text) < 20:
        issues.append("[LOW_QUALITY] seed模板过短")

    return {
        **row,
        "issues": issues,
        "has_issues": len(issues) > 0
    }


def analyze_dataset(dataset_dir: Path) -> None:
    """分析数据集质量"""
    print("=" * 80)
    print("数据集审计报告")
    print("=" * 80)

    # 加载数据
    splits = ["train", "val", "test", "manual_review"]
    all_samples = []

    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            samples = load_jsonl(path)
            print(f"\n{split.upper()}: {len(samples)} 条样本")

            # 标签分布
            label_counts = Counter(s.get("label") for s in samples)
            print(f"  标签分布: {dict(label_counts)}")

            # 来源分布
            source_counts = Counter(s.get("source", "").split(":")[0] for s in samples)
            print(f"  来源分布: {dict(source_counts)}")

            all_samples.extend(samples)

    # 检查问题样本
    print("\n" + "=" * 80)
    print("可疑样本 (前20条):")
    print("=" * 80)

    checked = [check_sample_label(s) for s in all_samples]
    questionable = [s for s in checked if s.get("has_issues", False)]

    # 按来源排序
    source_priority = {"question_public": 0, "manual_section": 1, "seed": 2, "seed_scene": 3, "playbook": 4, "synthetic_mixed": 5}
    questionable.sort(key=lambda x: (source_priority.get(x.get("source", "").split(":")[0], 99), -len(x.get("issues", []))))

    for i, s in enumerate(questionable[:20]):
        print(f"\n{i+1}. [{s['label']}] [{s.get('source', '?')}]")
        print(f"   {s['text'][:70]}")
        for issue in s.get("issues", []):
            print(f"   warning: {issue}")

    print(f"\n\n总计可疑样本: {len(questionable)} / {len(all_samples)} ({len(questionable)/len(all_samples)*100:.1f}%)")

    # 输出详细报告
    report_path = dataset_dir / "audit_report.json"
    report = {
        "total_samples": len(all_samples),
        "questionable_count": len(questionable),
        "questionable_percentage": round(len(questionable)/len(all_samples)*100, 2),
        "samples": questionable
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细报告已保存到: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="审计数据集标注质量")
    parser.add_argument(
        "--dataset-dir",
        default=str(PROJECT_ROOT / "knowledge_base" / "route_classifier" / "dataset"),
        help="数据集目录"
    )
    args = parser.parse_args()

    analyze_dataset(Path(args.dataset_dir))


if __name__ == "__main__":
    main()
