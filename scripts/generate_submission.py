"""
按比赛提交格式批量生成 submission.csv。

用途：
1. 读取 question_public.csv 中的题目；
2. 调用当前项目的回答主链路批量生成答案；
3. 按 submission_example.csv 的格式输出最终提交文件。

说明：
- 默认输出仓库根目录下的 submission.csv
- 若单题生成失败，会写入一条保底答复，保证提交文件完整
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from config import settings
from scripts.build_dual_route_kb import normalize_question
from src.modules.response_generator import ResponseGenerator


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


def load_submission_header(example_path: Path) -> List[str]:
    with example_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    return header or ["id", "ret"]


def fallback_answer(question: str, exc: Exception) -> str:
    """单题失败时的保底答复，保证 CSV 可提交。"""
    text = (question or "").strip()
    if "发票" in text:
        return "您好，商品通常支持开具发票，具体发票类型、开票时间和接收方式需以订单信息为准，建议您提供订单号或开票需求，我们帮您进一步核实。"
    if any(token in text for token in ["退款", "退货", "换货", "售后", "维修", "投诉", "物流", "运费"]):
        return "您好，相关售后问题需要结合订单和商品情况进一步核实，建议您提供订单号、问题详情及相关图片，我们会尽快为您处理。"
    if any(token in text for token in ["说明书", "手册", "按钮", "指示灯", "安装", "使用", "设置", "模式"]):
        return "您好，关于商品使用问题，建议您提供具体型号、部件名称或相关图片，这样我可以结合对应说明书内容为您进一步说明。"
    return "您好，您的问题已收到。为便于进一步核实，请您补充商品型号、订单信息或相关图片说明，我们会尽快为您解答。"


def write_submission(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row["id"], row["ret"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="生成比赛提交文件 submission.csv")
    parser.add_argument("--questions", default="question_public.csv", help="问题 CSV 路径")
    parser.add_argument("--example", default="submission_example.csv", help="提交样例 CSV 路径")
    parser.add_argument("--output", default="submission.csv", help="输出 submission.csv 路径")
    parser.add_argument("--disable-llm", action="store_true", help="关闭外部 LLM，仅使用检索链路")
    args = parser.parse_args()

    if args.disable_llm:
        settings.llm_provider = "disabled"
        settings.enable_cot_reasoning = False

    questions = load_questions(Path(args.questions))
    header = load_submission_header(Path(args.example))

    generator = ResponseGenerator()
    generator.initialize()

    submission_rows: List[Dict[str, str]] = []
    total = len(questions)

    for idx, row in enumerate(questions, start=1):
        try:
            result = generator.generate(row["question"])
            answer = str(result.get("response", "")).strip()
            if not answer:
                answer = fallback_answer(row["question"], RuntimeError("empty_response"))
        except Exception as exc:
            print(f"[警告] 第 {idx}/{total} 题生成失败，id={row['id']}，已写入保底答复: {type(exc).__name__}: {exc}")
            answer = fallback_answer(row["question"], exc)

        submission_rows.append(
            {
                "id": row["id"],
                "ret": answer,
            }
        )

        if idx % 20 == 0 or idx == total:
            print(f"[进度] 已完成 {idx}/{total}")

    output_path = Path(args.output)
    write_submission(output_path, header, submission_rows)
    print(f"已生成提交文件: {output_path.resolve()}")


if __name__ == "__main__":
    main()
