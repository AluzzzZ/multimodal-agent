"""
构建双路检索中的 service 路知识库。

目标：
1. 基于 question_public.csv 提炼客服问题分布与示例问题；
2. 合并正式客服政策 / FAQ 种子文档，形成可直接支撑回答的知识源；
3. 输出供双路检索器使用的统一 JSON 载荷。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


INTENT_SPECS: Dict[str, Dict[str, Any]] = {
    "return_exchange": {
        "title": "退货与换货",
        "keywords": ["退货", "换货", "7天无理由", "尺寸", "颜色", "包装盒", "拆封", "试用", "差价"],
        "checklist": [
            "确认商品状态：是否拆封、是否使用、是否存在污损",
            "确认时效：是否仍在无理由退换期内",
            "确认诉求：退款、换货、补差价还是补寄配件",
            "确认运费承担方与是否需要保留原包装",
        ],
        "clarify_fields": ["订单号", "签收时间", "商品状态", "商品类型", "图片凭证"],
    },
    "refund_cancel": {
        "title": "退款与取消订单",
        "keywords": ["退款", "取消订单", "原路返回", "到账", "信用卡", "全额退款"],
        "checklist": [
            "确认订单状态：待发货、已发货、已签收或售后中",
            "确认退款原因与退款路径",
            "说明退款时效通常与支付方式有关",
            "如题目缺关键信息，应提示需核实订单与支付记录",
        ],
        "clarify_fields": ["订单号", "支付方式", "订单状态", "退款原因"],
    },
    "invoice": {
        "title": "发票与抬头",
        "keywords": ["发票", "抬头", "公司", "开票", "重开", "纸质版", "电子版"],
        "checklist": [
            "确认发票类型：电子、纸质、个人或企业抬头",
            "确认开票信息是否填写错误",
            "确认是否已开具以及是否支持重开",
            "提醒用户提供订单号与开票信息截图",
        ],
        "clarify_fields": ["订单号", "发票类型", "抬头信息", "税号"],
    },
    "shipping_delivery": {
        "title": "物流配送与运费",
        "keywords": ["物流", "快递", "运费", "揽收", "补寄", "国外", "乡镇", "多久", "送达", "丢失"],
        "checklist": [
            "确认是未发货、运输中、待揽收、丢件还是签收异常",
            "确认收货地区、物流单号和下单时间",
            "如涉及补寄或索赔，提示保留物流记录和外包装",
            "回答应覆盖时效、运费或赔付流程中的对应部分",
        ],
        "clarify_fields": ["订单号", "物流单号", "收货地区", "签收状态"],
    },
    "packaging_damage": {
        "title": "包装破损、少件、污损与到货异常",
        "keywords": ["包装破损", "少发", "少了一件", "污渍", "二手", "损坏", "划痕", "瑕疵"],
        "checklist": [
            "确认问题类型：破损、少件、污损、瑕疵或疑似二手",
            "确认是否签收、是否当场验货、是否已使用",
            "提示提供外包装、面单、商品细节照片",
            "区分补发、换货、退货退款等处理路径",
        ],
        "clarify_fields": ["订单号", "签收时间", "问题照片", "是否使用"],
    },
    "repair_warranty": {
        "title": "维修、质保与售后范围",
        "keywords": ["维修", "售后", "质保", "保修", "人为损坏", "配件费", "翻新机", "修好"],
        "checklist": [
            "确认商品是否在质保期内",
            "确认是质量问题还是人为损坏",
            "确认维修状态：待寄回、维修中、超时未修好或二次故障",
            "若涉及收费，区分检测费、配件费和超保维修费",
        ],
        "clarify_fields": ["订单号", "故障描述", "购买时间", "维修单号"],
    },
    "complaint_rights": {
        "title": "投诉、赔偿与权益争议",
        "keywords": ["投诉", "赔偿", "辱骂", "假货", "虚假宣传", "翻新机", "态度差", "正品"],
        "checklist": [
            "明确投诉对象：商品、物流、安装人员、客服或售后",
            "梳理证据：页面宣传、聊天记录、通话记录、照片/视频",
            "区分投诉诉求：道歉、退款、换货、赔偿或升级处理",
            "避免直接承诺赔偿金额，优先说明需核实处理",
        ],
        "clarify_fields": ["订单号", "证据材料", "具体诉求", "发生时间"],
    },
    "installation_service": {
        "title": "上门安装与上门检修",
        "keywords": ["安装", "上门", "配件费", "检修", "安装人员", "大家电"],
        "checklist": [
            "确认是安装咨询、安装异常还是检修安排",
            "确认是否为免费安装范围内项目",
            "若涉及额外收费，说明需核实收费项目与标准",
            "如发生安装损坏，提示保留现场照片和服务记录",
        ],
        "clarify_fields": ["订单号", "商品型号", "预约时间", "收费明细"],
    },
    "sample_trial": {
        "title": "试用装与试用期",
        "keywords": ["试用装", "试用", "延长试用", "故障商品能更换吗"],
        "checklist": [
            "确认是否存在试用活动或试用装规则",
            "确认试用期间商品状态和故障原因",
            "区分试用期延长、换货和退款诉求",
            "提示需要核实活动规则与订单信息",
        ],
        "clarify_fields": ["活动信息", "订单号", "试用状态", "故障描述"],
    },
    "manual_instruction": {
        "title": "说明书与资料获取",
        "keywords": ["说明书", "电子版", "纸质版", "使用手册", "哪里可以找到"],
        "checklist": [
            "确认需要纸质版还是电子版资料",
            "若为产品操作问题，优先引导到对应手册内容",
            "若为资料缺失问题，说明需核实订单和商品型号",
            "涉及图片/图示时，优先返回对应手册配图",
        ],
        "clarify_fields": ["商品型号", "订单号", "资料类型"],
    },
}

SERVICE_KEYWORDS = sorted({kw for spec in INTENT_SPECS.values() for kw in spec["keywords"]}, key=len, reverse=True)


def normalize_question(raw_question: str) -> str:
    """
    将CSV中的原始问题文本归一化为干净的格式。

    CSV字段值通常被双引号包裹，且内部引号会被转义。
    本函数处理以下转义模式:
    - '""' -> '"'
    - 首尾多余引号
    - CSV单元格内的换行符（通常为'\n'字符串）
    - 单元格间逗号分隔导致的行内逗号

    Args:
        raw_question: CSV字段中的原始问题字符串

    Returns:
        归一化后的单行或多行问题文本
    """
    text = (raw_question or "").strip()
    text = text.replace('""', '"')
    # 去掉首尾多余引号
    text = re.sub(r'^\s*"', "", text)
    text = re.sub(r'"\s*$', "", text)
    # CSV内逗号分隔的单元格转换为换行
    text = re.sub(r'"\s*,\s*"', "\n", text)
    text = text.replace('",\n"', "\n")
    text = text.replace('","', "\n")
    # 处理转义换行符
    text = text.replace("\\n", "\n")
    lines = []
    for line in text.splitlines():
        # 清理每行首尾引号和逗号
        line = line.strip().strip('"').strip("，")
        if line:
            lines.append(line)
    return "\n".join(lines)


def classify_intents(question: str) -> List[str]:
    """
    根据关键词匹配判断问题属于哪些客服意图。

    匹配方式: 若问题文本包含某意图的任一关键词，则认为命中该意图。
    一个问题可能同时命中多个意图（如同时包含"退货"和"说明书"）。

    Args:
        question: 归一化后的问题文本

    Returns:
        命中的意图类型列表（可能为空）
    """
    intents = []
    for intent, spec in INTENT_SPECS.items():
        if any(keyword in question for keyword in spec["keywords"]):
            intents.append(intent)
    return intents


def classify_route(question: str, intents: List[str]) -> str:
    """
    根据意图列表判断路由类型。

    路由判断规则:
    - 无意图命中 -> manual（产品手册路由）
    - 仅有manual_instruction -> manual
    - 有manual_instruction + 其他服务意图 -> mixed
    - 有非manual意图 -> service（客服路由）

    Args:
        question: 归一化后的问题文本
        intents: classify_intents返回的意图列表

    Returns:
        路由类型: "manual", "service" 或 "mixed"
    """
    if not intents:
        return "manual"

    has_manual_signal = any(keyword in question for keyword in INTENT_SPECS["manual_instruction"]["keywords"])
    has_service_signal = any(
        intent != "manual_instruction"
        for intent in intents
    )

    if has_manual_signal and has_service_signal:
        return "mixed"
    if intents:
        return "service"
    return "manual"


def load_questions(csv_path: Path) -> List[Dict[str, str]]:
    """
    从CSV文件加载公开题并做路由标注。

    处理步骤:
    1. 读取CSV（跳过BOM头）
    2. 对每条问题归一化
    3. 匹配客服意图
    4. 判断路由类型
    5. 记录id/question/route/intents

    Args:
        csv_path: question_public.csv文件路径

    Returns:
        路由标注后的问题列表
    """
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = normalize_question(row["question"])
            intents = classify_intents(normalized)
            rows.append(
                {
                    "id": str(row["id"]).strip(),
                    "question": normalized,
                    "route": classify_route(normalized, intents),
                    "intents": intents,
                }
            )
    return rows


def build_playbook_docs(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped_questions: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        for intent in row["intents"]:
            grouped_questions[intent].append(row["question"])

    docs: List[Dict[str, Any]] = []
    for intent, spec in INTENT_SPECS.items():
        sample_questions = grouped_questions.get(intent, [])[:8]
        lines = [
            f"客服意图：{spec['title']}",
            f"命中关键词：{'、'.join(spec['keywords'])}",
            "回答时建议覆盖以下要点：",
        ]
        lines.extend([f"{idx}. {item}" for idx, item in enumerate(spec["checklist"], 1)])
        lines.append("如题目信息不足，优先补充核实以下字段：")
        lines.extend([f"- {field}" for field in spec["clarify_fields"]])
        if sample_questions:
            lines.append("相似公开题示例：")
            lines.extend([f"- {question.replace(chr(10), ' / ')}" for question in sample_questions])

        docs.append(
            {
                "doc_id": f"service_playbook_{intent}",
                "content": "\n".join(lines),
                "metadata": {
                    "route": "service",
                    "doc_type": "service_playbook",
                    "intent": intent,
                    "title": spec["title"],
                    "priority": 0.82,
                    "sample_count": len(grouped_questions.get(intent, [])),
                },
            }
        )
    return docs


def build_example_docs(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for row in rows:
        docs.append(
            {
                "doc_id": f"route_example_{row['id']}",
                "content": row["question"],
                "metadata": {
                    "route": row["route"],
                    "doc_type": "question_example",
                    "question_id": row["id"],
                    "intents": row["intents"],
                },
            }
        )
    return docs


def load_policy_seed_docs(seed_path: Path, rows: Sequence[Dict[str, str]], qa_seed_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """加载客服政策种子文档，包括基础政策和扩充 QA"""
    if not seed_path.exists():
        raise FileNotFoundError(f"未找到客服政策种子文件: {seed_path}")

    all_docs = []

    # 1. 加载基础政策种子
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    all_docs.extend(payload.get("documents", []))

    # 2. 加载扩充 QA 种子（如果存在）
    if qa_seed_path and qa_seed_path.exists():
        try:
            qa_payload = json.loads(qa_seed_path.read_text(encoding="utf-8"))
            all_docs.extend(qa_payload.get("documents", []))
        except Exception:
            pass

    grouped_questions: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        for intent in row["intents"]:
            grouped_questions[intent].append(row["question"])

    docs: List[Dict[str, Any]] = []
    for raw_doc in all_docs:
        intent = raw_doc["intent"]
        sample_questions = grouped_questions.get(intent, [])[:6]

        lines = [
            f"【客服主题】{raw_doc['title']}",
            f"【适用意图】{intent}",
        ]
        keywords = raw_doc.get("keywords", [])
        if keywords:
            lines.append(f"【关键词】{'、'.join(keywords)}")

        applies_to = raw_doc.get("applies_to", [])
        if applies_to:
            lines.append("【适用场景】")
            lines.extend([f"{idx}. {item}" for idx, item in enumerate(applies_to, 1)])

        answer_points = raw_doc.get("answer_points", [])
        if answer_points:
            lines.append("【标准答复】")
            lines.extend([f"{idx}. {item}" for idx, item in enumerate(answer_points, 1)])

        follow_up_actions = raw_doc.get("follow_up_actions", [])
        if follow_up_actions:
            lines.append("【处理动作】")
            lines.extend([f"{idx}. {item}" for idx, item in enumerate(follow_up_actions, 1)])

        clarify_fields = raw_doc.get("clarify_fields", [])
        if clarify_fields:
            lines.append("【需核实信息】")
            lines.extend([f"- {field}" for field in clarify_fields])

        guardrails = raw_doc.get("guardrails", [])
        if guardrails:
            lines.append("【边界提示】")
            lines.extend([f"- {item}" for item in guardrails])

        if sample_questions:
            lines.append("【相似公开题】")
            lines.extend([f"- {question.replace(chr(10), ' / ')}" for question in sample_questions])

        docs.append(
            {
                "doc_id": raw_doc["doc_id"],
                "content": "\n".join(lines),
                "metadata": {
                    "route": "service",
                    "doc_type": "service_policy",
                    "intent": intent,
                    "title": raw_doc["title"],
                    "keywords": keywords,
                    "priority": float(raw_doc.get("priority", 1.0)),
                    "clarify_fields": clarify_fields,
                },
            }
        )

    return docs


def build_payload(rows: List[Dict[str, str]], seed_path: Path, qa_seed_path: Optional[Path] = None) -> Dict[str, Any]:
    route_examples = build_example_docs(rows)
    policy_docs = load_policy_seed_docs(seed_path, rows, qa_seed_path)
    playbook_docs = build_playbook_docs(rows)
    service_docs = policy_docs + playbook_docs
    return {
        "version": "2.0.0",
        "question_count": len(rows),
        "service_question_count": sum(1 for row in rows if row["route"] in ("service", "mixed")),
        "manual_question_count": sum(1 for row in rows if row["route"] == "manual"),
        "mixed_question_count": sum(1 for row in rows if row["route"] == "mixed"),
        "intents": INTENT_SPECS,
        "route_examples": route_examples,
        "service_policy_documents": policy_docs,
        "service_playbook_documents": playbook_docs,
        "service_documents": service_docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建双路检索中的客服路由知识库")
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "question_public.csv"),
        help="公开题 CSV 路径",
    )
    parser.add_argument(
        "--output",
        default=str(settings.route_kb_path / settings.service_route_kb_file),
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--seed",
        default=str(settings.service_policy_seed_path / settings.service_policy_seed_file),
        help="客服政策种子 JSON 路径",
    )
    parser.add_argument(
        "--qa-seed",
        default=str(settings.service_policy_seed_path / "service_qa_seed.json"),
        help="扩充客服 QA 种子 JSON 路径",
    )
    args = parser.parse_args()

    question_path = Path(args.questions)
    output_path = Path(args.output)
    seed_path = Path(args.seed)
    qa_seed_path = Path(args.qa_seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_questions(question_path)
    payload = build_payload(rows, seed_path=seed_path, qa_seed_path=qa_seed_path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已生成客服路由知识库: {output_path}")
    print(
        json.dumps(
            {
                "question_count": payload["question_count"],
                "service_question_count": payload["service_question_count"],
                "manual_question_count": payload["manual_question_count"],
                "mixed_question_count": payload["mixed_question_count"],
                "policy_doc_count": len(payload["service_policy_documents"]),
                "playbook_count": len(payload["service_playbook_documents"]),
                "service_doc_count": len(payload["service_documents"]),
                "example_count": len(payload["route_examples"]),
                "qa_seed_loaded": qa_seed_path.exists(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
