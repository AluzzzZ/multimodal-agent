"""
使用 OCR API 测试图片描述能力。

注意：由于评测集只有文字描述没有真实图片，
这里使用 OCR API 对描述文本进行"增强理解"来模拟视觉分析。

用法：
    python scripts/test_ocr_caption.py
"""

import os
import sys

# 设置 HF_HOME
os.environ["HF_HOME"] = "D:/models/huggingface"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
import json

def generate_caption_from_description(description: str) -> str:
    """
    使用 OCR/视觉模型增强描述。
    由于没有真实图片，这里用 OCR API 对描述进行结构化分析。
    """
    from openai import OpenAI

    api_key = settings.vision_llm_api_key
    base_url = settings.vision_llm_base_url
    model = settings.vision_llm_model

    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = f"""你是一个产品图片分析助手。请根据以下图片描述，提取关键信息：

图片描述：{description}

请按以下格式回答（仅输出JSON，不要其他内容）：
{{
    "product_type": "产品类型",
    "visible_parts": ["可见部件1", "部件2"],
    "status": "状态/外观描述",
    "summary": "一句话总结（50字以内）"
}}

回答："""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3,
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"API 调用失败: {e}")
        return None


def test_on_eval_samples():
    """在评测样本上测试"""
    print("=" * 60)
    print("使用 OCR API 增强图片描述")
    print("=" * 60)

    # 加载评测数据
    with open("data/multimodal_eval_set.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"][:5]  # 只测试前5条

    print(f"\n测试 {len(items)} 个样本...\n")

    for i, item in enumerate(items):
        question = item["question"]
        description = item["image_description"]
        expected_tags = item["expected"]["image_tags"]

        print(f"【样本 {i+1}】")
        print(f"问题: {question}")
        print(f"描述: {description}")

        # 使用 OCR API 增强
        result = generate_caption_from_description(description)
        if result:
            print(f"OCR 增强结果:")
            print(result)
        else:
            print("API 失败")

        print(f"期望标签: {expected_tags}")
        print("-" * 60)


if __name__ == "__main__":
    test_on_eval_samples()
