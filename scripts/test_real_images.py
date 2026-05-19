"""
使用 Qwen3-VL 分析真实产品图片。

用法：
    python scripts/test_real_images.py
"""

import os
import sys
from pathlib import Path
import requests
from PIL import Image
import io
import base64

# 设置 HF_HOME
os.environ["HF_HOME"] = "D:/models/huggingface"

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings


def download_image(url: str) -> Image.Image:
    """从 URL 下载图片"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def image_to_base64(img: Image.Image) -> str:
    """将图片转换为 base64"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def analyze_image(img: Image.Image, question: str = None) -> str:
    """使用 Qwen3-VL 分析图片"""
    from openai import OpenAI

    api_key = settings.vision_llm_api_key
    base_url = settings.vision_llm_base_url
    model = settings.vision_llm_model

    client = OpenAI(api_key=api_key, base_url=base_url)

    if question is None:
        prompt = """请详细分析这张图片，用中文回答：

1. 产品类型（如：电钻、遥控器、健身器材、电子设备等）
2. 可见的部件（如：指示灯、按钮、屏幕、表带、接口等）
3. 状态或外观特征（如：指示灯颜色、屏幕显示内容、外观是否完好等）

请简洁回答，控制在150字以内。"""
    else:
        prompt = f"""请根据用户的问题分析这张图片：

问题：{question}

请用中文回答，控制在100字以内。"""

    img_base64 = image_to_base64(img)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"分析失败: {e}"


def test_with_sample_images():
    """使用样例图片测试"""
    print("=" * 60)
    print("Qwen3-VL 真实图片分析测试")
    print("=" * 60)

    # 测试图片 URL（电动工具产品图）
    test_images = [
        {
            "name": "电动工具",
            "url": "https://images.unsplash.com/photo-1504148455328-c376907d081c?w=400",
            "question": "这是什么产品？有什么特点？"
        },
        {
            "name": "电子产品",
            "url": "https://images.unsplash.com/photo-1550009158-9ebf69173e03?w=400",
            "question": "这是什么设备？"
        },
        {
            "name": "健身器材",
            "url": "https://images.unsplash.com/photo-1576678927484-cc907957088c?w=400",
            "question": "这是什么健身器材？"
        },
    ]

    for i, test in enumerate(test_images):
        print(f"\n【测试 {i+1}】{test['name']}")
        print(f"URL: {test['url']}")

        try:
            print("下载图片...")
            img = download_image(test['url'])
            print(f"图片尺寸: {img.size}")

            print("发送分析请求...")
            result = analyze_image(img, test['question'])

            print(f"\n问题: {test['question']}")
            print(f"\n分析结果:\n{result}")

        except Exception as e:
            print(f"❌ 失败: {e}")

        print("-" * 60)


def test_multimodal_integration():
    """测试多模态理解模块集成"""
    print("\n" + "=" * 60)
    print("测试多模态理解模块集成")
    print("=" * 60)

    from src.modules.multimodal_understanding import MultimodalUnderstanding

    # 初始化模块
    print("\n初始化多模态模块...")
    mm = MultimodalUnderstanding()
    mm.initialize()

    print(f"视觉大模型启用: {settings.vision_llm_enabled}")
    print(f"Caption 模型: {mm.image_parser._caption_model}")

    # 下载测试图片
    print("\n下载测试图片...")
    img = download_image("https://images.unsplash.com/photo-1504148455328-c376907d081c?w=400")

    # 转换为 base64
    img_base64 = "data:image/png;base64," + image_to_base64(img)

    # 测试分析
    question = "这是什么产品？有什么指示灯？状态如何？"

    print(f"\n问题: {question}")
    print("分析中...")

    result = mm.analyze(
        question=question,
        images=[img_base64],
    )

    print(f"\n分析结果:")
    print(f"  证据类型: {result.evidence_type}")
    print(f"  产品候选: {result.product_candidates}")
    print(f"  图片标签: {result.image_tags}")
    print(f"  视觉意图: {result.visual_intents}")


if __name__ == "__main__":
    # 测试 1: 直接分析真实图片
    test_with_sample_images()

    # 测试 2: 多模态模块集成
    # test_multimodal_integration()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
