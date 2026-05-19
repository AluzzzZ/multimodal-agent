"""
生产环境测试 - 使用 Qwen3-VL 进行真实图片分析。

用法：
    python scripts/test_vision_production.py
"""

import os
import sys
from pathlib import Path

# 设置 HF_HOME
os.environ["HF_HOME"] = "D:/models/huggingface"

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from PIL import Image
import base64
import io


def load_image(path: str) -> Image.Image:
    """加载图片"""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def image_to_base64(img: Image.Image) -> str:
    """将图片转换为 base64"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def test_vision_llm():
    """测试视觉大模型"""
    print("=" * 60)
    print("生产环境测试 - Qwen3-VL 视觉理解")
    print("=" * 60)

    from openai import OpenAI

    api_key = settings.vision_llm_api_key
    base_url = settings.vision_llm_base_url
    model = settings.vision_llm_model

    print(f"\n配置信息:")
    print(f"  API URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  Vision Enabled: {settings.vision_llm_enabled}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 测试不同类型的图片
    test_cases = [
        {
            "name": "产品指示灯",
            "image_path": None,  # 使用生成的测试图片
            "description": "电钻侧面照片，顶部有一个LED指示灯呈红色闪烁状态",
        },
    ]

    print("\n" + "-" * 60)
    print("测试案例")
    print("-" * 60)

    for i, test in enumerate(test_cases):
        print(f"\n【测试 {i+1}】{test['name']}")

        # 创建测试图片
        img = Image.new('RGB', (400, 300), color='lightgray')
        img_base64 = image_to_base64(img)

        prompt = f"""请分析这张图片，用中文回答：

1. 产品类型是什么？
2. 可见的部件有哪些？
3. 有什么状态或特征？

回答格式：
产品类型：[答案]
可见部件：[答案]
状态特征：[答案]

控制在100字以内。"""

        print(f"\n发送请求到 {model}...")
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
                max_tokens=200,
                temperature=0.3,
            )
            result = response.choices[0].message.content.strip()
            print(f"\n✅ 成功!")
            print(f"\n分析结果:\n{result}")

        except Exception as e:
            print(f"\n❌ 失败: {e}")
            import traceback
            traceback.print_exc()


def test_multimodal_module():
    """测试多模态模块"""
    print("\n" + "=" * 60)
    print("测试多模态理解模块 (使用视觉大模型)")
    print("=" * 60)

    from src.modules.multimodal_understanding import MultimodalUnderstanding

    mm = MultimodalUnderstanding()

    # 初始化（会加载视觉大模型）
    print("\n初始化多模态模块...")
    mm.initialize()
    print(f"  Caption 模型: {mm.image_parser._caption_model}")
    print(f"  视觉模型: {settings.vision_model}")
    print(f"  视觉大模型启用: {settings.vision_llm_enabled}")

    # 创建测试图片
    img = Image.new('RGB', (400, 300), color='lightgray')
    img_base64 = "data:image/png;base64," + image_to_base64(img)

    # 测试分析
    question = "这个产品是什么？有什么指示灯？"

    print(f"\n测试问题: {question}")
    result = mm.analyze(
        question=question,
        images=[img_base64],
    )

    print(f"\n分析结果:")
    print(f"  意图: {result.evidence_type}")
    print(f"  产品候选: {result.product_candidates}")
    print(f"  图片标签: {result.image_tags}")
    print(f"  视觉意图: {result.visual_intents}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Qwen3-VL 生产环境测试")
    print("=" * 60)

    # 测试 1: 直接调用 API
    test_vision_llm()

    # 测试 2: 通过多模态模块
    # test_multimodal_module()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
