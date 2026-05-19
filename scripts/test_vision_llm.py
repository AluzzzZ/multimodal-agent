"""
测试视觉大模型 API 连接。

用法：
    python scripts/test_vision_llm.py
"""

import os
import sys

# 设置 HF_HOME
os.environ["HF_HOME"] = "D:/models/huggingface"

from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

def test_openai_vision():
    """测试 OpenAI GPT-4V API"""
    print("\n" + "="*50)
    print("测试 OpenAI GPT-4V")
    print("="*50)

    api_key = settings.vision_llm_api_key
    base_url = settings.vision_llm_base_url

    if not api_key:
        print("❌ 未配置 VISION_LLM_API_KEY")
        return False

    print(f"API URL: {base_url}")
    print(f"Model: {settings.vision_llm_model}")

    try:
        import openai
        from PIL import Image
        import io
        import base64

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # 创建一个简单的测试图片（红色背景）
        img = Image.new('RGB', (100, 100), color='red')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        print("\n发送请求...")
        response = client.chat.completions.create(
            model=settings.vision_llm_model or "gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "用一句话描述这张图片。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    ],
                }
            ],
            max_tokens=50,
        )

        caption = response.choices[0].message.content.strip()
        print(f"✅ 成功! Caption: {caption}")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_anthropic_vision():
    """测试 Anthropic Claude Vision API"""
    print("\n" + "="*50)
    print("测试 Anthropic Claude Vision")
    print("="*50)

    # 这里需要配置 ANTHROPIC_API_KEY
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ 未配置 ANTHROPIC_API_KEY 环境变量")
        return False

    try:
        import anthropic
        from PIL import Image
        import io
        import base64

        client = anthropic.Anthropic(api_key=api_key)

        # 创建测试图片
        img = Image.new('RGB', (100, 100), color='blue')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        print("\n发送请求...")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "用一句话描述这张图片。"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_base64}},
                    ],
                }
            ],
        )

        caption = response.content[0].text.strip()
        print(f"✅ 成功! Caption: {caption}")
        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


if __name__ == "__main__":
    print("视觉大模型 API 测试")
    print("="*50)

    # 测试 OpenAI
    openai_ok = test_openai_vision()

    # 测试 Anthropic (可选)
    # anthropic_ok = test_anthropic_vision()

    print("\n" + "="*50)
    print("测试完成")
    print("="*50)

    if openai_ok:
        print("\n✅ OpenAI GPT-4V 可用")
        print("   设置 VISION_LLM_ENABLED=true 启用视觉大模型")
    else:
        print("\n❌ OpenAI GPT-4V 不可用")
        print("   请检查 VISION_LLM_API_KEY 配置")
