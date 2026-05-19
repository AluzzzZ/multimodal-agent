"""
为评测集生成图片。

使用 PIL 生成符合描述的图片，然后进行评测。

用法：
    python scripts/generate_eval_images.py
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import base64
import io

# 设置 HF_HOME
os.environ["HF_HOME"] = "D:/models/huggingface"

def create_simple_image(description: str, output_path: str = None) -> Image.Image:
    """
    根据描述创建简单的图片。

    这是一个简化的实现，实际应该使用更复杂的生成模型。
    这里创建基于关键词的示意图片。
    """
    # 创建基础图片
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)

    # 根据描述关键词选择颜色和形状
    description_lower = description.lower()

    # 检测产品类型
    colors = {
        '电钻': ('gray', 'orange'),
        '遥控器': ('darkblue', 'lightblue'),
        '健身': ('green', 'black'),
        '手环': ('purple', 'pink'),
        '手表': ('black', 'silver'),
        '空气': ('white', 'blue'),
        '键盘': ('black', 'gray'),
    }

    base_color = 'lightgray'
    accent_color = 'red'

    for product, colors in colors.items():
        if product in description_lower:
            base_color, accent_color = colors
            break

    # 绘制背景和产品轮廓
    img = Image.new('RGB', (400, 300), color=base_color)
    draw = ImageDraw.Draw(img)

    # 绘制一个矩形表示产品
    draw.rectangle([50, 50, 350, 250], outline=accent_color, width=3)

    # 如果描述提到指示灯，绘制一个小圆
    if '指示灯' in description_lower or 'led' in description_lower.lower():
        if '红' in description_lower:
            draw.ellipse([300, 80, 330, 110], fill='red')
        elif '绿' in description_lower:
            draw.ellipse([300, 80, 330, 110], fill='green')
        else:
            draw.ellipse([300, 80, 330, 110], fill='yellow')

    # 如果描述提到按钮，绘制按钮
    if '按钮' in description_lower:
        draw.ellipse([200, 120, 230, 150], fill='darkblue')

    # 如果描述提到屏幕，绘制矩形屏幕
    if '屏幕' in description_lower:
        draw.rectangle([150, 100, 250, 180], fill='lightblue', outline='black')

    # 添加文字说明
    try:
        draw.text((20, 280), "Generated Image", fill='black')
    except:
        pass

    return img


def image_to_base64(img: Image.Image) -> str:
    """将图片转换为 base64 字符串"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_and_save_images():
    """为评测集生成图片"""
    print("加载评测数据...")

    with open("data/multimodal_eval_set.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"]
    print(f"共有 {len(items)} 个评测样本")

    output_dir = Path("data/eval_images")
    output_dir.mkdir(exist_ok=True)

    print(f"\n生成图片到 {output_dir}...")

    for i, item in enumerate(items):
        description = item.get("image_description", "")
        item_id = item.get("id", f"item_{i}")

        # 生成图片
        img = create_simple_image(description)

        # 保存图片
        img_path = output_dir / f"{item_id}.png"
        img.save(img_path)

        # 生成 base64
        img_base64 = "data:image/png;base64," + image_to_base64(img)

        # 更新 item
        item["generated_image"] = img_base64
        item["generated_image_path"] = str(img_path)

        if (i + 1) % 20 == 0:
            print(f"已生成 {i + 1}/{len(items)} 张图片")

    # 保存带图片的数据
    output_file = output_dir / "eval_with_images.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！图片已保存到 {output_dir}")
    print(f"带图片的评测数据已保存到 {output_file}")


if __name__ == "__main__":
    generate_and_save_images()
