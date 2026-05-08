"""
图片处理工具模块

职责:
1. 提供 Base64 图片的验证、解码、编码等基础操作
2. 支持图片尺寸调整、裁剪等预处理
3. 提供从文本中提取图片 ID 的工具函数

依赖:
- PIL (Pillow): 图片解码与处理
- base64: Base64 编解码
"""

import base64
import io
import re
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np
from loguru import logger

from config import settings


class ImageProcessor:
    """
    图片处理器 - 提供图片验证、解码、编码、尺寸调整等操作。

    所有方法均为类方法（classmethod），无需实例化即可调用。
    """

    # 支持的图片格式映射（小写扩展名 -> PIL 格式名称）
    SUPPORTED_FORMATS = {
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "png": "PNG",
        "webp": "WEBP",
        "bmp": "BMP"
    }
    
    @classmethod
    def validate_image(cls, image_data: str) -> Tuple[bool, str]:
        """
        验证图片数据
        
        Args:
            image_data: Base64编码的图片数据
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            # 提取数据部分
            if ',' in image_data:
                data_part = image_data.split(',')[1]
            else:
                data_part = image_data
            
            # 解码并验证
            image_bytes = base64.b64decode(data_part)
            
            # 检查大小
            if len(image_bytes) > settings.max_image_size:
                return False, f"图片大小超过限制 ({settings.max_image_size // (1024*1024)}MB)"
            
            # 验证图片
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            
            return True, ""
            
        except Exception as e:
            return False, f"图片验证失败: {str(e)}"
    
    @classmethod
    def decode_image(cls, image_data: str) -> Optional[Image.Image]:
        """
        解码Base64图片
        
        Args:
            image_data: Base64编码的图片数据
            
        Returns:
            PIL Image对象
        """
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return image
            
        except Exception as e:
            logger.error(f"图片解码失败: {e}")
            return None
    
    @classmethod
    def encode_image(cls, image: Image.Image, format: str = "PNG") -> str:
        """
        将图片编码为Base64
        
        Args:
            image: PIL Image对象
            format: 输出格式
            
        Returns:
            Base64编码的图片数据
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            buffer.seek(0)
            
            encoded = base64.b64encode(buffer.read()).decode('utf-8')
            return f"data:image/{format.lower()};base64,{encoded}"
            
        except Exception as e:
            logger.error(f"图片编码失败: {e}")
            return ""
    
    @classmethod
    def resize_image(
        cls,
        image: Image.Image,
        max_size: Tuple[int, int] = (1024, 1024),
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        调整图片大小
        
        Args:
            image: PIL Image对象
            max_size: 最大尺寸 (width, height)
            maintain_aspect: 是否保持宽高比
            
        Returns:
            调整后的图片
        """
        if not maintain_aspect:
            return image.resize(max_size, Image.Resampling.LANCZOS)
        
        # 计算缩放比例
        width, height = image.size
        max_width, max_height = max_size
        
        ratio = min(max_width / width, max_height / height)
        
        if ratio >= 1:
            return image
        
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    @classmethod
    def crop_to_aspect(
        cls,
        image: Image.Image,
        aspect_ratio: float = 16/9
    ) -> Image.Image:
        """
        按宽高比裁剪图片
        
        Args:
            image: PIL Image对象
            aspect_ratio: 目标宽高比
            
        Returns:
            裁剪后的图片
        """
        width, height = image.size
        current_ratio = width / height
        
        if current_ratio > aspect_ratio:
            # 图片更宽，裁剪左右
            new_width = int(height * aspect_ratio)
            left = (width - new_width) // 2
            return image.crop((left, 0, left + new_width, height))
        else:
            # 图片更高，裁剪上下
            new_height = int(width / aspect_ratio)
            top = (height - new_height) // 2
            return image.crop((0, top, width, top + new_height))
    
    @classmethod
    def get_image_info(cls, image: Image.Image) -> dict:
        """
        获取图片信息
        
        Args:
            image: PIL Image对象
            
        Returns:
            图片信息字典
        """
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
            "aspect_ratio": round(image.width / image.height, 2)
        }


def extract_image_ids_from_text(text: str) -> List[str]:
    """
    从文本中提取图片 ID。

    图片 ID 以 [xxx] 格式嵌入文本中，例如 "<PIC>[image_001]" 表示引用 image_001。
    此函数用于从手册文本中解析出所有引用的图片 ID，供后续图片检索使用。

    Args:
        text: 包含 <PIC> 标记的文本内容

    Returns:
        图片 ID 列表，例如 ["image_001", "image_002"]

    Example:
        >>> extract_image_ids_from_text("请参见图示 <PIC>[fig_001] 和 [fig_002]")
        ['fig_001', 'fig_002']
    """
    if not text:
        return []

    # 支持两类常见格式：
    # 1. <PIC>[image_001]
    # 2. 普通引用 [image_001]
    # 仅提取中括号中的 ID，并按出现顺序去重。
    pattern = re.compile(r"\[([^\[\]\s]+)\]")
    seen = set()
    image_ids: List[str] = []

    for match in pattern.finditer(text):
        image_id = match.group(1).strip()
        if not image_id or image_id in seen:
            continue
        seen.add(image_id)
        image_ids.append(image_id)

    return image_ids


def insert_picture_markers(
    text: str,
    image_ids: List[str]
) -> str:
    """
    在文本中插入图片标记
    
    Args:
        text: 原始文本
        image_ids: 图片ID列表
        
    Returns:
        插入标记后的文本
    """
    if not image_ids:
        return text
    
    # 在适当位置插入<PIC>标记
    result = text
    
    # 简单策略：在文本中插入图片引用
    for img_id in image_ids:
        marker = f"<PIC>[{img_id}]"
        # 在段落末尾添加
        if "\n\n" in result:
            # 在最后一段后添加
            parts = result.rsplit("\n\n", 1)
            result = parts[0] + f"\n\n{marker}\n\n" + (parts[1] if len(parts) > 1 else "")
        else:
            result = result + f"\n\n{marker}"
    
    return result
