"""
正确配置 huggingface_hub 使用 HF Mirror 镜像。

关键：设置 HF_HUB_OFFLINE=0 和正确配置 endpoint
"""

import os

# 方法1: 直接设置环境变量（在代码中）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"

from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "openai/clip-vit-base-patch32"  # 使用较小的模型
CACHE_DIR = Path("D:/models/huggingface")

print(f"使用 HF Mirror 下载: {MODEL_ID}")

try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=False,
        endpoint="https://hf-mirror.com",  # 直接指定 endpoint
    )
    print(f"下载成功: {path}")
except Exception as e:
    print(f"下载失败: {e}")
