"""
使用 requests 从 HF Mirror 下载 CLIP 模型文件。

用法：
    python scripts/download_clip_model.py
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

# 设置镜像
HF_MIRROR = "https://hf-mirror.com"
MODEL_ID = "openai/clip-vit-large-patch14"
CACHE_DIR = Path("D:/models/huggingface/hub/models--openai--clip-vit-large-patch14")

# 创建缓存目录
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"正在从 HF Mirror 下载 CLIP 模型: {MODEL_ID}")
print(f"使用镜像: {HF_MIRROR}")
print(f"缓存目录: {CACHE_DIR}")

# 需要下载的文件列表
model_files = [
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
]


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """下载文件并显示进度"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


# 先下载小文件
print("\n下载配置文件...")
for filename in model_files:
    if filename.endswith(".safetensors"):
        continue  # 先跳过大的 safetensors 文件
    
    filepath = CACHE_DIR / filename
    if filepath.exists():
        print(f"  {filename} 已存在，跳过")
        continue
    
    url = f"{HF_MIRROR}/{MODEL_ID}/resolve/main/{filename}"
    print(f"  下载 {filename}...")
    
    try:
        download_file(url, filepath)
        print(f"  完成: {filename}")
    except Exception as e:
        print(f"  失败: {filename} - {e}")

# 下载 safetensors 文件（最大）
print("\n下载模型文件（分片下载）...")
safetensors_files = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]

for filename in safetensors_files:
    filepath = CACHE_DIR / filename
    if filepath.exists():
        size = filepath.stat().st_size
        if size > 100_000_000:  # 大于 100MB 认为下载完成
            print(f"  {filename} 已存在 ({size / (1024*1024):.1f} MB)，跳过")
            continue
    
    url = f"{HF_MIRROR}/{MODEL_ID}/resolve/main/{filename}"
    print(f"  下载 {filename}...")
    
    try:
        download_file(url, filepath)
        size = filepath.stat().st_size / (1024 * 1024)
        print(f"  完成: {filename} ({size:.1f} MB)")
    except Exception as e:
        print(f"  失败: {filename} - {e}")

print("\n检查下载的文件:")
total_size = 0
for f in CACHE_DIR.glob("*"):
    if f.is_file():
        size = f.stat().st_size
        total_size += size
        size_mb = size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

print(f"\n总计下载大小: {total_size / (1024 * 1024):.1f} MB")
print(f"\n下载完成! 缓存目录: {CACHE_DIR}")
