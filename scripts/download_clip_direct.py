"""
直接从 HuggingFace.co 下载较小的 CLIP 模型文件。

使用 openai/clip-vit-base-patch32 (约 86MB) 而不是大的 clip-vit-large-patch14 (约 900MB)
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

HF_ENDPOINT = "https://huggingface.co"
MODEL_ID = "openai/clip-vit-base-patch32"

CACHE_DIR = Path("D:/models/huggingface/hub/models--openai--clip-vit-base-patch32")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"从 HuggingFace.co 下载 CLIP 模型: {MODEL_ID}")
print(f"缓存目录: {CACHE_DIR}")


def download_with_progress(url: str, dest: Path):
    """下载文件并显示进度"""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    filename = dest.name
    
    print(f"\n下载 {filename} ({total_size / (1024*1024):.1f} MB)...")
    
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size * 100
                    if downloaded % (1024*1024*10) == 0:  # 每10MB显示一次
                        print(f"  已下载: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB ({progress:.1f}%)")
    
    return dest.stat().st_size


# 需要下载的文件
files = [
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]

for filename in files:
    filepath = CACHE_DIR / filename
    
    # 检查是否已存在
    if filepath.exists():
        size = filepath.stat().st_size
        if size > 1000:  # 大于1KB认为有效
            print(f"已存在: {filename} ({size/(1024*1024):.2f} MB)，跳过")
            continue
    
    url = f"{HF_ENDPOINT}/{MODEL_ID}/resolve/main/{filename}"
    
    try:
        size = download_with_progress(url, filepath)
        print(f"完成: {filename} ({size/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"失败: {filename} - {e}")

print("\n检查下载的文件:")
total_size = 0
for f in CACHE_DIR.glob("*"):
    if f.is_file():
        size = f.stat().st_size
        total_size += size
        if size > 1024*1024:
            print(f"  {f.name}: {size/(1024*1024):.1f} MB")
        else:
            print(f"  {f.name}: {size/1024:.1f} KB")

print(f"\n总计: {total_size/(1024*1024):.1f} MB")
print(f"缓存目录: {CACHE_DIR}")
