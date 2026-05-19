"""
尝试从其他来源下载 CLIP 模型。

方案1: 使用较小的 CLIP 模型 openai/clip-vit-base-patch32
方案2: 使用 OpenCLIP 兼容模型
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

# 尝试多个镜像源
MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",  # 原始源
]

MODEL_ID = "openai/clip-vit-base-patch32"  # 使用较小的模型 (86MB vs 900MB)

CACHE_DIR = Path("D:/models/huggingface/hub/models--openai--clip-vit-base-patch32")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"尝试下载较小 CLIP 模型: {MODEL_ID}")
print(f"缓存目录: {CACHE_DIR}")


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """下载文件并显示进度"""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


# 尝试从镜像下载
for mirror in MIRRORS:
    print(f"\n尝试从 {mirror} 下载...")
    
    files_to_download = [
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",  # 较小的模型使用 bin 格式
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    success_count = 0
    
    for filename in files_to_download:
        filepath = CACHE_DIR / filename
        if filepath.exists() and filepath.stat().st_size > 1000:
            print(f"  {filename} 已存在，跳过")
            success_count += 1
            continue
        
        url = f"{mirror}/{MODEL_ID}/resolve/main/{filename}"
        print(f"  下载 {filename}...")
        
        try:
            download_file(url, filepath)
            print(f"  完成: {filename}")
            success_count += 1
        except Exception as e:
            print(f"  失败: {filename} - {e}")
            # 如果是镜像失败，尝试下一个镜像
            if mirror != MIRRORS[-1]:
                continue
            break
    
    if success_count >= 3:  # 如果成功下载了足够的文件
        print(f"\n从 {mirror} 下载成功!")
        break

print("\n检查下载的文件:")
for f in CACHE_DIR.glob("*"):
    if f.is_file():
        size = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size:.2f} MB")
