"""
多模态客服智能体 - API服务入口
直接运行此文件启动服务: python api.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.api import app
import uvicorn

if __name__ == "__main__":
    print(f"启动多模态客服智能体 API 服务...")
    print(f"地址: http://{settings.api_host}:{settings.api_port}")
    print(f"文档: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"Token: {settings.api_token}")
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower()
    )
