"""
多模态客服智能体 - 主入口
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from config import settings

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level=settings.log_level
)

# 添加文件日志
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "app_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)


def run_api_server():
    """运行API服务器"""
    from src.api import app
    import uvicorn
    
    logger.info(f"启动API服务器: http://{settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower()
    )


def run_knowledge_builder():
    """运行知识库构建工具"""
    from scripts.build_knowledge_base import KnowledgeBaseBuilder

    logger.info("开始构建知识库...")
    logger.info(
        "当前为低内存友好模式，默认建议使用 hashing 嵌入后端；"
        "如需高质量语义检索，可通过 .env 切换到 sentence_transformer"
    )

    builder = KnowledgeBaseBuilder()
    builder.build(force_rebuild=True)

    logger.info("知识库构建完成!")


def run_interactive_mode():
    """运行交互式测试模式"""
    from src.modules import get_multimodal_understanding, get_rag_engine, get_response_generator
    
    logger.info("初始化模块...")
    
    multimodal = get_multimodal_understanding()
    multimodal.initialize()
    
    rag_engine = get_rag_engine()
    rag_engine.initialize()
    
    response_generator = get_response_generator()
    response_generator.initialize()
    
    logger.info("初始化完成! 开始交互式测试...")
    print("\n" + "="*50)
    print("多模态客服智能体 - 交互式测试")
    print("="*50)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'reset' 重置对话历史")
    print("="*50 + "\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见!")
                break
            
            if user_input.lower() == 'reset':
                conversation_history = []
                print("对话历史已重置")
                continue
            
            if not user_input:
                continue
            
            # 生成回复
            result = response_generator.generate(
                query=user_input,
                conversation_history=conversation_history
            )
            
            print(f"\n助手: {result['response']}")
            
            if result.get('images'):
                print(f"[相关图片: {', '.join(result['images'])}]")
            
            if result.get('reasoning', {}).get('is_complex'):
                print(f"[思维链: 已分析 {len(result['reasoning']['sub_questions'])} 个子问题]")
            
            # 更新历史
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": result['response']})
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            logger.error(f"处理失败: {e}")
            print(f"错误: {e}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="多模态客服智能体")
    parser.add_argument(
        "mode",
        nargs="?",
        default="api",
        choices=["api", "build", "interactive", "test"],
        help="运行模式: api(默认), build(构建知识库), interactive(交互测试), test(单元测试)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api_server()
    elif args.mode == "build":
        run_knowledge_builder()
    elif args.mode == "interactive":
        run_interactive_mode()
    elif args.mode == "test":
        # 运行简单的测试
        from src.modules import get_multimodal_understanding
        multimodal = get_multimodal_understanding()
        multimodal.initialize()
        
        result = multimodal.understand(
            text="我的电钻指示灯闪烁代表什么含义？"
        )
        print(f"测试结果: {result}")


if __name__ == "__main__":
    main()
