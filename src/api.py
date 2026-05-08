"""
多模态客服智能体 - API接口
符合赛题评审标准的RESTful API服务
接口定义: POST /chat (核心端点), Bearer Token认证
"""

import time
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import uvicorn
from loguru import logger

from config import settings
from .modules import (
    get_multimodal_understanding,
    get_rag_engine,
    get_conversation_manager,
    get_response_generator
)


# ============ 配置扩展 ============

class APIConfig(BaseSettings):
    api_token: str = "sk_customer_20260304"
    max_images: int = 3
    max_image_size: int = 5 * 1024 * 1024  # 5MB

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


api_config = APIConfig()


# ============ Pydantic请求/响应模型 ============

class ChatRequest(BaseModel):
    """赛题标准请求模型 - 极简字段设计"""
    question: str = Field(..., description="核心输入：用户的客服问题字符串，长度>=1")
    images: List[str] = Field(default_factory=list, description="Base64格式图片列表，0-3张，每张<=5MB")
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    stream: bool = Field(False, description="是否流式响应")

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question长度必须>=1")
        return v.strip()

    @field_validator('images')
    @classmethod
    def images_limit(cls, v: List[str]) -> List[str]:
        if len(v) > api_config.max_images:
            raise ValueError(f"images最多{api_config.max_images}张")
        return v


class StandardResponse(BaseModel):
    """赛题标准响应外层"""
    code: int = Field(..., description="状态码，0表示成功")
    msg: str = Field(..., description="状态信息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")


class ChatResponseData(BaseModel):
    """chat接口响应数据体"""
    answer: str = Field(..., description="智能体返回的客服答案")
    session_id: str = Field(..., description="会话ID，用于多轮对话续接")
    timestamp: int = Field(..., description="响应时间戳（秒）")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    components: Dict[str, str]


class KnowledgeAddRequest(BaseModel):
    """知识库增量写入请求"""
    documents: List[Dict[str, Any]] = Field(..., min_length=1, description="待写入的文档列表")
    doc_type: str = Field("text", description="文档类型: text 或 image")


# ============ 认证依赖 ============

async def verify_token(
    authorization: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
    x_client_type: Optional[str] = Header(None, alias="X-Client-Type")
) -> Dict[str, str]:
    """
    验证Bearer Token认证
    赛题要求: Authorization: Bearer {KAFU_API_TOKEN}
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Expected: Bearer {token}",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = authorization[7:]

    if token != api_config.api_token:
        logger.warning(f"Invalid token attempt: {token[:8]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return {
        "request_id": x_request_id or str(uuid.uuid4()),
        "client_type": x_client_type or "unknown"
    }


# ============ FastAPI应用 ============

app = FastAPI(
    title="多模态客服智能体API",
    version="1.0.0",
    description="多模态客服智能体API，支持文本对话、图片理解和RAG知识检索 | 赛题接口",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 初始化 ============

@app.on_event("startup")
async def startup_event():
    """应用启动时按需初始化模块，降低启动内存峰值。"""
    logger.info("正在初始化多模态客服智能体...")

    try:
        conversation_manager = get_conversation_manager()
        conversation_manager.initialize()
        logger.info("对话管理器初始化完成")

        if settings.eager_init_rag:
            rag_engine = get_rag_engine()
            rag_engine.initialize()
            logger.info("RAG引擎初始化完成")

        if settings.eager_init_multimodal:
            multimodal = get_multimodal_understanding()
            multimodal.initialize()
            logger.info("多模态理解模块初始化完成")

        if settings.eager_init_response_generator:
            response_generator = get_response_generator()
            response_generator.initialize()
            logger.info("回答生成器初始化完成")

        logger.info("应用启动完成，重型模块将按需加载")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise


# ============ 核心API端点 ============

@app.post("/chat", response_model=StandardResponse, tags=["对话"])
async def chat(
    request: ChatRequest,
    auth_info: Dict[str, str] = Depends(verify_token)
) -> StandardResponse:
    """
    多模态对话交互核心端点

    赛题接口规范:
    - 请求方式: POST
    - 认证: Bearer Token (必填)
    - 请求体: { "question": string, "images": string[], "session_id": string, "stream": boolean }
    - 响应: { "code": 0, "msg": "success", "data": { "answer": string, "session_id": string, "timestamp": int } }
    """
    request_id = auth_info["request_id"]
    logger.info(f"[{request_id}] 收到chat请求, question长度: {len(request.question)}")

    try:
        conversation_manager = get_conversation_manager()
        response_generator = get_response_generator()

        # 获取或创建会话
        session_id = request.session_id
        if not session_id:
            session_id = conversation_manager.create_session()

        # 添加用户消息到会话历史
        conversation_manager.add_message(
            session_id=session_id,
            role="user",
            content=request.question,
            images=request.images if request.images else None
        )

        # 获取对话历史上下文
        history = conversation_manager.get_conversation_history(session_id, limit=6)

        # 生成回答
        result = response_generator.generate(
            query=request.question,
            images=request.images if request.images else None,
            conversation_history=history
        )

        # 添加助手回复到会话历史
        conversation_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=result["response"]
        )

        route_summary = ", ".join(
            [f"{item.get('question', '')}->{item.get('route', '')}" for item in result.get("routes", [])[:4]]
        )
        logger.info(
            f"[{request_id}] 生成回答完成, 置信度: {result.get('confidence', 0):.2f}, 路由: {route_summary or 'n/a'}"
        )

        # 构建赛题标准响应格式
        return StandardResponse(
            code=0,
            msg="success",
            data={
                "answer": result["response"],
                "session_id": session_id,
                "timestamp": int(time.time())
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] 处理请求失败: {e}")
        return StandardResponse(
            code=500,
            msg=f"Internal error: {str(e)}",
            data=None
        )


# ============ 会话管理API ============

@app.post("/session/create", tags=["会话管理"])
async def create_session(
    user_id: Optional[str] = None,
    auth_info: Dict[str, str] = Depends(verify_token)
) -> StandardResponse:
    """创建新会话"""
    conversation_manager = get_conversation_manager()
    session_id = conversation_manager.create_session(user_id)
    return StandardResponse(
        code=0,
        msg="success",
        data={"session_id": session_id}
    )


@app.get("/session/{session_id}", tags=["会话管理"])
async def get_session(
    session_id: str,
    auth_info: Dict[str, str] = Depends(verify_token)
) -> StandardResponse:
    """获取会话信息"""
    conversation_manager = get_conversation_manager()
    session = conversation_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    return StandardResponse(
        code=0,
        msg="success",
        data={
            "session_id": session.session_id,
            "messages": session.get_history(),
            "created_at": session.created_at,
            "last_active": session.last_active
        }
    )


@app.delete("/session/{session_id}", tags=["会话管理"])
async def delete_session(
    session_id: str,
    auth_info: Dict[str, str] = Depends(verify_token)
) -> StandardResponse:
    """删除会话"""
    conversation_manager = get_conversation_manager()
    success = conversation_manager.clear_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")

    return StandardResponse(
        code=0,
        msg="success",
        data={"status": "deleted", "session_id": session_id}
    )


# ============ 知识库管理API ============

@app.post("/knowledge/add", tags=["知识库"])
async def add_documents(
    request: KnowledgeAddRequest,
    auth_info: Dict[str, str] = Depends(verify_token)
) -> StandardResponse:
    """
    添加文档到知识库

    documents: 文档列表，每个包含:
        - content: 文档内容
        - doc_id: 文档ID
        - metadata: 元数据（可选）
    doc_type: 文档类型 ("text" 或 "image")
    """
    try:
        rag_engine = get_rag_engine()

        if not rag_engine._initialized:
            rag_engine.initialize()

        rag_engine.add_documents(request.documents, request.doc_type)

        return StandardResponse(
            code=0,
            msg="success",
            data={
                "added_count": len(request.documents),
                "doc_type": request.doc_type
            }
        )
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        return StandardResponse(
            code=500,
            msg=str(e),
            data=None
        )


@app.post("/knowledge/build", tags=["知识库"])
async def build_index(
    auth_info: Dict[str, str] = Depends(verify_token)
) -> StandardResponse:
    """构建/重建知识库索引"""
    try:
        rag_engine = get_rag_engine()
        rag_engine.save_knowledge_base()
        return StandardResponse(
            code=0,
            msg="success",
            data={"message": "知识库索引构建完成"}
        )
    except Exception as e:
        logger.error(f"构建索引失败: {e}")
        return StandardResponse(
            code=500,
            msg=str(e),
            data=None
        )


@app.post("/knowledge/retrieve", tags=["知识库"])
async def retrieve_knowledge(
    query: str,
    top_k: int = 5,
    auth_info: Dict[str, str] = Depends(verify_token)
) -> StandardResponse:
    """检索知识库"""
    try:
        rag_engine = get_rag_engine()

        if not rag_engine._initialized:
            rag_engine.initialize()

        results = rag_engine.retrieve(query, top_k=top_k)

        return StandardResponse(
            code=0,
            msg="success",
            data={
                "query": query,
                "results": results,
                "count": len(results)
            }
        )
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return StandardResponse(
            code=500,
            msg=str(e),
            data=None
        )


# ============ 系统API ============

@app.get("/health", tags=["系统"])
async def health_check() -> StandardResponse:
    """健康检查"""
    rag_engine = get_rag_engine()
    return StandardResponse(
        code=0,
        msg="success",
        data={
            "status": "healthy",
            "version": settings.api_version,
            "components": {
                "api": "running",
                "rag": "ready" if rag_engine._initialized else "initializing",
                "conversation": "ready"
            }
        }
    )


@app.get("/", tags=["系统"])
async def root() -> StandardResponse:
    """根路径"""
    return StandardResponse(
        code=0,
        msg="success",
        data={
            "name": "多模态客服智能体API",
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/health"
        }
    )


# ============ 运行入口 ============

def run_server():
    """运行API服务器"""
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    run_server()
