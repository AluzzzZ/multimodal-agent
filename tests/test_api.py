"""
API测试 - 赛题接口规范

测试用例覆盖:
1. 认证机制 (Bearer Token)
2. 核心chat接口 (question/images格式)
3. 会话管理
4. 知识库操作
5. 错误处理
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import base64
import io
from PIL import Image

# 使用mock避免初始化实际模块
with patch.dict('os.environ', {'API_TOKEN': 'sk_customer_20260304'}):
    from src.api import app

# 测试用Token
TEST_TOKEN = "sk_customer_20260304"
TEST_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}


def _make_valid_png_data_url() -> str:
    image = Image.new("RGB", (1, 1), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


VALID_PNG_DATA_URL = _make_valid_png_data_url()

client = TestClient(app)


class TestAuthentication:
    """认证机制测试"""

    def test_missing_auth_header(self):
        """测试缺少认证头"""
        response = client.post("/chat", json={
            "question": "测试问题"
        })
        assert response.status_code == 401
        assert "Authorization" in response.json()["detail"]

    def test_invalid_auth_format(self):
        """测试无效认证格式"""
        response = client.post(
            "/chat",
            json={"question": "测试问题"},
            headers={"Authorization": "InvalidFormat token"}
        )
        assert response.status_code == 401

    def test_invalid_token(self):
        """测试无效Token"""
        response = client.post(
            "/chat",
            json={"question": "测试问题"},
            headers={"Authorization": "Bearer wrong_token"}
        )
        assert response.status_code == 401


class TestHealthAPI:
    """健康检查API测试"""

    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["msg"] == "success"
        assert "data" in data


class TestChatAPI:
    """对话API测试 - 赛题接口规范"""

    def test_chat_with_text_only(self):
        """测试纯文本对话 - 符合赛题格式"""
        with patch('src.api.get_response_generator') as mock_gen:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {
                "response": "测试回答内容",
                "images": [],
                "sources": [],
                "reasoning": None,
                "confidence": 0.9
            }
            mock_gen.return_value = mock_instance

            with patch('src.api.get_conversation_manager') as mock_cm:
                mock_cm_instance = MagicMock()
                mock_cm_instance.create_session.return_value = "test_session_123"
                mock_cm_instance.get_conversation_history.return_value = []
                mock_cm.return_value = mock_cm_instance

                response = client.post(
                    "/chat",
                    json={"question": "我的电钻指示灯闪烁是什么意思？"},
                    headers=TEST_HEADERS
                )

                assert response.status_code == 200
                data = response.json()
                # 验证赛题响应格式
                assert data["code"] == 0
                assert data["msg"] == "success"
                assert "data" in data
                assert "answer" in data["data"]
                assert "session_id" in data["data"]
                assert "timestamp" in data["data"]

    def test_chat_with_images(self):
        """测试带图片的对话"""
        with patch('src.api.get_response_generator') as mock_gen:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {
                "response": "根据您上传的图片，这是电池充电指示灯。",
                "images": ["drill0_04"],
                "sources": [],
                "reasoning": None,
                "confidence": 0.95
            }
            mock_gen.return_value = mock_instance

            with patch('src.api.get_conversation_manager') as mock_cm:
                mock_cm_instance = MagicMock()
                mock_cm_instance.create_session.return_value = "test_session_456"
                mock_cm_instance.get_conversation_history.return_value = []
                mock_cm.return_value = mock_cm_instance

                response = client.post(
                    "/chat",
                    json={
                        "question": "这张图片显示的是什么？",
                        "images": [VALID_PNG_DATA_URL]
                    },
                    headers=TEST_HEADERS
                )

                assert response.status_code == 200
                data = response.json()
                assert data["code"] == 0

    def test_chat_with_stream_flag_falls_back_to_sync(self):
        """测试 stream=true 时当前版本仍按同步完整响应返回"""
        with patch('src.api.get_response_generator') as mock_gen:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {
                "response": "这是同步完整响应。",
                "images": [],
                "sources": [],
                "reasoning": None,
                "confidence": 0.9
            }
            mock_gen.return_value = mock_instance

            with patch('src.api.get_conversation_manager') as mock_cm:
                mock_cm_instance = MagicMock()
                mock_cm_instance.create_session.return_value = "stream_session_001"
                mock_cm_instance.get_conversation_history.return_value = []
                mock_cm.return_value = mock_cm_instance

                response = client.post(
                    "/chat",
                    json={
                        "question": "测试流式字段",
                        "stream": True
                    },
                    headers=TEST_HEADERS
                )

                assert response.status_code == 200
                data = response.json()
                assert data["code"] == 0
                assert data["msg"] == "success"
                assert data["data"]["answer"] == "这是同步完整响应。"

    def test_chat_with_session_id(self):
        """测试带session_id的多轮对话"""
        with patch('src.api.get_response_generator') as mock_gen:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {
                "response": "这是第二回合的回复。",
                "images": [],
                "sources": [],
                "reasoning": None,
                "confidence": 0.85
            }
            mock_gen.return_value = mock_instance

            with patch('src.api.get_conversation_manager') as mock_cm:
                mock_cm_instance = MagicMock()
                mock_cm_instance.get_conversation_history.return_value = [
                    {"role": "user", "content": "第一回合的问题"}
                ]
                mock_cm.return_value = mock_cm_instance

                response = client.post(
                    "/chat",
                    json={
                        "question": "第二回合的问题",
                        "session_id": "existing_session_789"
                    },
                    headers=TEST_HEADERS
                )

                assert response.status_code == 200
                data = response.json()
                # 应该使用传入的session_id
                assert data["data"]["session_id"] == "existing_session_789"

    def test_chat_empty_question(self):
        """测试空问题"""
        response = client.post(
            "/chat",
            json={"question": ""},
            headers=TEST_HEADERS
        )
        assert response.status_code == 422  # Pydantic验证失败

    def test_chat_missing_question(self):
        """测试缺少question字段"""
        response = client.post(
            "/chat",
            json={"session_id": "test"},
            headers=TEST_HEADERS
        )
        assert response.status_code == 422

    def test_chat_too_many_images(self):
        """测试超过3张图片"""
        response = client.post(
            "/chat",
            json={
                "question": "测试问题",
                "images": ["img1", "img2", "img3", "img4"]
            },
            headers=TEST_HEADERS
        )
        assert response.status_code == 422

    def test_chat_invalid_image_prefix(self):
        """测试图片未携带规定的 data:image 前缀"""
        response = client.post(
            "/chat",
            json={
                "question": "测试图片格式",
                "images": ["iVBORw0KGgo="]
            },
            headers=TEST_HEADERS
        )
        assert response.status_code == 422
        assert "data:image" in response.json()["detail"]

    def test_chat_image_too_large(self):
        """测试单张图片超过 5MB 限制"""
        oversized_bytes = b"\x00" * (5 * 1024 * 1024 + 1)
        oversized = base64.b64encode(oversized_bytes).decode("utf-8")
        response = client.post(
            "/chat",
            json={
                "question": "测试超大图片",
                "images": [f"data:image/png;base64,{oversized}"]
            },
            headers=TEST_HEADERS
        )
        assert response.status_code == 422
        assert "超过限制" in response.json()["detail"]


class TestSessionAPI:
    """会话管理API测试"""

    def test_create_session(self):
        """测试创建会话"""
        with patch('src.api.get_conversation_manager') as mock_cm:
            mock_cm_instance = MagicMock()
            mock_cm_instance.create_session.return_value = "new_session_id"
            mock_cm.return_value = mock_cm_instance

            response = client.post(
                "/session/create",
                headers=TEST_HEADERS
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            assert data["data"]["session_id"] == "new_session_id"

    def test_get_nonexistent_session(self):
        """测试获取不存在的会话"""
        with patch('src.api.get_conversation_manager') as mock_cm:
            mock_cm_instance = MagicMock()
            mock_cm_instance.get_session.return_value = None
            mock_cm.return_value = mock_cm_instance

            response = client.get(
                "/session/nonexistent-id",
                headers=TEST_HEADERS
            )
            assert response.status_code == 404

    def test_delete_session(self):
        """测试删除会话"""
        with patch('src.api.get_conversation_manager') as mock_cm:
            mock_cm_instance = MagicMock()
            mock_cm_instance.clear_session.return_value = True
            mock_cm.return_value = mock_cm_instance

            response = client.delete(
                "/session/test-session-id",
                headers=TEST_HEADERS
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0


class TestKnowledgeAPI:
    """知识库API测试"""

    def test_add_documents(self):
        """测试添加文档"""
        with patch('src.api.get_rag_engine') as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance

            response = client.post(
                "/knowledge/add",
                json={
                    "documents": [
                        {
                            "content": "DCB107电池组指示灯说明",
                            "doc_id": "drill_battery_001",
                            "metadata": {"category": "battery"}
                        }
                    ],
                    "doc_type": "text"
                },
                headers=TEST_HEADERS
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            assert data["data"]["added_count"] == 1

    def test_retrieve_knowledge(self):
        """测试检索知识"""
        with patch('src.api.get_rag_engine') as mock_engine:
            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = [
                {
                    "content": "测试内容",
                    "doc_id": "test_001",
                    "relevance_score": 0.95,
                    "image_ids": ["img1"]
                }
            ]
            mock_engine.return_value = mock_instance

            response = client.post(
                "/knowledge/retrieve",
                params={"query": "电钻指示灯", "top_k": 5},
                headers=TEST_HEADERS
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            assert len(data["data"]["results"]) == 1

    def test_build_index(self):
        """测试构建索引"""
        with patch('src.api.get_rag_engine') as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance

            response = client.post(
                "/knowledge/build",
                headers=TEST_HEADERS
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0


class TestRootAPI:
    """根路径API测试"""

    def test_root(self):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "name" in data["data"]
        assert "version" in data["data"]
        assert data["data"]["docs"] == "/docs"


class TestResponseFormat:
    """响应格式测试 - 确保符合赛题规范"""

    def test_standard_response_structure(self):
        """测试标准响应结构"""
        with patch('src.api.get_response_generator') as mock_gen:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {
                "response": "测试回答",
                "images": [],
                "sources": [],
                "confidence": 0.9
            }
            mock_gen.return_value = mock_instance

            with patch('src.api.get_conversation_manager') as mock_cm:
                mock_cm_instance = MagicMock()
                mock_cm_instance.create_session.return_value = "session_test"
                mock_cm_instance.get_conversation_history.return_value = []
                mock_cm.return_value = mock_cm_instance

                response = client.post(
                    "/chat",
                    json={"question": "测试"},
                    headers=TEST_HEADERS
                )

                data = response.json()

                # 验证标准响应格式
                assert "code" in data
                assert "msg" in data
                assert "data" in data
                assert data["code"] == 0
                assert data["msg"] == "success"

                # 验证data内部结构
                assert "answer" in data["data"]
                assert "session_id" in data["data"]
                assert "timestamp" in data["data"]

                # 验证时间戳格式(Unix时间戳)
                assert isinstance(data["data"]["timestamp"], int)
                assert data["data"]["timestamp"] > 0
