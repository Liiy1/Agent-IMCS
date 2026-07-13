"""API 集成测试

使用 FastAPI TestClient 测试 HTTP 端点。
数据库和 Redis 相关测试需要真实服务运行，默认跳过；
核心问诊逻辑通过 mock 覆盖。
"""

import sys
sys.path.insert(0, "backend")

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


# ── /health ────────────────────────────────────────────────

def test_health_endpoint():
    """健康检查端点应返回 200"""
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AI智能问诊系统"


# ── /api/v1/consultation/consult ──────────────────────────

@pytest.mark.skip(reason="需要 MySQL + Redis 运行")
def test_consult_missing_token():
    """缺少 user_token 应返回 400"""
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/consultation/consult", json={
            "message": "头痛",
        })
    assert resp.status_code == 400


@pytest.mark.skip(reason="需要 MySQL + Redis 运行")
def test_consult_empty_message():
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/consultation/consult", json={
            "message": "",
            "user_token": "test",
        })
    assert resp.status_code == 400


# ── /api/v1/report/export ─────────────────────────────────

@pytest.mark.skip(reason="需要 MySQL + Redis 运行")
def test_export_missing_session():
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/report/export", params={
            "session_id": "nonexistent",
            "user_token": "test",
        })
    assert resp.status_code == 404
