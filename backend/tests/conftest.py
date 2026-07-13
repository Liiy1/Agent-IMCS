"""共享 fixtures 和 mock 辅助函数"""

from unittest.mock import AsyncMock, MagicMock
import pytest


@pytest.fixture
def mock_llm_service():
    """返回一个全 mock 的 LLMService，所有方法都是 AsyncMock"""
    mock = MagicMock()
    mock.extract_symptom = AsyncMock(return_value={
        "symptoms": ["头痛", "发烧"],
        "history": [],
        "next_ask": "",
    })
    mock.analyze_diagnosis = AsyncMock(return_value={
        "content": "模拟诊断分析：可能是上呼吸道感染",
        "urgency_level": "medium",
        "department": "内科",
    })
    mock.generate_report = AsyncMock(return_value="## 模拟问诊报告\n\n诊断：上呼吸道感染")
    return mock


@pytest.fixture
def mock_rag_retriever():
    """返回一个全 mock 的 MedicalRAGRetriever"""
    mock = MagicMock()
    mock.retrieve = AsyncMock(return_value=[
        {"id": "med_1", "content": "头痛分为原发性头痛和继发性头痛。", "metadata": {}, "score": 0.9},
        {"id": "med_0", "content": "发热是指体温超过37.3°C。", "metadata": {}, "score": 0.8},
    ])
    return mock


@pytest.fixture
def sample_state():
    """一个典型的 ConsultationState 样本"""
    return {
        "session_id": "test-session",
        "user_message": "我头痛三天了，有点发烧",
        "collected_symptoms": ["头痛", "发烧"],
        "medical_history": [],
        "rag_context": "",
        "analysis_result": "",
        "urgency_level": "low",
        "recommended_department": "",
        "report": "",
        "next_action": "",
        "is_complete": False,
        "error": "",
    }
