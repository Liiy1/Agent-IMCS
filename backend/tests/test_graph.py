"""咨询工作流图单元测试

测试条件边逻辑和节点行为，使用 mock 的 LLM 和 RAG 服务。
"""

import sys
sys.path.insert(0, "backend")

import pytest
from app.agent.graph import ConsultationGraph
from app.agent.nodes import ConsultationNodes


# ── should_continue ────────────────────────────────────────

def test_should_continue_to_retrieve():
    """症状足够 → 应返回 continue"""
    graph = ConsultationGraph.__new__(ConsultationGraph)
    state = {
        "next_action": "continue",
        "collected_symptoms": ["头痛", "发烧", "咳嗽"],
        "medical_history": [],
    }
    assert graph.should_continue(state) == "continue"


def test_should_continue_to_ask():
    """症状不足且无病史 → 应要求追问"""
    graph = ConsultationGraph.__new__(ConsultationGraph)
    state = {
        "next_action": "ask",
        "collected_symptoms": ["头痛"],
        "medical_history": [],
    }
    assert graph.should_continue(state) == "ask"


# ── collect_symptoms ───────────────────────────────────────

@pytest.mark.asyncio
async def test_collect_extends_symptoms(sample_state, mock_llm_service, mock_rag_retriever):
    """collect 节点应将 LLM 提取的症状合并到 state"""
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.collect_symptoms(dict(sample_state))
    assert "头痛" in result["collected_symptoms"]
    assert "发烧" in result["collected_symptoms"]


@pytest.mark.asyncio
async def test_collect_enough_symptoms_continues(sample_state, mock_llm_service, mock_rag_retriever):
    """≥3 症状 → next_action 为 continue"""
    state = dict(sample_state)
    state["collected_symptoms"] = ["头痛", "发烧", "咳嗽"]
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.collect_symptoms(state)
    assert result["next_action"] == "continue"


@pytest.mark.asyncio
async def test_collect_few_symptoms_asks(sample_state, mock_llm_service, mock_rag_retriever):
    """<3 症状且无病史 → next_action 为 ask"""
    state = dict(sample_state)
    state["collected_symptoms"] = ["头痛"]
    state["medical_history"] = []
    # mock 返回空症状
    mock_llm_service.extract_symptom.return_value = {
        "symptoms": [], "history": [], "next_ask": ""
    }
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.collect_symptoms(state)
    assert result["next_action"] == "ask"
    assert result["is_complete"] is False
    assert "请再详细描述" in result["report"]


# ── retrieve_knowledge ─────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_fills_rag_context(sample_state, mock_llm_service, mock_rag_retriever):
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.retrieve_knowledge(dict(sample_state))
    assert "头痛" in result["rag_context"]
    assert "发热" in result["rag_context"]


# ── analyze_symptoms ───────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_fills_results(sample_state, mock_llm_service, mock_rag_retriever):
    state = dict(sample_state)
    state["rag_context"] = "头痛分为原发性头痛和继发性头痛。"
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.analyze_symptoms(state)
    assert result["analysis_result"] != ""
    assert result["urgency_level"] == "medium"
    assert result["recommended_department"] == "内科"


# ── generate_report ────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_report_marks_complete(sample_state, mock_llm_service, mock_rag_retriever):
    state = dict(sample_state)
    state["analysis_result"] = "上呼吸道感染"
    state["urgency_level"] = "medium"
    state["recommended_department"] = "内科"
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.generate_report(state)
    assert result["is_complete"] is True
    assert "问诊报告" in result["report"]


# ── 完整工作流 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_workflow(sample_state, mock_llm_service, mock_rag_retriever):
    """模拟完整 4 节点流程"""
    state = dict(sample_state)
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)

    state = await nodes.collect_symptoms(state)
    assert state["next_action"] in ("ask", "continue")

    # 跳过 'ask' 场景直接测试完整流 — 设足够症状确保继续
    state["collected_symptoms"] = ["头痛", "发烧", "咳嗽"]
    state["next_action"] = "continue"

    state = await nodes.retrieve_knowledge(state)
    assert state["rag_context"] != ""

    state = await nodes.analyze_symptoms(state)
    assert state["analysis_result"] != ""
    assert state["urgency_level"] != "low"  # mock 返回 medium

    state = await nodes.generate_report(state)
    assert state["is_complete"] is True
