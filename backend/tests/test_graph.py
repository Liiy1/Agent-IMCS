"""咨询工作流图单元测试

测试条件边逻辑和节点行为（含反思机制），使用 mock 的 LLM 和 RAG 服务。
"""

import sys
sys.path.insert(0, "backend")

import pytest
from unittest.mock import ANY
from app.agent.graph import ConsultationGraph
from app.agent.nodes import ConsultationNodes


# ── should_continue (原条件边) ────────────────────────────

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
    """症状不足且无病史 → 应返回 ask"""
    graph = ConsultationGraph.__new__(ConsultationGraph)
    state = {
        "next_action": "ask",
        "collected_symptoms": ["头痛"],
        "medical_history": [],
    }
    assert graph.should_continue(state) == "ask"


# ── should_reflect (新条件边) ─────────────────────────────

def test_should_reflect_continue():
    """反思通过 → finalize"""
    graph = ConsultationGraph.__new__(ConsultationGraph)
    assert graph.should_reflect({"next_action": "continue"}) == "continue"


def test_should_reflect_refine():
    """反思不通过 → refine"""
    graph = ConsultationGraph.__new__(ConsultationGraph)
    assert graph.should_reflect({"next_action": "refine"}) == "refine"


# ── collect_symptoms ──────────────────────────────────────

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
    """<3 症状且无病史 → next_action 为 ask，并使用 LLM 生成追问"""
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
    # 验证调用了 LLM 追问生成，并使用了返回的话术
    mock_llm_service.generate_followup.assert_called_once_with(
        symptoms=["头痛"],
        history=[],
        user_message=state["user_message"],
    )
    assert "头痛" in result["report"]
    assert result["report"] != ""


@pytest.mark.asyncio
async def test_collect_few_symptoms_fallback_template(
    sample_state, mock_llm_service, mock_rag_retriever,
):
    """LLM 追问生成失败时 → 回退到默认模板"""
    state = dict(sample_state)
    state["collected_symptoms"] = ["头痛"]
    state["medical_history"] = []
    mock_llm_service.extract_symptom.return_value = {
        "symptoms": [], "history": [], "next_ask": ""
    }
    # simulate LLM failure
    mock_llm_service.generate_followup.side_effect = Exception("API 错误")
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
    # 初版分析轮次应为 1
    assert result["reflection_round"] == 1


@pytest.mark.asyncio
async def test_analyze_includes_feedback_when_refining(sample_state, mock_llm_service, mock_rag_retriever):
    """精炼轮次中，analyze_diagnosis 应收到上一版分析和反思反馈"""
    state = dict(sample_state)
    state["rag_context"] = "头痛分类知识"
    state["analysis_result"] = "旧版分析：普通感冒"
    state["reflection_feedback"] = "紧急程度不匹配，应改为 medium"
    state["reflection_round"] = 1  # 精炼轮次

    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.analyze_symptoms(state)

    # 验证 analyze_diagnosis 被以精炼模式调用（含前两版分析和反馈参数）
    mock_llm_service.analyze_diagnosis.assert_called_with(
        symptoms=state["collected_symptoms"],
        history=state["medical_history"],
        rag_context=state["rag_context"],
        previous_analysis="旧版分析：普通感冒",
        reflection_feedback="紧急程度不匹配，应改为 medium",
    )
    assert result["analysis_result"] != ""
    assert result["reflection_round"] == 2  # 轮次递增


# ── reflect_analysis (新节点) ─────────────────────────────

@pytest.mark.asyncio
async def test_reflect_passes_good_analysis(sample_state, mock_llm_service, mock_rag_retriever):
    """高质量分析 → reflect 通过，next_action=continue"""
    state = dict(sample_state)
    state["analysis_result"] = "高质量诊断分析：上呼吸道感染"
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.reflect_analysis(state)
    assert result["next_action"] == "continue"
    assert result["reflection_passed"] is True
    assert result["reflection_score"] == 4


@pytest.mark.asyncio
async def test_reflect_fails_poor_analysis(sample_state, mock_llm_service_reflect_fail, mock_rag_retriever):
    """低质量分析 → reflect 不通过，next_action=refine"""
    state = dict(sample_state)
    state["analysis_result"] = "低质量分析"
    nodes = ConsultationNodes(mock_llm_service_reflect_fail, mock_rag_retriever)
    result = await nodes.reflect_analysis(state)
    assert result["next_action"] == "refine"
    assert result["reflection_passed"] is False
    assert result["reflection_score"] == 2


@pytest.mark.asyncio
async def test_reflect_empty_analysis_triggers_refine(sample_state, mock_llm_service, mock_rag_retriever):
    """analysis_result 为空 → 直接 trigger refine，不调用 LLM"""
    state = dict(sample_state)
    state["analysis_result"] = ""
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.reflect_analysis(state)
    assert result["next_action"] == "refine"
    # 确保没有调用 LLM（空分析跳过反思）
    mock_llm_service.reflect_analysis.assert_not_called()


@pytest.mark.asyncio
async def test_reflect_max_rounds_force_continue(sample_state, mock_llm_service_reflect_fail, mock_rag_retriever):
    """达到最大轮次 → 即使不通过也 continue"""
    state = dict(sample_state)
    state["analysis_result"] = "仍不达标的分析"
    state["reflection_round"] = 2  # 已达上限 (MAX_REFLECTION_ROUNDS=2)
    nodes = ConsultationNodes(mock_llm_service_reflect_fail, mock_rag_retriever)
    result = await nodes.reflect_analysis(state)
    assert result["next_action"] == "continue"
    assert result["reflection_passed"] is False


# ── generate_report ───────────────────────────────────────

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


@pytest.mark.asyncio
async def test_generate_report_no_disclaimer_when_passed(sample_state, mock_llm_service, mock_rag_retriever):
    """反思通过后 → 报告不应含'建议人工复核'声明"""
    state = dict(sample_state)
    state["analysis_result"] = "诊断分析"
    state["urgency_level"] = "medium"
    state["recommended_department"] = "内科"
    state["reflection_round"] = 1
    state["reflection_score"] = 4  # 通过
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.generate_report(state)
    assert "建议人工复核" not in result["report"]


@pytest.mark.asyncio
async def test_generate_report_has_disclaimer_when_failed(sample_state, mock_llm_service, mock_rag_retriever):
    """轮次用尽且评分不达标 → 报告开头添加'建议人工复核'声明"""
    state = dict(sample_state)
    state["analysis_result"] = "诊断分析"
    state["urgency_level"] = "medium"
    state["recommended_department"] = "内科"
    state["reflection_round"] = 2  # 用尽
    state["reflection_score"] = 2  # 不达标
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)
    result = await nodes.generate_report(state)
    assert result["is_complete"] is True
    assert "建议人工复核" in result["report"]
    # 声明应出现在报告开头
    assert result["report"].startswith("> ⚠️ **建议人工复核**")


# ── 完整工作流（含反思） ─────────────────────────────────

@pytest.mark.asyncio
async def test_full_workflow(sample_state, mock_llm_service, mock_rag_retriever):
    """模拟完整流程（collect → retrieve → analyze → reflect → generate_report）"""
    state = dict(sample_state)
    nodes = ConsultationNodes(mock_llm_service, mock_rag_retriever)

    # 节点1: collect
    state.update(await nodes.collect_symptoms(state))
    assert state["next_action"] in ("ask", "continue")

    # 跳过 'ask' 场景直接测试完整流
    state["collected_symptoms"] = ["头痛", "发烧", "咳嗽"]
    state["next_action"] = "continue"

    # 节点2: retrieve
    state.update(await nodes.retrieve_knowledge(state))
    assert state["rag_context"] != ""

    # 节点3: analyze
    state.update(await nodes.analyze_symptoms(state))
    assert state["analysis_result"] != ""
    assert state["urgency_level"] != "low"
    assert state["reflection_round"] == 1

    # 节点4: reflect
    state.update(await nodes.reflect_analysis(state))
    assert state["next_action"] in ("continue", "refine")
    # mock 返回高分，所以应为 continue
    assert state["next_action"] == "continue"
    assert state["reflection_passed"] is True
    assert state["reflection_score"] >= 3

    # 节点5: generate_report
    state.update(await nodes.generate_report(state))
    assert state["is_complete"] is True
    assert "问诊报告" in state["report"]


@pytest.mark.asyncio
async def test_full_workflow_with_refinement(
    sample_state, mock_llm_service_reflect_fail, mock_rag_retriever,
):
    """完整流程 — 模拟反思不通过 → 精炼 → 再反思 → 通过"""
    state = dict(sample_state)
    nodes = ConsultationNodes(mock_llm_service_reflect_fail, mock_rag_retriever)

    # 设足够症状
    state["collected_symptoms"] = ["头痛", "发烧", "咳嗽", "乏力"]
    state["rag_context"] = "头痛分类知识"

    # 初次分析
    state.update(await nodes.analyze_symptoms(state))
    assert state["analysis_result"] != ""

    # 反思 — 不通过
    state.update(await nodes.reflect_analysis(state))
    assert state["next_action"] == "refine"
    assert state["reflection_passed"] is False
    assert state["reflection_feedback"] != ""

    # 精炼分析（应传入上一版分析和反馈）
    state.update(await nodes.analyze_symptoms(state))
    assert state["reflection_round"] == 2
    # 验证 LLM 在精炼轮次收到了反馈参数
    mock_llm_service_reflect_fail.analyze_diagnosis.assert_called_with(
        symptoms=ANY,
        history=ANY,
        rag_context=ANY,
        previous_analysis=ANY,
        reflection_feedback=ANY,
    )

    # 再反思 — 仍然不通过但轮次已达上限，强制继续
    state.update(await nodes.reflect_analysis(state))
    assert state["next_action"] == "continue"

    # 生成报告 → 应含人工复核声明
    state.update(await nodes.generate_report(state))
    assert state["is_complete"] is True
    assert "建议人工复核" in state["report"]
