import logging
from typing import Dict
from app.agent.state import ConsultationState

logger = logging.getLogger(__name__)


class ConsultationNodes:
    def __init__(self, llm_service, rag_retriever):
        self.llm = llm_service
        self.rag = rag_retriever

    async def collect_symptoms(self, state: ConsultationState) -> Dict:
        user_msg = state["user_message"]

        # 构造完整上下文，包含之前累积的信息
        context_parts = []
        if state["collected_symptoms"]:
            context_parts.append(
                f"当前已知症状：{', '.join(state['collected_symptoms'])}。"
            )
        if state["medical_history"]:
            history_str = ", ".join(state["medical_history"])
            context_parts.append(f"当前已知病史：{history_str}。")
        context_parts.append(f"用户最新描述：{user_msg}")
        context = "\n".join(context_parts)

        # 调试日志：查看传给 LLM 的上下文
        logger.debug("context sent to LLM: %s", context)

        symptom_res = await self.llm.extract_symptom(context)

        # 调试日志：查看 LLM 返回的结果
        logger.debug("symptom_res: %s", symptom_res)

        state["collected_symptoms"].extend(symptom_res.get("symptoms", []))

        # 病史为 JSON 数组，去重合并
        existing = state.get("medical_history", [])
        if isinstance(existing, str):
            existing = [existing] if existing else []
        new_entries = symptom_res.get("history", [])
        if isinstance(new_entries, str):
            new_entries = [new_entries] if new_entries else []
        for entry in new_entries:
            if entry and entry not in existing:
                existing.append(entry)
        state["medical_history"] = existing

        # 新的追问策略：症状不足 3 个且无病史时继续追问
        if len(state["collected_symptoms"]) < 3 and not state["medical_history"]:
            symptoms_str = ', '.join(state["collected_symptoms"]) if state["collected_symptoms"] else "无"
            state["report"] = f"您已提到：{symptoms_str}。请再详细描述一下，比如疼痛部位、持续时间，或是否有其他症状？"
            state["next_action"] = "ask"
            state["is_complete"] = False
        else:
            state["next_action"] = "continue"

        return state

    async def retrieve_knowledge(self, state: ConsultationState) -> Dict:
        """节点2：RAG混合检索医学知识库"""
        query = f"{state['user_message']} {state['medical_history']}"
        rag_docs = await self.rag.retrieve(query, top_k=5)
        rag_text = "\n".join([doc["content"] for doc in rag_docs])
        state["rag_context"] = rag_text
        return state

    async def analyze_symptoms(self, state: ConsultationState) -> Dict:
        """节点3：症状分析、判断紧急程度、推荐科室"""
        analysis = await self.llm.analyze_diagnosis(
            symptoms=state["collected_symptoms"],
            history=state["medical_history"],
            rag_context=state["rag_context"],
        )
        state["analysis_result"] = analysis["content"]
        state["urgency_level"] = analysis["urgency_level"]
        state["recommended_department"] = analysis["department"]
        return state

    async def generate_report(self, state: ConsultationState) -> Dict:
        """节点4：生成最终问诊报告"""
        report_text = await self.llm.generate_report(
            analysis=state["analysis_result"],
            urgency=state["urgency_level"],
            dept=state["recommended_department"],
        )
        state["report"] = report_text
        state["is_complete"] = True
        return state