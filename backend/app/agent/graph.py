from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from app.agent.state import ConsultationState
from app.agent.nodes import ConsultationNodes


class ConsultationGraph:
    def __init__(self, llm_service, rag_retriever):
        self.nodes = ConsultationNodes(llm_service, rag_retriever)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ConsultationState)

        workflow.add_node("collect", self.nodes.collect_symptoms)
        workflow.add_node("retrieve", self.nodes.retrieve_knowledge)
        workflow.add_node("analyze", self.nodes.analyze_symptoms)
        workflow.add_node("generate_report", self.nodes.generate_report)

        workflow.set_entry_point("collect")

        # 条件边：根据 collect 结果决定是追问结束还是继续
        workflow.add_conditional_edges(
            "collect",
            self.should_continue,
            {
                "ask": END,                # 需要追问，直接结束
                "continue": "retrieve"     # 信息足够，进入 RAG 检索
            }
        )

        workflow.add_edge("retrieve", "analyze")
        workflow.add_edge("analyze", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def should_continue(self, state: ConsultationState) -> str:
        """判断是否继续工作流"""
        if state.get("next_action") == "ask":
            return "ask"
        return "continue"

    async def run(
        self,
        session_id: str,
        user_message: str,
        accumulated_state: Optional[Dict[str, Any]] = None,
    ) -> ConsultationState:
        """执行问诊工作流

        Args:
            session_id: 会话 ID
            user_message: 用户输入
            accumulated_state: 前序轮次累积的状态（多轮对话恢复）

        Returns:
            完整的 ConsultationState
        """
        acc = accumulated_state or {}

        initial_state: ConsultationState = {
            "session_id": session_id,
            "user_message": user_message,
            "collected_symptoms": acc.get("collected_symptoms", []),
            "medical_history": acc.get("medical_history", ""),
            "rag_context": "",
            "analysis_result": "",
            "urgency_level": acc.get("urgency_level", "low"),
            "recommended_department": acc.get("recommended_department", ""),
            "report": "",
            "next_action": "",
            "is_complete": False,
            "error": ""
        }

        final_state: ConsultationState = await self.graph.ainvoke(initial_state)
        return final_state