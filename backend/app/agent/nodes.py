import logging
from typing import Dict
from app.agent.state import ConsultationState
from app.config import settings

logger = logging.getLogger(__name__)


# ── 反思统计追踪器 ────────────────────────────────────────

class ReflectionTracker:
    """追踪反思轮次、评分分布和精炼触发率"""
    def __init__(self):
        self.total_reflections = 0
        self.refinements_triggered = 0
        self.scores = []
        self.round_counts = []

    def record_reflection(self, score: int, passed: bool, round_num: int):
        self.total_reflections += 1
        self.scores.append(score)
        self.round_counts.append(round_num)
        if not passed:
            self.refinements_triggered += 1

    def report(self) -> str:
        if not self.total_reflections:
            return "（尚无反思数据）"
        avg_score = sum(self.scores) / len(self.scores)
        trigger_rate = self.refinements_triggered / self.total_reflections * 100
        return (
            f"反思总次数={self.total_reflections}, "
            f"平均评分={avg_score:.1f}, "
            f"精炼触发率={trigger_rate:.1f}%"
        )


reflection_tracker = ReflectionTracker()


class ConsultationNodes:
    def __init__(self, llm_service, rag_retriever):
        self.llm = llm_service
        self.rag = rag_retriever

    async def collect_symptoms(self, state: ConsultationState) -> Dict:
        """节点1：收集症状（返回部分状态更新）"""
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

        logger.debug("context sent to LLM: %s", context)

        symptom_res = await self.llm.extract_symptom(context)

        logger.debug("symptom_res: %s", symptom_res)

        # 合并症状
        new_symptoms = symptom_res.get("symptoms", [])
        collected = list(state["collected_symptoms"])
        for s in new_symptoms:
            if s and s not in collected:
                collected.append(s)

        # 合并病史（去重）
        raw_existing = state.get("medical_history", [])
        if isinstance(raw_existing, str):
            existing = [raw_existing] if raw_existing else []
        else:
            existing = list(raw_existing) if raw_existing else []
        new_entries = symptom_res.get("history", [])
        if isinstance(new_entries, str):
            new_entries = [new_entries] if new_entries else []
        for entry in new_entries:
            if entry and entry not in existing:
                existing.append(entry)

        # 追问策略：症状不足 3 个且无病史时继续追问
        if len(collected) < 3 and not existing:
            symptoms_str = ', '.join(collected) if collected else "无"
            # 优先让 LLM 生成有针对性的追问，失败则用默认模板
            question = None
            try:
                followup = await self.llm.generate_followup(
                    symptoms=collected,
                    history=existing,
                    user_message=state["user_message"],
                )
                question = followup.get("question", "")
            except Exception as e:
                logger.warning("LLM 追问生成失败，使用默认模板: %s", e)

            if not question:
                question = (
                    f"您已提到：{symptoms_str}。"
                    f"请再详细描述一下，比如疼痛部位、持续时间，或是否有其他症状？"
                )

            return {
                "collected_symptoms": collected,
                "medical_history": existing,
                "report": question,
                "next_action": "ask",
                "is_complete": False,
            }
        else:
            return {
                "collected_symptoms": collected,
                "medical_history": existing,
                "next_action": "continue",
            }

    async def retrieve_knowledge(self, state: ConsultationState) -> Dict:
        """节点2：RAG混合检索医学知识库（返回部分状态更新）"""
        query = f"{state['user_message']} {state['medical_history']}"
        rag_docs = await self.rag.retrieve(query, top_k=5)
        rag_text = "\n".join([doc["content"] for doc in rag_docs])
        return {"rag_context": rag_text}

    async def analyze_symptoms(self, state: ConsultationState) -> Dict:
        """节点3：症状分析 — 支持反思循环中的精炼（返回部分状态更新）"""
        feedback = state.get("reflection_feedback", "")
        prev_analysis = state.get("analysis_result", "")

        # 如果是精炼轮次且有反馈，传给 LLM
        if feedback and state.get("reflection_round", 0) > 0:
            logger.info(
                "精炼分析: round=%d, feedback=%.150s",
                state["reflection_round"], feedback,
            )
            analysis = await self.llm.analyze_diagnosis(
                symptoms=state["collected_symptoms"],
                history=state["medical_history"],
                rag_context=state["rag_context"],
                previous_analysis=prev_analysis,
                reflection_feedback=feedback,
            )
        else:
            analysis = await self.llm.analyze_diagnosis(
                symptoms=state["collected_symptoms"],
                history=state["medical_history"],
                rag_context=state["rag_context"],
            )

        return {
            "analysis_result": analysis["content"],
            "urgency_level": analysis["urgency_level"],
            "recommended_department": analysis["department"],
            "reflection_round": state.get("reflection_round", 0) + 1,
        }

    async def reflect_analysis(self, state: ConsultationState) -> Dict:
        """节点4：反思诊断分析质量，决定是否需要精炼（返回部分状态更新）"""
        # 校验 analysis_result 是否为空
        if not state.get("analysis_result"):
            logger.warning(
                "analysis_result 为空，触发精炼: session=%s",
                state.get("session_id", "?"),
            )
            return {"next_action": "refine"}

        threshold = settings.REFLECTION_PASS_THRESHOLD
        max_rounds = settings.REFLECTION_MAX_ROUNDS

        reflection = await self.llm.reflect_analysis(
            symptoms=state["collected_symptoms"],
            history=state["medical_history"],
            rag_context=state["rag_context"],
            analysis_result=state["analysis_result"],
            urgency_level=state["urgency_level"],
            department=state["recommended_department"],
        )

        score = reflection.get("score", threshold)
        passed = reflection.get("passed", score >= threshold)
        feedback = reflection.get("feedback", "")
        critical = reflection.get("critical_issues", [])

        current_round = state.get("reflection_round", 0)

        # 结构化日志
        logger.info(
            "REFLECT_RESULT session=%s round=%d/%d score=%d passed=%s issues=%d",
            state.get("session_id", "?"),
            current_round, max_rounds, score, passed, len(critical),
        )

        # 追踪统计数据
        reflection_tracker.record_reflection(score, passed, current_round)

        # 决定 next_action
        if passed:
            next_action = "continue"
        elif current_round >= max_rounds:
            logger.warning(
                "反思轮次已用尽但质量仍不达标: session=%s round=%d score=%d",
                state.get("session_id", "?"), current_round, score,
            )
            next_action = "continue"
        else:
            logger.info(
                "触发精炼: session=%s round=%d/%d score=%d",
                state.get("session_id", "?"),
                current_round, max_rounds, score,
            )
            next_action = "refine"

        return {
            "reflection_feedback": feedback,
            "reflection_score": score,
            "reflection_passed": passed,
            "next_action": next_action,
        }

    async def generate_report(self, state: ConsultationState) -> Dict:
        """节点5：生成最终问诊报告（返回部分状态更新）

        反思反馈仅用于精炼 prompt，严禁写入最终报告。
        若轮次用尽后评分仍不达标，报告开头添加"建议人工复核"声明。
        """
        report_text = await self.llm.generate_report(
            analysis=state["analysis_result"],
            urgency=state["urgency_level"],
            dept=state["recommended_department"],
        )

        # 检查是否需要添加人工复核声明
        max_rounds = settings.REFLECTION_MAX_ROUNDS
        threshold = settings.REFLECTION_PASS_THRESHOLD
        reflection_round = state.get("reflection_round", 0)
        reflection_score = state.get("reflection_score", threshold)

        if reflection_round >= max_rounds and reflection_score < threshold:
            disclaimer = (
                "> ⚠️ **建议人工复核**：本报告由 AI 自动生成，"
                "系统内部质量评估未达到自动通过标准，"
                "建议由专业医务人员复核后使用。\n\n"
            )
            report_text = disclaimer + report_text
            logger.warning(
                "报告添加人工复核声明: session=%s round=%d score=%d",
                state.get("session_id", "?"),
                reflection_round, reflection_score,
            )

        return {"report": report_text, "is_complete": True}
