import json
import logging
import re
from typing import Dict, Any, List, Optional

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


# ── 常用药品名称关键词（用于 MCP 工具预检） ──────────────

COMMON_DRUG_KEYWORDS = [
    "阿司匹林", "布洛芬", "对乙酰氨基酚", "双氯芬酸", "萘普生",
    "阿莫西林", "头孢", "左氧氟沙星", "阿奇霉素", "克拉霉素", "甲硝唑",
    "硝苯地平", "氨氯地平", "卡托普利", "缬沙坦", "美托洛尔",
    "二甲双胍", "格列本脲", "阿卡波糖",
    "阿托伐他汀", "辛伐他汀",
    "奥美拉唑", "多潘立酮", "蒙脱石散",
    "氯雷他定", "西替利嗪",
    "氨溴索", "右美沙芬",
    "地西泮", "卡马西平",
    "华法林", "氯吡格雷",
    "胰岛素", "青霉素", "维生素",
]


class ConsultationNodes:
    def __init__(self, llm_service, rag_retriever, mcp_client=None):
        self.llm = llm_service
        self.rag = rag_retriever
        self.mcp_client = mcp_client

    # ── MCP 工具辅助方法 ─────────────────────────────────

    async def _detect_and_call_tools(self, user_message: str) -> str:
        """预检用户输入，自动调用相关 MCP 工具并返回上下文文本

        关键词检测策略：
        - 药品名称 → get_drug_info / check_drug_interaction
        - 文件路径 → read_lab_report
        - "历史"/"上次" → get_patient_history
        """
        if not self.mcp_client:
            return ""

        context_parts = []

        # 1) 药品检测
        found_drugs = []
        for drug_keyword in COMMON_DRUG_KEYWORDS:
            if drug_keyword in user_message:
                found_drugs.append(drug_keyword)
        # 去重（如多个药品名被匹配）
        found_drugs = list(dict.fromkeys(found_drugs))

        for drug_name in found_drugs:
            result = await self.mcp_client.call_tool("get_drug_info", {"drug_name": drug_name})
            if not result.get("is_error") and result.get("content", {}).get("found"):
                drug = result["content"]["drug"]
                context_parts.append(
                    f"【药品信息 - {drug_name}】适应症：{drug['indications']}。"
                    f"副作用：{drug['side_effects']}。禁忌：{drug['contraindications']}。"
                )
                logger.info("MCP 预检: 查询药品「%s」成功", drug_name)

        # 如果检测到两种药品，同时查询相互作用
        if len(found_drugs) >= 2:
            for i in range(len(found_drugs)):
                for j in range(i + 1, len(found_drugs)):
                    result = await self.mcp_client.call_tool(
                        "check_drug_interaction",
                        {"drug_a": found_drugs[i], "drug_b": found_drugs[j]},
                    )
                    if not result.get("is_error") and result.get("content", {}).get("found"):
                        c = result["content"]
                        if c.get("has_interaction"):
                            context_parts.append(
                                f"【药物相互作用】{c['drug_a']} + {c['drug_b']}: "
                                f"严重程度={c['severity_label']}。{c['description']}。"
                                f"建议：{c['recommendation']}。"
                            )

        # 2) 文件路径检测
        file_patterns = re.findall(r'(?:报告|化验单|检查单|血常规|报告单)\s*(?:\S*\.(?:pdf|txt|json))?', user_message)
        # 也检测路径模式
        path_patterns = re.findall(r'[\w/\\:]+\.(pdf|txt|json)', user_message)
        all_paths = file_patterns + path_patterns
        for fp in all_paths:
            # 尝试读取，无实际文件时会返回模拟数据
            result = await self.mcp_client.call_tool("read_lab_report", {"file_path": fp})
            if not result.get("is_error") and result.get("content", {}).get("found"):
                context_parts.append(
                    f"【检验报告】{result['content'].get('content', '')}"
                )
                logger.info("MCP 预检: 读取文件「%s」", fp)

        # 3) 历史病历检测
        if any(kw in user_message for kw in ["上次", "历史", "以前", "过往", "老毛病"]):
            # 使用当前 state 中的 session_id 或 user_token
            result = await self.mcp_client.call_tool("get_patient_history", {"patient_id": "test"})
            if not result.get("is_error") and result.get("content", {}).get("found"):
                histories = result["content"].get("histories", [])
                if histories:
                    history_text = "; ".join(
                        f"{h['date']}: {h['complaint']} → {h['diagnosis']}"
                        for h in histories
                    )
                    context_parts.append(f"【历史病历】{history_text}")
                    logger.info("MCP 预检: 查询历史病历成功")

        return "\n\n".join(context_parts)

    async def _execute_llm_tool_calls(self, tool_calls_raw: Any, user_message: str) -> str:
        """执行 LLM 请求的工具调用，返回结果文本"""
        if not self.mcp_client:
            return ""

        if isinstance(tool_calls_raw, str):
            try:
                tool_calls_raw = json.loads(tool_calls_raw)
            except json.JSONDecodeError:
                return ""

        if not isinstance(tool_calls_raw, list):
            tool_calls_raw = [tool_calls_raw]

        results = []
        for tc in tool_calls_raw:
            tool_name = tc.get("tool", "") if isinstance(tc, dict) else ""
            args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
            if not tool_name:
                continue

            result = await self.mcp_client.call_tool(tool_name, args)
            if not result.get("is_error"):
                content = result.get("content", {})
                results.append(json.dumps(content, ensure_ascii=False, indent=2))
                logger.info("LLM 请求工具调用: %s(%s) 成功", tool_name, args)
            else:
                logger.warning("LLM 请求工具调用: %s(%s) 失败: %s", tool_name, args, result.get("content"))

        return "\n\n".join(results)

    # ── 节点实现 ────────────────────────────────────────

    async def collect_symptoms(self, state: ConsultationState) -> Dict:
        """节点1：收集症状（含 MCP 工具调用子步骤）

        流程：
        1. 关键词预检 → 自动调用 MCP 工具（药品/病历/文件）
        2. LLM 提取症状 + 病史（含工具结果作为上下文）
        3. 若 LLM 请求额外工具调用 → 执行 → 重新提取
        4. 追问策略判断（与现有逻辑一致）
        """
        user_msg = state["user_message"]

        # ── 步骤 1：MCP 工具预检 ──
        mcp_context = ""
        if self.mcp_client and settings.MCP_ENABLED:
            mcp_context = await self._detect_and_call_tools(user_msg)

        # 构造完整上下文
        context_parts = []
        if state["collected_symptoms"]:
            context_parts.append(
                f"当前已知症状：{', '.join(state['collected_symptoms'])}。"
            )
        if state["medical_history"]:
            history_str = ", ".join(state["medical_history"])
            context_parts.append(f"当前已知病史：{history_str}。")
        context_parts.append(f"用户最新描述：{user_msg}")
        if mcp_context:
            context_parts.append(f"外部数据查询结果：\n{mcp_context}")

        context = "\n".join(context_parts)
        logger.debug("collect_symptoms context sent to LLM: %.500s", context)

        # ── 步骤 2：LLM 提取症状 ──
        symptom_res = await self.llm.extract_symptom(context)
        logger.debug("symptom_res: %s", symptom_res)

        # ── 步骤 3：处理 LLM 请求的额外工具调用 ──
        tool_calls = symptom_res.get("tool_calls")
        if tool_calls and self.mcp_client and settings.MCP_ENABLED:
            logger.info("LLM 请求工具调用，开始执行...")
            tool_results = await self._execute_llm_tool_calls(tool_calls, user_msg)
            if tool_results:
                enriched_context = context + f"\n\n补充查询结果：\n{tool_results}"
                logger.debug("重新提取症状（含工具结果）...")
                symptom_res = await self.llm.extract_symptom(enriched_context)

        # ── 合并症状（去重） ──
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

        # ── 追问策略：症状不足 3 个且无病史时继续追问 ──
        if len(collected) < 3 and not existing:
            symptoms_str = ', '.join(collected) if collected else "无"
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
        """节点3：症状分析 — 支持 MCP 工具调用 + 反思精炼

        在分析前，若有药品/文件/历史查询需求，调用 MCP 工具
        并将结果合并到 rag_context 中。
        """
        # ── MCP 工具调用：检查是否需要补充外部数据 ──
        enriched_rag = state["rag_context"]
        if self.mcp_client and settings.MCP_ENABLED:
            extra_context = await self._detect_analyze_tool_needs(state)
            if extra_context:
                enriched_rag = enriched_rag + "\n\n" + extra_context
                logger.info("analyze_symptoms: MCP 数据已合并到 RAG 上下文")

        # ── LLM 分析 ──
        feedback = state.get("reflection_feedback", "")
        prev_analysis = state.get("analysis_result", "")

        if feedback and state.get("reflection_round", 0) > 0:
            logger.info(
                "精炼分析: round=%d, feedback=%.150s",
                state["reflection_round"], feedback,
            )
            analysis = await self.llm.analyze_diagnosis(
                symptoms=state["collected_symptoms"],
                history=state["medical_history"],
                rag_context=enriched_rag,
                previous_analysis=prev_analysis,
                reflection_feedback=feedback,
            )
        else:
            analysis = await self.llm.analyze_diagnosis(
                symptoms=state["collected_symptoms"],
                history=state["medical_history"],
                rag_context=enriched_rag,
            )

        return {
            "analysis_result": analysis["content"],
            "urgency_level": analysis.get("urgency_level", "low"),
            "recommended_department": analysis.get("department", ""),
            "reflection_round": state.get("reflection_round", 0) + 1,
        }

    async def _detect_analyze_tool_needs(self, state: ConsultationState) -> str:
        """analyze 节点专用的 MCP 工具检测

        检查症状列表和用户消息是否包含需要额外查询的内容。
        """
        if not self.mcp_client:
            return ""

        combined_text = " ".join([
            state.get("user_message", ""),
            " ".join(state.get("collected_symptoms", [])),
        ])

        # 检测药品查询意图
        found_drugs = []
        for drug_keyword in COMMON_DRUG_KEYWORDS:
            if drug_keyword in combined_text:
                found_drugs.append(drug_keyword)
        found_drugs = list(dict.fromkeys(found_drugs))

        # 如果症状分析前尚未查询过药品信息
        if found_drugs:
            parts = []
            for drug_name in found_drugs:
                result = await self.mcp_client.call_tool("get_drug_info", {"drug_name": drug_name})
                if not result.get("is_error") and result.get("content", {}).get("found"):
                    drug = result["content"]["drug"]
                    parts.append(
                        f"【药品信息 - {drug_name}】适应症：{drug['indications']}。"
                        f"副作用：{drug['side_effects']}。禁忌：{drug['contraindications']}。"
                    )
            return "\n\n".join(parts)

        return ""

    async def reflect_analysis(self, state: ConsultationState) -> Dict:
        """节点4：反思诊断分析质量，决定是否需要精炼（返回部分状态更新）"""
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

        logger.info(
            "REFLECT_RESULT session=%s round=%d/%d score=%d passed=%s issues=%d",
            state.get("session_id", "?"),
            current_round, max_rounds, score, passed, len(critical),
        )

        reflection_tracker.record_reflection(score, passed, current_round)

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
