# backend/app/services/llm_service.py
import httpx
import json
import logging
import time
from typing import Dict, Any, Optional
from collections import deque

from app.config import settings

logger = logging.getLogger(__name__)


class LLMCallTracker:
    """追踪每次 LLM 调用的延迟"""
    def __init__(self, window=2000):
        self._calls = deque(maxlen=window)

    def record(self, method: str, duration: float, success: bool):
        self._calls.append({
            "method": method,
            "ms": duration * 1000,
            "success": success,
        })

    def stats(self, method: str = None):
        vals = [c["ms"] for c in self._calls
                if (method is None or c["method"] == method) and c["success"]]
        if not vals:
            return {}
        vals.sort()
        n = len(vals)
        return {
            "count": n,
            "avg_ms": sum(vals) / n,
            "p50_ms": vals[int(n * 0.50)],
            "p95_ms": vals[int(n * 0.95)],
            "p99_ms": vals[int(n * 0.99)],
            "max_ms": vals[-1],
        }

    def report(self):
        lines = []
        for method in ["extract_symptom", "analyze_diagnosis", "reflect_analysis", "generate_report", "generate_followup"]:
            s = self.stats(method)
            if s:
                lines.append(f"  {method}: avg={s['avg_ms']:.0f}ms, P50={s['p50_ms']:.0f}ms, P95={s['p95_ms']:.0f}ms (n={s['count']})")
        return "\n".join(lines) if lines else "（尚无数据）"


llm_tracker = LLMCallTracker()


class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.client = httpx.AsyncClient(timeout=60)

    async def _call_llm(self, prompt: str) -> str:
        """统一调用大模型接口"""
        _t0 = time.monotonic()
        if self.provider == "deepseek":
            headers = {"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": settings.DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            resp = await self.client.post(f"{settings.DEEPSEEK_BASE_URL}/chat/completions", json=payload, headers=headers)
        else:
            headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": settings.OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            resp = await self.client.post(f"{settings.OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers)

        _elapsed = time.monotonic() - _t0
        logger.info("LLM RESPONSE: status=%s, duration=%.3fs, body=%.500s", resp.status_code, _elapsed, resp.text)
        if resp.status_code != 200:
            raise Exception(f"API Error: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    def _clean_json_response(self, text: str) -> str:
        """清理LLM返回的JSON字符串，去除可能的Markdown标记"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    async def extract_symptom(self, user_text: str) -> Dict[str, Any]:
        """提取用户症状、病史（不含追问生成，后端自行控制追问逻辑）"""
        _t0 = time.monotonic()
        prompt = f"""
从以下用户输入中提取所有症状和病史，以 JSON 格式输出。
- symptoms: 字符串列表，列出所有身体不适（如头痛、发烧、咳嗽等）。
- history: 字符串列表，列出所有病史、慢性病、用药史、过敏史等。
- next_ask: 固定返回空字符串 ""。
只输出 JSON，不要其他任何文本。

示例输入："我头痛3天了，伴有恶心"
输出：{{"symptoms": ["头痛", "恶心"], "history": [], "next_ask": ""}}

示例输入："有点发烧，还有鼻塞"
输出：{{"symptoms": ["发烧", "鼻塞"], "history": [], "next_ask": ""}}

用户输入：{user_text}
输出："""
        logger.info("extract_symptom user_text: %s", user_text)
        logger.info("extract_symptom FULL PROMPT:\n%s", prompt)
        try:
            res = await self._call_llm(prompt)
            res = self._clean_json_response(res)
            data = json.loads(res)
            data["next_ask"] = ""  # 强制空，后端自行追问
            llm_tracker.record("extract_symptom", time.monotonic() - _t0, True)
            return data
        except Exception as e:
            llm_tracker.record("extract_symptom", time.monotonic() - _t0, False)
            raise

    async def generate_followup(
        self,
        symptoms: list,
        history: list,
        user_message: str,
    ) -> dict:
        """根据已收集信息生成有针对性的追问

        当症状信息不足时，判断最关键的缺失信息，生成一句自然的口语化追问。
        返回结构化 JSON 含 question / missing_aspect / priority。

        Args:
            symptoms: 当前已收集的症状列表
            history: 当前已收集的病史列表
            user_message: 用户本轮输入原文

        Returns:
            {"question": str, "missing_aspect": str, "priority": str}
        """
        _t0 = time.monotonic()
        prompt = f"""
你是一位耐心、细致的医生，正在通过问诊收集患者病情。

目前已收集信息：
- 症状：{symptoms or "暂无"}
- 病史：{history or "暂无"}
- 患者刚说：{user_message}

请判断当前问诊中缺少哪些关键信息（从以下维度思考）：
1. 发病时间/诱因：什么时候开始的？有没有明显诱因？
2. 部位：具体在哪个位置？
3. 性质：是什么样感觉（钝痛/刺痛/灼烧感/麻木等）？
4. 持续时间/规律：持续性的还是阵发性的？
5. 伴随症状：有没有伴随其他不适？
6. 加重/缓解因素：什么情况下会加重或缓解？
7. 既往史：以前有过类似情况吗？有无基础病、用药史、过敏史？

选择一个当前最紧要、最有助于判断病情的信息缺口，生成一句有针对性的追问。
要求：自然、口语化，像医生在说话，不要一次问多个问题，不要评价患者。

只输出 JSON，不要其他文本：
{{
    "question": "你生成的追问话术",
    "missing_aspect": "这句话尝试收集什么信息（如：疼痛性质、发病时间等）",
    "priority": "high/medium/low"
}}
"""
        try:
            res = await self._call_llm(prompt)
            res = self._clean_json_response(res)
            data = json.loads(res)
            llm_tracker.record("generate_followup", time.monotonic() - _t0, True)
            return data
        except Exception as e:
            llm_tracker.record("generate_followup", time.monotonic() - _t0, False)
            raise

    async def analyze_diagnosis(
        self,
        symptoms: list,
        history: str,
        rag_context: str,
        previous_analysis: str = "",
        reflection_feedback: str = "",
    ) -> Dict[str, str]:
        """症状诊断、紧急分级、推荐科室 — 支持反思精炼

        Args:
            symptoms: 患者症状列表
            history: 病史
            rag_context: RAG 检索的医学知识
            previous_analysis: 上一版分析（空字符串表示初版分析）
            reflection_feedback: 反思反馈（仅精炼轮次传入）
        """
        _t0 = time.monotonic()
        prompt = f"""
你是一名专业内科医生。请根据以下信息进行分析，并以 JSON 格式返回诊断结果。

参考医学知识库：
{rag_context}

用户症状：{symptoms}
病史：{history}
"""

        if previous_analysis and reflection_feedback:
            prompt += f"""
上一版分析：
{previous_analysis}

改进反馈（请据此改进分析）：
{reflection_feedback}

"""

        prompt += """
请分析并返回以下 JSON 格式：
{
    "content": "详细的诊断分析、可能病因、建议检查",
    "urgency_level": "low/medium/high/emergency",
    "department": "推荐就诊科室"
}

只输出 JSON，不要其他文本。
"""
        try:
            res = await self._call_llm(prompt)
            res = self._clean_json_response(res)
            data = json.loads(res)
            llm_tracker.record("analyze_diagnosis", time.monotonic() - _t0, True)
            return data
        except Exception as e:
            llm_tracker.record("analyze_diagnosis", time.monotonic() - _t0, False)
            raise

    async def reflect_analysis(
        self,
        symptoms: list,
        history: str,
        rag_context: str,
        analysis_result: str,
        urgency_level: str,
        department: str,
    ) -> Dict[str, Any]:
        """反思诊断分析质量，返回评分和改进建议

        对症状列表和 RAG 上下文做长度截断以防止超 token。
        异常时返回保守通过值，不阻塞工作流。
        """
        _t0 = time.monotonic()

        # 截断症状列表：最多 20 个
        truncated_symptoms = symptoms[:20] if len(symptoms) > 20 else symptoms
        # 截断 RAG 上下文：最多 2000 字符
        truncated_rag = rag_context[:2000] if len(rag_context) > 2000 else rag_context

        prompt = f"""
你是一位资深医学专家，负责对AI生成的诊断分析进行质量评审。
请从以下三个维度严格评估：

1. 一致性：紧急程度是否与症状和诊断匹配？科室推荐是否合理？
2. 完整性：是否覆盖了所有症状和可能病因？有无重要遗漏？
3. 医学准确性：分析是否与参考医学知识一致？有无明显错误？

参考医学知识库：
{truncated_rag}

患者症状：{truncated_symptoms}
病史：{history}

当前诊断分析：
{analysis_result}

当前紧急程度：{urgency_level}
当前推荐科室：{department}

请按以下 JSON 格式返回评审结果（只输出 JSON，不要其他文本）：
{{
    "score": <整数 1-5，1=很差需要大改，3=及格，5=优秀无需修改>,
    "passed": <bool，score>=3 为通过>,
    "feedback": "具体的改进建议，指出哪里需要改进（供分析精炼使用）",
    "critical_issues": ["严重问题列表"],
    "minor_issues": ["次要问题列表"]
}}
"""
        try:
            res = await self._call_llm(prompt)
            res = self._clean_json_response(res)
            data = json.loads(res)
            llm_tracker.record("reflect_analysis", time.monotonic() - _t0, True)
            return data
        except Exception as e:
            llm_tracker.record("reflect_analysis", time.monotonic() - _t0, False)
            logger.error("反思分析调用失败，降级通过: %s", e, exc_info=True)
            # 降级：保守返回通过值，不阻塞工作流
            return {
                "score": 4,
                "passed": True,
                "feedback": "反思引擎暂时不可用，已跳过审核。",
                "critical_issues": [],
                "minor_issues": [],
            }

    async def generate_report(self, analysis: str, urgency: str, dept: str) -> str:
        """根据诊断分析生成结构化的问诊报告（Markdown格式）"""
        _t0 = time.monotonic()
        prompt = f"""
你是一名专业内科医生。请根据以下诊断分析，生成一份结构化的问诊报告（Markdown格式）。

诊断分析：{analysis}
紧急程度：{urgency}
推荐科室：{dept}

报告应包含以下部分：
1. **主诉** — 总结患者主要症状
2. **诊断分析** — 详细分析可能的病因
3. **紧急程度** — 评估是否需要立即就医
4. **就诊建议** — 推荐科室及注意事项
5. **建议检查** — 推荐的相关检查项目

请以 Markdown 格式输出，语言简洁专业。
"""
        try:
            res = await self._call_llm(prompt)
            llm_tracker.record("generate_report", time.monotonic() - _t0, True)
            return res.strip()
        except Exception as e:
            llm_tracker.record("generate_report", time.monotonic() - _t0, False)
            raise
