# backend/app/services/llm_service.py
import httpx
import json
import logging
from typing import Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.client = httpx.AsyncClient(timeout=60)

    async def _call_llm(self, prompt: str) -> str:
        """统一调用大模型接口"""
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

        logger.info("LLM RESPONSE: status=%s, body=%.500s", resp.status_code, resp.text)
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
        res = await self._call_llm(prompt)
        res = self._clean_json_response(res)
        data = json.loads(res)
        data["next_ask"] = ""  # 强制空，后端自行追问
        return data

    async def analyze_diagnosis(self, symptoms: list, history: str, rag_context: str) -> Dict[str, str]:
        """症状诊断、紧急分级、推荐科室"""
        prompt = f"""
你是一名专业内科医生。请根据以下信息进行分析，并以 JSON 格式返回诊断结果。

参考医学知识库：
{rag_context}

用户症状：{symptoms}
病史：{history}

请分析并返回以下 JSON 格式：
{{
    "content": "详细的诊断分析、可能病因、建议检查",
    "urgency_level": "low/medium/high/emergency",
    "department": "推荐就诊科室"
}}

只输出 JSON，不要其他文本。
"""
        res = await self._call_llm(prompt)
        res = self._clean_json_response(res)
        return json.loads(res)

    async def generate_report(self, analysis: str, urgency: str, dept: str) -> str:
        """根据诊断分析生成结构化的问诊报告（Markdown格式）"""
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
        res = await self._call_llm(prompt)
        return res.strip()
