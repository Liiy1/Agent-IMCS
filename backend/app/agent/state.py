from typing import TypedDict, List, Dict, Optional, Union

class ConsultationState(TypedDict):
    """LangGraph 问诊会话状态 — 含反思机制"""
    session_id: str
    user_message: str
    collected_symptoms: List[str]
    medical_history: Union[str, List[str]]  # 兼容：单轮为 str，多轮累积为 List[str]
    rag_context: str
    analysis_result: str
    urgency_level: str  # low / medium / high / emergency
    recommended_department: str
    report: str
    next_action: str
    is_complete: bool
    error: str

    # --- 反思机制字段 ---
    reflection_feedback: str   # 反思反馈（仅用于精炼 prompt，不写入最终报告）
    reflection_score: int      # 质量评分 1-5（5=优秀）
    reflection_round: int      # 当前分析轮次（0=初版，1+=精炼）
    reflection_passed: bool    # 本轮反思是否通过质量检查