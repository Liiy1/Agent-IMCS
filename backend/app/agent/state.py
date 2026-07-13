from typing import TypedDict, List, Dict, Optional, Union

class ConsultationState(TypedDict):
    """LangGraph 问诊会话状态"""
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