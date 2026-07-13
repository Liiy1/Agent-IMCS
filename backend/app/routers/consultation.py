from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import uuid
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.session_service import SessionService, VersionConflict, SessionNotFound

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Pydantic 请求/响应模型
# ------------------------------------------------------------------

class ConsultRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    user_token: Optional[str] = None


class ConsultResponse(BaseModel):
    session_id: str
    response: str
    urgency_level: str
    is_complete: bool
    user_token: str


class MessageItem(BaseModel):
    role: str
    content: str
    created_at: Optional[str] = None


class HistoryResponse(BaseModel):
    session_id: str
    user_token: str
    messages: List[MessageItem]


# ------------------------------------------------------------------
# 依赖注入：从全局获取 SessionService
# ------------------------------------------------------------------

def get_session_service() -> SessionService:
    from app.main import session_service
    return session_service


# ------------------------------------------------------------------
# 路由
# ------------------------------------------------------------------

@router.post("/consult", response_model=ConsultResponse)
async def consult(
    request: ConsultRequest,
    db: AsyncSession = Depends(get_db),
    svc: SessionService = Depends(get_session_service),
):
    """问诊对话入口 — 带持久化状态保存/恢复"""
    # 1) 参数校验
    if not request.user_token:
        raise HTTPException(status_code=400, detail="user_token 是必填参数")
    if not request.message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    session_id = request.session_id or str(uuid.uuid4())

    # 2) 加载累积状态（多轮对话恢复）
    accumulated = await svc.load_state(db, session_id)
    if accumulated is None:
        accumulated = {}  # 首次访问，使用空默认值
    logger.info("accumulated state: %s", accumulated)

    # 3) 执行问诊工作流
    from app.main import consultation_graph

    try:
        final_state = await consultation_graph.run(
            session_id=session_id,
            user_message=request.message,
            accumulated_state=accumulated,
        )
    except Exception as e:
        logger.error("问诊工作流执行失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="问诊服务异常，请稍后重试")

    # 4) 先持久化累积状态（确保 Conversation 记录存在且 user_token 正确）
    try:
        state_payload = {
            "collected_symptoms": final_state.get("collected_symptoms", []),
            "medical_history": final_state.get("medical_history", []),
            "rag_context": final_state.get("rag_context", ""),
            "analysis_result": final_state.get("analysis_result", ""),
            "urgency_level": final_state.get("urgency_level", "low"),
            "recommended_department": final_state.get("recommended_department", ""),
            "is_complete": final_state.get("is_complete", False),
            "report": final_state.get("report", ""),
            "_version": accumulated.get("_version", 0),
        }
        await svc.save_state(db, session_id, request.user_token, state_payload)
    except VersionConflict:
        logger.warning("会话 %s 版本冲突，重试保存", session_id)
        try:
            reloaded = await svc.load_state(db, session_id) or {}
            state_payload["_version"] = reloaded.get("_version", 0)
            await svc.save_state(db, session_id, request.user_token, state_payload)
        except VersionConflict:
            logger.error("会话 %s 重试仍冲突，放弃保存", session_id)
    except Exception as e:
        logger.error("状态持久化失败: %s", e)

    # 5) 再持久化消息（此时 Conversation 记录已存在）
    try:
        await svc.save_message(db, session_id, "user", request.message)
        await svc.save_message(
            db, session_id, "assistant",
            final_state.get("report", ""),
            metadata={
                "urgency_level": final_state.get("urgency_level", "low"),
                "is_complete": final_state.get("is_complete", False),
            },
        )
    except Exception as e:
        logger.error("消息持久化失败: %s", e)

    # 6) 返回结果
    return ConsultResponse(
        session_id=session_id,
        response=final_state.get("report", ""),
        urgency_level=final_state.get("urgency_level", "low"),
        is_complete=final_state.get("is_complete", False),
        user_token=request.user_token,
    )


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    session_id: str = Query(...),
    user_token: str = Query(...),
    db: AsyncSession = Depends(get_db),
    svc: SessionService = Depends(get_session_service),
):
    """获取会话历史消息（带 user_token 归属校验）"""
    try:
        messages = await svc.get_messages(db, session_id, user_token)
    except SessionNotFound:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")

    return HistoryResponse(
        session_id=session_id,
        user_token=user_token,
        messages=[MessageItem(**m) for m in messages],
    )