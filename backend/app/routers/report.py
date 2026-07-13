from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Dict, Any
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database import get_db
from app.services.session_service import SessionService, SessionNotFound

logger = logging.getLogger(__name__)

router = APIRouter()


class ReportResponse(BaseModel):
    session_id: str
    report_content: str
    status: str
    urgency_level: str
    recommended_department: str
    created_at: Optional[str] = None


def get_session_service() -> SessionService:
    from app.main import session_service
    return session_service


@router.get("/export", response_model=ReportResponse)
async def export_report(
    session_id: str = Query(...),
    user_token: str = Query(...),
    db: AsyncSession = Depends(get_db),
    svc: SessionService = Depends(get_session_service),
):
    """导出问诊报告（带 user_token 归属校验）"""
    conv = await svc.get_conversation(db, session_id, user_token)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")

    if not conv.final_report:
        raise HTTPException(status_code=400, detail="该会话尚未完成问诊，无报告可导出")

    return ReportResponse(
        session_id=session_id,
        report_content=conv.final_report,
        status=conv.status,
        urgency_level=conv.urgency_level or "low",
        recommended_department=conv.recommended_department or "",
        created_at=conv.created_at.isoformat() if conv.created_at else None,
    )
