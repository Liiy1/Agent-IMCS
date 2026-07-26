from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import uuid
import logging
import time
import statistics
from collections import deque

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.session_service import SessionService, VersionConflict, SessionNotFound
from app.services.llm_service import llm_tracker


# ── 延迟统计 ────────────────────────────────────────────
# 全局 LatencyTracker，请求完成后记录延迟
class LatencyTracker:
    def __init__(self, window=5000):
        self._latencies = deque(maxlen=window)

    def record(self, seconds: float):
        self._latencies.append(seconds * 1000)  # 转 ms

    def stats(self):
        if not self._latencies:
            return {}
        vals = sorted(self._latencies)
        n = len(vals)
        return {
            "count": n,
            "avg_ms": sum(vals) / n,
            "min_ms": vals[0],
            "p50_ms": vals[int(n * 0.50)],
            "p90_ms": vals[int(n * 0.90)],
            "p95_ms": vals[int(n * 0.95)],
            "p99_ms": vals[int(n * 0.99)],
            "max_ms": vals[-1],
        }

    def report(self):
        s = self.stats()
        if not s:
            return "（尚无数据）"
        return (
            f"请求次数={s['count']}, "
            f"平均={s['avg_ms']:.0f}ms, "
            f"P50={s['p50_ms']:.0f}ms, "
            f"P90={s['p90_ms']:.0f}ms, "
            f"P95={s['p95_ms']:.0f}ms, "
            f"P99={s['p99_ms']:.0f}ms, "
            f"最大={s['max_ms']:.0f}ms"
        )

# 全局实例（在模块级别共享）
request_tracker = LatencyTracker()

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
    reflection_rounds: int = 0  # 反思轮次（0=未触发，1=初版通过，2+=精炼后）


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
    _t0 = time.monotonic()

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

    # 6) 计时 + 返回结果
    elapsed = time.monotonic() - _t0
    request_tracker.record(elapsed)

    reflection_round = final_state.get("reflection_round", 0)
    reflection_score = final_state.get("reflection_score", 0)
    logger.info(
        "LATENCY session=%s duration=%.3fs next_action=%s symptoms=%d urgency=%s "
        "reflect_round=%d reflect_score=%d",
        session_id, elapsed,
        final_state.get("next_action", "?"),
        len(final_state.get("collected_symptoms", [])),
        final_state.get("urgency_level", "?"),
        reflection_round, reflection_score,
    )
    # 每 50 次输出一次聚合统计
    if request_tracker.stats().get("count", 0) % 50 == 0:
        logger.info("LATENCY_SUMMARY %s", request_tracker.report())
        from app.agent.nodes import reflection_tracker
        logger.info("REFLECTION_SUMMARY %s", reflection_tracker.report())

    return ConsultResponse(
        session_id=session_id,
        response=final_state.get("report", ""),
        urgency_level=final_state.get("urgency_level", "low"),
        is_complete=final_state.get("is_complete", False),
        user_token=request.user_token,
        reflection_rounds=reflection_round,
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


# ── 性能统计（仅调试用，不在 OpenAPI 文档中暴露） ─────────

from pydantic import BaseModel as PM

class StatsResponse(PM):
    request_stats: dict
    llm_stats: dict

@router.get("/stats", include_in_schema=False)
async def get_stats():
    """获取请求延迟和 LLM 调用统计"""
    return StatsResponse(
        request_stats=request_tracker.stats(),
        llm_stats={
            "extract_symptom": llm_tracker.stats("extract_symptom"),
            "analyze_diagnosis": llm_tracker.stats("analyze_diagnosis"),
            "generate_report": llm_tracker.stats("generate_report"),
        },
    )