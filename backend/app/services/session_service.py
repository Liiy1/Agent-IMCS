import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.models import Conversation, Message, ConversationState
from app.services.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class VersionConflict(Exception):
    """乐观锁版本冲突"""
    pass


class SessionNotFound(Exception):
    """会话不存在或无权访问"""
    pass


class SessionService:
    """问诊会话持久化服务

    职责：
    - 跨多轮对话保存/恢复 ConsultationState
    - 持久化消息记录
    - 通过 user_token 实现简单的会话归属校验
    - 乐观锁防止并发写入覆盖
    """

    def __init__(self, redis: RedisCache):
        self.redis = redis

    # ------------------------------------------------------------------
    # 状态持久化
    # ------------------------------------------------------------------

    async def load_state(self, db: AsyncSession, session_id: str) -> Optional[Dict[str, Any]]:
        """加载累积状态，优先从 Redis 读取，回退到 MySQL

        Returns:
            dict: 包含 collected_symptoms, medical_history, urgency_level, ...
            None: 会话不存在（首次访问）
        """
        # 1) Redis 快速读取
        cached = await self.redis.get_state(session_id)
        if cached is not None:
            return cached

        # 2) 回退到 MySQL
        result = await db.execute(
            select(Conversation, ConversationState)
            .outerjoin(ConversationState, ConversationState.conversation_id == Conversation.id)
            .where(Conversation.session_id == session_id)
        )
        row = result.one_or_none()
        if row is None:
            return None  # 首次访问

        conv, state = row

        payload = {
            "collected_symptoms": state.collected_symptoms if state else [],
            "medical_history": state.medical_history if state else [],
            "rag_context": state.rag_context if state else "",
            "analysis_result": state.analysis_result if state else "",
            "urgency_level": conv.urgency_level or "low",
            "recommended_department": conv.recommended_department or "",
            "_version": state.version if state else 1,
        }

        # 3) 回写 Redis 缓存
        await self.redis.set_state(session_id, payload)

        return payload

    async def save_state(
        self,
        db: AsyncSession,
        session_id: str,
        user_token: str,
        state: Dict[str, Any],
    ) -> None:
        """保存累积状态（upsert + 乐观锁）

        Raises:
            VersionConflict: 版本冲突（并发写入）
        """
        # 1) 查找或创建 Conversation
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conv = result.scalar_one_or_none()

        if conv is None:
            conv = Conversation(
                session_id=session_id,
                user_token=user_token,
            )
            db.add(conv)
            await db.flush()  # 获取 ID
        else:
            # 更新会话级字段
            conv.urgency_level = state.get("urgency_level", "low")
            conv.recommended_department = state.get("recommended_department", "")
            if state.get("is_complete"):
                conv.status = "complete"
                conv.final_report = state.get("report", "")

        # 2) 乐观锁写入 ConversationState
        old_version = state.get("_version", 0)

        result = await db.execute(
            select(ConversationState).where(
                ConversationState.conversation_id == conv.id
            )
        )
        state_row = result.scalar_one_or_none()

        if state_row is None:
            # 首次创建
            state_row = ConversationState(
                conversation_id=conv.id,
                collected_symptoms=state.get("collected_symptoms", []),
                medical_history=state.get("medical_history", []),
                rag_context=state.get("rag_context", ""),
                analysis_result=state.get("analysis_result", ""),
                version=1,
            )
            db.add(state_row)
        else:
            # 乐观锁检测
            if state_row.version != old_version:
                raise VersionConflict(
                    f"会话 {session_id} 版本冲突: 期望 version={old_version}, "
                    f"实际 version={state_row.version}"
                )
            state_row.collected_symptoms = state.get("collected_symptoms", [])
            state_row.medical_history = state.get("medical_history", [])
            state_row.rag_context = state.get("rag_context", "")
            state_row.analysis_result = state.get("analysis_result", "")
            state_row.version += 1

        await db.commit()

        # 3) 更新 Redis 缓存（不含 _version，以最新版本写入）
        cache_payload = {**state}
        cache_payload["_version"] = state_row.version if state_row else 1
        await self.redis.set_state(session_id, cache_payload)

    # ------------------------------------------------------------------
    # 消息持久化
    # ------------------------------------------------------------------

    async def save_message(
        self,
        db: AsyncSession,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存一条消息（需先调用 save_state 创建 Conversation 记录）"""
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conv = result.scalar_one_or_none()
        if conv is None:
            logger.warning("会话 %s Conversation 不存在，跳过消息保存", session_id)
            return

        msg = Message(
            conversation_id=conv.id,
            role=role,
            content=content,
            metadata_json=metadata,
        )
        db.add(msg)
        await db.commit()

    # ------------------------------------------------------------------
    # 会话查询（带 user_token 归属校验）
    # ------------------------------------------------------------------

    async def get_conversation(
        self,
        db: AsyncSession,
        session_id: str,
        user_token: str,
    ) -> Optional[Conversation]:
        """获取会话，校验 user_token，不匹配返回 None"""
        result = await db.execute(
            select(Conversation).where(
                Conversation.session_id == session_id,
                Conversation.user_token == user_token,
            )
        )
        return result.scalar_one_or_none()

    async def get_messages(
        self,
        db: AsyncSession,
        session_id: str,
        user_token: str,
    ) -> List[Dict[str, Any]]:
        """获取会话消息列表，带归属校验"""
        result = await db.execute(
            select(Conversation).where(
                Conversation.session_id == session_id,
                Conversation.user_token == user_token,
            )
        )
        conv = result.scalar_one_or_none()
        if conv is None:
            raise SessionNotFound("会话不存在或无权访问")

        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(Message.created_at.asc())
        )
        messages = msg_result.scalars().all()

        return [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ]
