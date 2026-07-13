from sqlalchemy import Column, ForeignKey, String, Text, DateTime, Boolean, Integer, JSON, Enum as SAEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class Conversation(Base):
    """问诊会话 — 一个 session_id 对应一行"""
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_token = Column(String(64), nullable=False, index=True)

    status = Column(String(16), nullable=False, default="active")  # active | complete
    urgency_level = Column(String(20), default="low")
    recommended_department = Column(String(100), default="")
    final_report = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    state = relationship("ConversationState", back_populates="conversation", uselist=False, cascade="all, delete-orphan")


class Message(Base):
    """对话消息 — 用户提问或助手回复"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"),nullable=False)
    # 注意: SQLite 不支持 ALTER ADD FOREIGN KEY，但 MySQL/asyncmy 支持
    # conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)

    role = Column(String(16), nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages", foreign_keys=[conversation_id])


class ConversationState(Base):
    """LangGraph 累积状态 — 与会话 1:1"""
    __tablename__ = "conversation_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    #conversation_id = Column(String(36), unique=True, nullable=False)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), unique=True, nullable=False)

    collected_symptoms = Column(JSON, default=list)
    medical_history = Column(JSON, default=list)  # JSON 数组，去重存储
    rag_context = Column(Text, default="")
    analysis_result = Column(Text, default="")

    version = Column(Integer, nullable=False, default=1)  # 乐观锁

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    conversation = relationship("Conversation", back_populates="state", foreign_keys=[conversation_id])
