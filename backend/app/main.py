from dotenv import load_dotenv
load_dotenv()  # 默认读取当前目录下的 .env
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.database import engine, Base
from app.agent.graph import ConsultationGraph
from app.rag.retriever import MedicalRAGRetriever
from app.services.llm_service import LLMService
from app.services.redis_cache import RedisCache
from app.services.session_service import SessionService
from app.routers import consultation, report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局单例服务
llm_service = None
rag_retriever = None
consultation_graph = None
redis_cache = None
session_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_service, rag_retriever, consultation_graph
    global redis_cache, session_service
    logger.info("启动AI智能问诊系统...")

    # 设置 Hugging Face 镜像（必须在加载模型前）
    import os
    os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

    # 初始化缓存与持久化服务
    redis_cache = RedisCache()
    await redis_cache.init()
    session_service = SessionService(redis_cache)

    # 初始化业务服务
    llm_service = LLMService()
    rag_retriever = MedicalRAGRetriever()
    consultation_graph = ConsultationGraph(llm_service, rag_retriever)

    # 导入 ORM 模型确保注册，然后自动创建数据库表
    import app.models  # noqa: F401 — 注册模型到 Base.metadata
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("系统初始化完成")
    yield
    logger.info("关闭系统...")

    # 关闭 Redis 连接
    if redis_cache:
        await redis_cache.close()


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI智能问诊系统 - 基于LLM的医疗咨询助手",
    lifespan=lifespan,
    docs_url="/api/docs"
)

# 跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(consultation.router, prefix="/api/v1/consultation", tags=["问诊"])
app.include_router(report.router, prefix="/api/v1/report", tags=["报告"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.APP_NAME}
