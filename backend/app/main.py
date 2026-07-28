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
from app.mcp.client import MCPClient
from app.mcp.server import drug_db, patient_history, file_reader, scheduler
from app.routers import consultation, report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局单例服务
llm_service = None
rag_retriever = None
consultation_graph = None
redis_cache = None
session_service = None
mcp_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_service, rag_retriever, consultation_graph
    global redis_cache, session_service, mcp_client
    logger.info("启动AI智能问诊系统...")

    # 设置 Hugging Face 镜像（必须在加载模型前）
    import os
    os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

    # 初始化缓存与持久化服务
    redis_cache = RedisCache()
    await redis_cache.init()
    session_service = SessionService(redis_cache)

    # 初始化 MCP 工具层
    if settings.MCP_ENABLED:
        mcp_client = MCPClient(mode="local", server_url=settings.MCP_SERVER_URL)
        await mcp_client.init()
        # 注册 MCP Server 工具
        mcp_client.register_server(drug_db)
        mcp_client.register_server(file_reader)
        mcp_client.register_server(scheduler)
        # patient_history 需要 SessionService 注入
        patient_history.set_session_service(session_service)
        mcp_client.register_server(patient_history)
        logger.info("MCP 工具层已初始化，已注册 %d 个工具", len(mcp_client._tools))
    else:
        mcp_client = None
        logger.info("MCP 工具层已禁用")

    # 初始化业务服务
    llm_service = LLMService()
    rag_retriever = MedicalRAGRetriever()
    consultation_graph = ConsultationGraph(llm_service, rag_retriever, mcp_client=mcp_client)

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

    # 关闭 MCP 客户端
    if mcp_client:
        await mcp_client.close()


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
