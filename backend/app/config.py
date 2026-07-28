from typing import List, Optional, Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI智能问诊系统"
    DEBUG: bool = False
    SECRET_KEY: str = "sk-2da0b333816849719475b76bdbf04ad8"

    # 数据库配置
    DATABASE_URL: str = "mysql://user:pass@localhost:3306/medical_consult"
    REDIS_URL: str = "redis://localhost:6379/0"
    HF_ENDPOINT: str = "https://huggingface.co" 

    # 大模型配置
    LLM_PROVIDER: Literal["deepseek", "openai"] = "deepseek"
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4-turbo-preview"

    # 向量库配置
    VECTOR_DB_PATH: str = "./data/chroma"

    # JWT配置
    JWT_SECRET_KEY: str = "jwt-secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440

    # Redis 缓存
    REDIS_CACHE_TTL: int = 1800  # 会话状态缓存 TTL（秒）

    # 跨域白名单
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # MCP 工具层配置
    MCP_ENABLED: bool = True              # MCP 工具层总开关
    MCP_SERVER_URL: str = "http://localhost:8001"  # MCP Server 地址（远程模式预留）

    # 反思机制配置
    REFLECTION_ENABLED: bool = True       # 总开关
    REFLECTION_MAX_ROUNDS: int = 2        # 最大分析轮次（含初版，即最多 1 次精炼）
    REFLECTION_PASS_THRESHOLD: int = 3    # 最低通过分数（1-5 评分制）

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
import logging
logging.getLogger(__name__).debug("DEEPSEEK_KEY_PREFIX: %s...", repr(settings.DEEPSEEK_API_KEY)[:15])