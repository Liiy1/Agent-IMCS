import json
import logging
from typing import Optional, Dict, Any

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

# Redis key 前缀
STATE_KEY_PREFIX = "session"


class RedisCache:
    """异步 Redis 缓存封装，带优雅降级

    当 Redis 不可用时，所有方法静默降级为 no-op，
    系统继续以 MySQL-only 模式运行。
    """

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None
        self._available = False

    async def init(self) -> None:
        """初始化连接并检查可用性"""
        try:
            self._client = aioredis.from_url(
                settings.REDIS_URL,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            await self._client.ping()
            self._available = True
            logger.info("Redis 连接成功")
        except Exception as e:
            self._available = False
            logger.warning("Redis 不可用，降级为 MySQL-only 模式: %s", e)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._available = False

    def _state_key(self, session_id: str) -> str:
        return f"{STATE_KEY_PREFIX}:{session_id}:state"

    async def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self._available or not self._client:
            return None
        try:
            data = await self._client.get(self._state_key(session_id))
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning("Redis get_state 失败: %s", e)
        return None

    async def set_state(self, session_id: str, state: Dict[str, Any], ttl: int = 1800) -> None:
        if not self._available or not self._client:
            return
        try:
            await self._client.setex(
                self._state_key(session_id),
                ttl,
                json.dumps(state, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning("Redis set_state 失败: %s", e)

    async def delete_state(self, session_id: str) -> None:
        if not self._available or not self._client:
            return
        try:
            await self._client.delete(self._state_key(session_id))
        except Exception as e:
            logger.warning("Redis delete_state 失败: %s", e)
