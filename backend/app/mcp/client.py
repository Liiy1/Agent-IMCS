"""轻量 MCP 客户端

支持两种运行模式：
- local（默认）：同进程调用注册的 handler 函数
- remote：通过 httpx 发送 JSON-RPC 请求到 MCP Server（预留）

用法：
    client = MCPClient()
    client.register_server(drug_db_module)
    result = await client.call_tool("get_drug_info", {"drug_name": "阿司匹林"})
"""

import logging
import httpx
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 客户端 — 管理工具注册与调用"""

    def __init__(self, mode: str = "local", server_url: str = ""):
        """
        Args:
            mode: "local" 或 "remote"
            server_url: remote 模式下的 MCP Server 地址
        """
        self.mode = mode
        self.server_url = server_url
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._session_service = None  # 由外部注入

    # ── 生命周期 ─────────────────────────────────────

    async def init(self):
        """初始化（remote 模式时创建 HTTP 客户端）"""
        if self.mode == "remote":
            self._http_client = httpx.AsyncClient(timeout=30)
            logger.info("MCPClient remote mode: server=%s", self.server_url)
        else:
            logger.info("MCPClient local mode: 共 0 个工具注册")

    async def close(self):
        """清理资源"""
        if self._http_client:
            await self._http_client.aclose()

    def set_session_service(self, service):
        """注入 SessionService（供 patient_history 工具使用）"""
        self._session_service = service

    # ── 工具注册 ─────────────────────────────────────

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        input_schema: Optional[dict] = None,
    ):
        """注册单个工具

        Args:
            name: 工具名称
            handler: async (name, arguments) -> dict
            description: 工具描述
            input_schema: JSON Schema 格式的输入描述
        """
        self._tools[name] = {
            "handler": handler,
            "description": description,
            "input_schema": input_schema or {},
        }
        logger.debug("MCPClient 注册工具: %s", name)

    def register_server(self, server_module):
        """批量注册一个 Server 模块的所有工具

        模块需暴露:
        - TOOLS: List[Dict]  — 每个元素含 name, description, input_schema
        - async handle_tool(name, arguments) -> Dict
        """
        for tool_def in getattr(server_module, "TOOLS", []):
            name = tool_def.get("name")
            if not name:
                continue
            self._tools[name] = {
                "handler": server_module.handle_tool,
                "description": tool_def.get("description", ""),
                "input_schema": tool_def.get("input_schema", {}),
            }
            logger.info("MCPClient 注册 Server 工具: %s", name)

    # ── 工具查询与调用 ─────────────────────────────────

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具（MCP 兼容格式）"""
        return [
            {
                "name": name,
                "description": info["description"],
                "input_schema": info["input_schema"],
            }
            for name, info in self._tools.items()
        ]

    async def has_tool(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具

        Args:
            name: 工具名称
            arguments: 参数字典

        Returns:
            {"content": Any, "is_error": bool}
            调用失败时返回 is_error=True + error 信息，不抛出异常
        """
        if name not in self._tools:
            return {
                "content": f"未知工具: {name}",
                "is_error": True,
            }

        if self.mode == "remote":
            return await self._call_remote(name, arguments)

        return await self._call_local(name, arguments)

    async def _call_local(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """本地模式：直接调用 handler"""
        handler = self._tools[name]["handler"]
        try:
            result = await handler(name, arguments)
            return {"content": result, "is_error": False}
        except Exception as e:
            logger.warning("MCP 工具调用失败 [%s]: %s", name, e, exc_info=True)
            return {
                "content": {
                    "error": str(e),
                    "message": f"工具「{name}」调用失败，请稍后重试。",
                },
                "is_error": True,
            }

    async def _call_remote(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """远程模式：JSON-RPC over HTTP（预留）"""
        if not self._http_client:
            return {"content": "MCPClient 未初始化", "is_error": True}

        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
            "id": 1,
        }
        try:
            resp = await self._http_client.post(
                self.server_url.rstrip("/") + "/mcp",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return {"content": data.get("result", {}), "is_error": False}
        except Exception as e:
            logger.warning("MCP 远程调用失败 [%s]: %s", name, e)
            return {"content": {"error": str(e)}, "is_error": True}
