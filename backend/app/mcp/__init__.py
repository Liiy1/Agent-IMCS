"""MCP (Model Context Protocol) 工具层

提供药品查询、病历查询、文件读取、科室排班等工具，
供 LangGraph 工作流中的 LLM 按需调用。

设计原则：
- 所有 Server 当前以本地函数调用方式运行（同进程），
  客户端接口兼容 MCP JSON-RPC 协议，为后续独立进程部署预留。
- 所有工具调用失败时优雅降级，不阻塞工作流。
"""

__version__ = "0.1.0"
