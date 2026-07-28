"""MCP Server 模块

每个子模块暴露两个约定：
- TOOLS: List[Dict]  — 工具定义列表（name, description, input_schema）
- async handle_tool(name, arguments) -> Dict  — 工具调度入口
"""
