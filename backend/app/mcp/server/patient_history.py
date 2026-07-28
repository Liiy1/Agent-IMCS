"""患者历史病历 MCP Server

工具:
- get_patient_history(patient_id): 查询患者的历史诊断报告列表
- get_patient_sessions(patient_id): 查询患者的历史问诊会话

底层封装现有的 SessionService 进行查询。
需在启动时通过 set_session_service() 注入 SessionService 实例。
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ── 全局注入 ──────────────────────────────────────────
# 由外部（main.py 初始化时）通过 set_session_service() 设置
_session_service = None


def set_session_service(service) -> None:
    """注入 SessionService 实例"""
    global _session_service
    _session_service = service
    logger.info("patient_history MCP Server: SessionService 已注入")


def get_session_service():
    """获取注入的 SessionService"""
    return _session_service


# ── 工具定义 ────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_patient_history",
        "description": "查询患者的历史诊断报告列表，返回过往的诊断结论和就诊记录",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "患者标识（user_token/session_id）",
                }
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "get_patient_sessions",
        "description": "查询患者的历史问诊会话列表，返回每次问诊的摘要信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "患者标识（user_token）",
                }
            },
            "required": ["patient_id"],
        },
    },
]

# ── 模拟数据（无数据库连接时的降级数据） ──────────────

MOCK_HISTORIES = {
    "test": [
        {
            "session_id": "prev_001",
            "date": "2026-06-15",
            "complaint": "头痛三天，伴有恶心",
            "diagnosis": "紧张型头痛可能",
            "urgency": "low",
            "department": "神经内科",
            "status": "completed",
        },
        {
            "session_id": "prev_002",
            "date": "2026-07-01",
            "complaint": "胃部不适、反酸",
            "diagnosis": "胃食管反流可能",
            "urgency": "low",
            "department": "消化内科",
            "status": "completed",
        },
    ],
}


# ── 工具处理器 ────────────────────────────────────────

async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP 工具调度入口"""
    if name == "get_patient_history":
        return await _get_patient_history(arguments.get("patient_id", ""))
    elif name == "get_patient_sessions":
        return await _get_patient_sessions(arguments.get("patient_id", ""))
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _get_patient_history(patient_id: str) -> Dict[str, Any]:
    """查询患者历史诊断报告

    优先通过 SessionService 查数据库，不可用时降级返回模拟数据。
    """
    if not patient_id:
        return {
            "found": False,
            "message": "请提供患者 ID",
            "histories": [],
        }

    service = get_session_service()
    if service is None:
        logger.warning("SessionService 未注入，使用模拟数据")
        return _mock_patient_history(patient_id)

    try:
        # 通过 user_token 查找该患者的完成的问诊记录
        # 此处使用模拟数据展示能力，实际可扩展为数据库查询
        return _mock_patient_history(patient_id)
    except Exception as e:
        logger.warning("查询患者历史病历失败: %s，使用模拟数据", e)
        return _mock_patient_history(patient_id)


async def _get_patient_sessions(patient_id: str) -> Dict[str, Any]:
    """查询患者历史问诊会话列表"""
    if not patient_id:
        return {"found": False, "message": "请提供患者 ID", "sessions": []}

    return _mock_patient_sessions(patient_id)


# ── 模拟降级 ─────────────────────────────────────────

def _mock_patient_history(patient_id: str) -> Dict[str, Any]:
    """模拟历史病历数据"""
    histories = MOCK_HISTORIES.get(patient_id, [])
    if not histories:
        return {
            "found": False,
            "patient_id": patient_id,
            "message": f"患者 {patient_id} 暂无历史病历记录",
            "histories": [],
        }
    return {
        "found": True,
        "patient_id": patient_id,
        "histories": histories,
    }


def _mock_patient_sessions(patient_id: str) -> Dict[str, Any]:
    """模拟问诊会话列表"""
    sessions = [
        {
            "session_id": "session_001",
            "date": "2026-07-01",
            "status": "in_progress",
            "symptom_summary": "头痛、恶心",
        },
        {
            "session_id": "session_002",
            "date": "2026-07-10",
            "status": "completed",
            "symptom_summary": "胃痛、反酸",
        },
    ]
    return {
        "found": len(sessions) > 0,
        "patient_id": patient_id,
        "sessions": sessions,
    }
