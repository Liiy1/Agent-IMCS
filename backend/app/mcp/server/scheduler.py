"""科室排班/挂号可用性查询 MCP Server

工具:
- get_department_schedule(dept, date): 查询指定科室某日的排班信息
- check_available_slots(dept, date): 查询指定科室某日的可用余号

当前使用硬编码 Mock 数据，接口设计兼容对接真实 HIS 系统。
"""

import logging
from typing import Dict, Any, Optional
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# ── 工具定义 ────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_department_schedule",
        "description": "查询指定科室在某日的医生排班信息，包括出诊医生、科室位置等",
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "科室名称，如'神经内科'、'消化内科'",
                },
                "date": {
                    "type": "string",
                    "description": "日期（YYYY-MM-DD格式），默认当天",
                },
            },
            "required": ["department"],
        },
    },
    {
        "name": "check_available_slots",
        "description": "查询指定科室在某日的可用挂号余号数量",
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "科室名称",
                },
                "date": {
                    "type": "string",
                    "description": "日期（YYYY-MM-DD格式），默认当天",
                },
            },
            "required": ["department"],
        },
    },
]

# ── Mock 排班数据 ────────────────────────────────────────

MOCK_SCHEDULES = {
    "神经内科": {
        "location": "门诊楼3层 神经内科诊区",
        "doctors": [
            {"name": "张主任", "title": "主任医师", "time_slots": ["上午", "下午"]},
            {"name": "李医生", "title": "副主任医师", "time_slots": ["上午"]},
        ],
    },
    "消化内科": {
        "location": "门诊楼2层 消化内科诊区",
        "doctors": [
            {"name": "王主任", "title": "主任医师", "time_slots": ["上午"]},
            {"name": "赵医生", "title": "主治医师", "time_slots": ["上午", "下午"]},
        ],
    },
    "呼吸内科": {
        "location": "门诊楼2层 呼吸内科诊区",
        "doctors": [
            {"name": "刘主任", "title": "主任医师", "time_slots": ["上午", "下午"]},
            {"name": "陈医生", "title": "主治医师", "time_slots": ["下午"]},
        ],
    },
    "心血管内科": {
        "location": "门诊楼3层 心血管内科诊区",
        "doctors": [
            {"name": "周主任", "title": "主任医师", "time_slots": ["上午"]},
            {"name": "吴医生", "title": "副主任医师", "time_slots": ["上午", "下午"]},
        ],
    },
    "感染科": {
        "location": "门诊楼1层 感染科诊区",
        "doctors": [
            {"name": "孙主任", "title": "主任医师", "time_slots": ["上午"]},
        ],
    },
    "急诊科": {
        "location": "急诊楼1层",
        "doctors": [
            {"name": "急诊值班医生", "title": "主治医师", "time_slots": ["全天24小时"]},
        ],
    },
}

MOCK_AVAILABLE_SLOTS = {
    "神经内科": {"morning": 12, "afternoon": 8, "total": 20},
    "消化内科": {"morning": 5, "afternoon": 15, "total": 20},
    "呼吸内科": {"morning": 3, "afternoon": 10, "total": 13},
    "心血管内科": {"morning": 8, "afternoon": 6, "total": 14},
    "感染科": {"morning": 10, "afternoon": 0, "total": 10},
    "急诊科": {"morning": 30, "afternoon": 30, "total": 60},
}

# ── 工具处理器 ────────────────────────────────────────

async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP 工具调度入口"""
    department = arguments.get("department", "")
    query_date = arguments.get("date", date.today().isoformat())

    if name == "get_department_schedule":
        return _get_schedule(department, query_date)
    elif name == "check_available_slots":
        return _check_slots(department, query_date)
    else:
        raise ValueError(f"Unknown tool: {name}")


def _get_schedule(department: str, query_date: str) -> Dict[str, Any]:
    """获取科室排班信息"""
    if department not in MOCK_SCHEDULES:
        # 模糊匹配
        matched = [d for d in MOCK_SCHEDULES if d in department or department in d]
        if matched:
            department = matched[0]
        else:
            return {
                "found": False,
                "message": f"未找到「{department}」的排班信息，请确认科室名称",
                "available_departments": list(MOCK_SCHEDULES.keys()),
            }

    info = MOCK_SCHEDULES[department]
    return {
        "found": True,
        "department": department,
        "date": query_date,
        "location": info["location"],
        "doctors": info["doctors"],
        "note": "此为模拟数据，实际排班以医院当日公示为准",
    }


def _check_slots(department: str, query_date: str) -> Dict[str, Any]:
    """查询挂号余号"""
    if department not in MOCK_AVAILABLE_SLOTS:
        matched = [d for d in MOCK_AVAILABLE_SLOTS if d in department or department in d]
        if matched:
            department = matched[0]
        else:
            return {
                "found": False,
                "message": f"未找到「{department}」的挂号信息",
                "available_departments": list(MOCK_AVAILABLE_SLOTS.keys()),
            }

    slots = MOCK_AVAILABLE_SLOTS[department]

    # 根据日期动态调整余号（模拟上午减少）
    import random
    seed = hash(f"{department}_{query_date}")
    rng = random.Random(seed)
    morning = max(0, slots["morning"] - rng.randint(0, min(5, slots["morning"])))
    afternoon = max(0, slots["afternoon"] - rng.randint(0, min(3, slots["afternoon"])))

    return {
        "found": True,
        "department": department,
        "date": query_date,
        "morning_slots": morning,
        "afternoon_slots": afternoon,
        "total_slots": morning + afternoon,
        "note": "此为模拟数据，实际号源以医院当日公示为准",
    }
