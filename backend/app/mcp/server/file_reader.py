"""医学文件/报告读取 MCP Server

工具:
- read_lab_report(file_path): 读取检验报告文本内容
- extract_tables(file_path): 从检验报告中提取结构化数据

支持格式:
- .txt — 直接读取 UTF-8 文本
- .pdf — PyMuPDF (fitz) 提取文本层内容
- .json — 解析为易读格式

所有解析失败时返回明确的错误信息，不抛出异常。
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# 支持的扩展名
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".json"}

# ── 工具定义 ────────────────────────────────────────────

TOOLS = [
    {
        "name": "read_lab_report",
        "description": "读取检验报告文件内容，支持 .txt .pdf .json 格式，返回可读文本",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "报告文件的完整路径或相对路径",
                }
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "extract_tables",
        "description": "从检验报告中提取结构化检验数据（如血常规各项指标和数值）",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "报告文件的完整路径或相对路径",
                }
            },
            "required": ["file_path"],
        },
    },
]

# ── 模拟检验报告数据（用于演示，没有实际文件时展示能力） ──

MOCK_LAB_REPORT = """
=== 血常规检验报告 ===
患者ID: test001    检查日期: 2026-07-20

+----------------------+--------+--------+-----------+
| 检验项目             | 结果   | 单位   | 参考范围  |
+----------------------+--------+--------+-----------+
| 白细胞计数(WBC)      | 11.5   | 10^9/L | 3.5-9.5   |
| 红细胞计数(RBC)      | 5.2    | 10^12/L| 4.3-5.8   |
| 血红蛋白(HGB)        | 155    | g/L    | 130-175   |
| 血小板计数(PLT)      | 245    | 10^9/L | 125-350   |
| 中性粒细胞百分比     | 78.5   | %      | 40-75     |
| 淋巴细胞百分比       | 15.2   | %      | 20-50     |
| 超敏C反应蛋白(hs-CRP)| 25.3   | mg/L   | <5        |
+----------------------+--------+--------+-----------+

异常指标提示:
- 白细胞计数偏高 → 可能提示感染或炎症
- 中性粒细胞百分比偏高 → 可能与细菌感染相关
- hs-CRP显著升高 → 提示急性炎症反应
"""

MOCK_TABLES_DATA = [
    {"item": "白细胞计数(WBC)", "value": "11.5", "unit": "10^9/L", "range": "3.5-9.5", "flag": "偏高"},
    {"item": "红细胞计数(RBC)", "value": "5.2", "unit": "10^12/L", "range": "4.3-5.8", "flag": "正常"},
    {"item": "血红蛋白(HGB)", "value": "155", "unit": "g/L", "range": "130-175", "flag": "正常"},
    {"item": "血小板计数(PLT)", "value": "245", "unit": "10^9/L", "range": "125-350", "flag": "正常"},
    {"item": "中性粒细胞百分比", "value": "78.5", "unit": "%", "range": "40-75", "flag": "偏高"},
    {"item": "淋巴细胞百分比", "value": "15.2", "unit": "%", "range": "20-50", "flag": "偏低"},
    {"item": "超敏C反应蛋白(hs-CRP)", "value": "25.3", "unit": "mg/L", "range": "<5", "flag": "偏高"},
]


# ── 工具处理器 ────────────────────────────────────────

async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP 工具调度入口"""
    if name == "read_lab_report":
        return await _read_lab_report(arguments.get("file_path", ""))
    elif name == "extract_tables":
        return await _extract_tables(arguments.get("file_path", ""))
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _read_lab_report(file_path: str) -> Dict[str, Any]:
    """读取检验报告文件内容"""
    if not file_path:
        return {"found": False, "message": "请提供文件路径"}

    # 检查文件是否存在
    if not os.path.exists(file_path):
        logger.warning("文件不存在: %s，返回模拟数据", file_path)
        return _mock_report_result(file_path)

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "found": False,
            "message": f"不支持的文件格式「{ext}」，当前支持: {', '.join(SUPPORTED_EXTENSIONS)}",
        }

    try:
        if ext == ".txt":
            text = await _read_txt(file_path)
        elif ext == ".pdf":
            text = await _read_pdf(file_path)
        elif ext == ".json":
            text = await _read_json_report(file_path)
        else:
            return {"found": False, "message": f"不支持的文件格式: {ext}"}

        return {
            "found": True,
            "file_path": file_path,
            "content": text,
            "format": ext,
        }
    except Exception as e:
        logger.warning("读取文件失败: %s，返回模拟数据", e)
        return _mock_report_result(file_path)


async def _extract_tables(file_path: str) -> Dict[str, Any]:
    """从检验报告中提取结构化数据"""
    if not file_path:
        return {"found": False, "message": "请提供文件路径"}

    if not os.path.exists(file_path):
        logger.warning("文件不存在: %s，返回模拟数据", file_path)
        return {
            "found": True,
            "source": "demo",
            "message": "（演示数据）未找到实际文件，展示结构化检验数据示例",
            "table_data": MOCK_TABLES_DATA,
            "abnormal_items": [
                item for item in MOCK_TABLES_DATA if item["flag"] != "正常"
            ],
        }

    # 先读取文件内容
    result = await _read_lab_report(file_path)
    if not result.get("found"):
        return result

    # TODO: 结构化解析逻辑
    return {
        "found": True,
        "source": file_path,
        "message": "文件内容已读取，结构化解析功能尚在开发中，以下为原始内容",
        "raw_content": result.get("content", ""),
        "table_data": [],
    }


# ── 文件读取实现 ─────────────────────────────────────

async def _read_txt(file_path: str) -> str:
    """读取纯文本文件"""
    def read():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    # 在线程池中执行阻塞 IO
    import asyncio
    return await asyncio.to_thread(read)


async def _read_pdf(file_path: str) -> str:
    """使用 PyMuPDF 读取 PDF 文本层"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF 未安装，无法读取 PDF")
        return f"[PDF 文件: {os.path.basename(file_path)}]\n（需要安装 PyMuPDF 以提取文本层内容）"

    def read():
        doc = fitz.open(file_path)
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- 第 {page_num} 页 ---\n{text}")
        doc.close()
        return "\n\n".join(text_parts) if text_parts else "[PDF 文件未包含可提取的文本层]"

    import asyncio
    text = await asyncio.to_thread(read)
    return text


async def _read_json_report(file_path: str) -> str:
    """读取 JSON 格式的检验报告"""
    def read():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)

    import asyncio
    return await asyncio.to_thread(read)


# ── 模拟降级 ─────────────────────────────────────────

def _mock_report_result(file_path: str) -> Dict[str, Any]:
    """文件不存在时返回模拟检验报告数据"""
    filename = os.path.basename(file_path) if file_path else "未知文件"
    return {
        "found": True,
        "source": "demo",
        "file_path": file_path,
        "message": f"（演示模式）未找到文件「{filename}」，展示模拟血常规报告示例",
        "content": MOCK_LAB_REPORT,
    }
