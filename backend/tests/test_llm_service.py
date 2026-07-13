"""LLM 服务单元测试

测试纯函数 _clean_json_response，不依赖网络或 API 密钥。
"""

import sys
sys.path.insert(0, "backend")

import json
from app.services.llm_service import LLMService


def make_llm():
    return LLMService.__new__(LLMService)


# ── _clean_json_response ──────────────────────────────────

def test_clean_plain_json():
    llm = make_llm()
    raw = '{"symptoms": ["头痛"]}'
    assert llm._clean_json_response(raw) == raw


def test_clean_json_block():
    llm = make_llm()
    raw = '```json\n{"symptoms": ["头痛"]}\n```'
    assert llm._clean_json_response(raw) == '{"symptoms": ["头痛"]}'


def test_clean_markdown_block():
    llm = make_llm()
    raw = '```\n{"symptoms": []}\n```'
    assert llm._clean_json_response(raw) == '{"symptoms": []}'


def test_clean_with_whitespace():
    llm = make_llm()
    raw = '  {"symptoms": ["发烧"]}  '
    assert llm._clean_json_response(raw) == '{"symptoms": ["发烧"]}'


def test_clean_then_parse():
    """清理后的输出应是合法 JSON"""
    llm = make_llm()
    raw = '```json\n{"symptoms": ["头痛", "发烧"], "history": [], "next_ask": ""}\n```'
    cleaned = llm._clean_json_response(raw)
    parsed = json.loads(cleaned)
    assert parsed["symptoms"] == ["头痛", "发烧"]
    assert parsed["history"] == []


def test_clean_no_op_for_plain_text():
    llm = make_llm()
    raw = '{"key": "value"}'
    assert llm._clean_json_response(raw) == raw
