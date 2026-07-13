"""RAG 检索器单元测试

测试纯函数部分：_tokenize、_keyword_search、_rrf_fusion
无需 ChromaDB 或 embedding 模型，可独立运行。
"""

import sys
sys.path.insert(0, "backend")

from app.rag.retriever import MedicalRAGRetriever


def make_rag():
    """不初始化的空实例，仅测试实例方法"""
    return MedicalRAGRetriever.__new__(MedicalRAGRetriever)


# ── _tokenize ──────────────────────────────────────────────

def test_tokenize_english():
    rag = make_rag()
    tokens = rag._tokenize("hello world test")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens


def test_tokenize_chinese():
    rag = make_rag()
    tokens = rag._tokenize("头痛发烧咳嗽")
    assert "头痛" in tokens or "发烧" in tokens or "咳嗽" in tokens
    # jieba 会正确切分出这些词
    assert len(tokens) >= 2


def test_tokenize_mixed():
    rag = make_rag()
    tokens = rag._tokenize("头痛 fever 38.5°C")
    # 中文词
    assert "头痛" in tokens
    # 英文/数字
    assert "fever" in tokens
    assert "38" in tokens  # jieba 可能在 "38.5" 上拆分


def test_tokenize_empty():
    rag = make_rag()
    tokens = rag._tokenize("")
    assert tokens == set()


def test_tokenize_short_chars():
    """单字应被过滤掉以减少噪音"""
    rag = make_rag()
    tokens = rag._tokenize("的一了是")
    assert len(tokens) == 0


# ── _keyword_search ───────────────────────────────────────

def test_keyword_search_basic():
    rag = make_rag()
    candidates = {
        "ids": [["med_0", "med_1"]],
        "documents": [["发热是指体温超过37.3°C", "头痛分为原发性和继发性头痛"]],
        "metadatas": [[{"title": "发热"}, {"title": "头痛"}]],
    }
    results = rag._keyword_search("头痛发热", candidates)
    assert len(results) == 2
    # 两条文档都匹配了 "头痛" 或 "发热"，所以都有分
    assert results[0]["keyword_score"] > 0
    assert results[1]["keyword_score"] > 0


def test_keyword_search_no_match():
    rag = make_rag()
    candidates = {
        "ids": [["med_0"]],
        "documents": [["阿司匹林具有解热镇痛作用"]],
        "metadatas": [[{"title": "阿司匹林"}]],
    }
    results = rag._keyword_search("白内障", candidates)
    assert len(results) == 1
    assert results[0]["keyword_score"] == 0  # 无匹配


# ── _rrf_fusion ────────────────────────────────────────────

def test_rrf_fusion_combined():
    """向量检索和关键词检索结果应被融合"""
    rag = make_rag()
    vec_results = {
        "ids": [["med_0", "med_1", "med_2"]],
        "documents": [["doc A", "doc B", "doc C"]],
        "metadatas": [[{}, {}, {}]],
    }
    kw_results = [
        {"id": "med_1", "content": "doc B", "metadata": {}, "keyword_score": 3},
        {"id": "med_0", "content": "doc A", "metadata": {}, "keyword_score": 1},
    ]
    fused = rag._rrf_fusion(vec_results, kw_results, top_k=2)
    assert len(fused) == 2
    # 两个结果都应该有分数
    assert all(r["score"] > 0 for r in fused)


def test_rrf_fusion_top_k():
    """top_k 参数应被尊重"""
    rag = make_rag()
    vec_results = {
        "ids": [["med_0", "med_1", "med_2", "med_3"]],
        "documents": [["A", "B", "C", "D"]],
        "metadatas": [[{}, {}, {}, {}]],
    }
    kw_results = [
        {"id": "med_0", "content": "A", "keyword_score": 5},
        {"id": "med_1", "content": "B", "keyword_score": 3},
    ]
    fused = rag._rrf_fusion(vec_results, kw_results, top_k=3)
    assert len(fused) == 3


def test_rrf_fusion_order():
    """分数更高的文档应排在前面"""
    rag = make_rag()
    vec_results = {
        "ids": [["med_0", "med_1"]],
        "documents": [["低分文档", "高分文档"]],
        "metadatas": [[{}, {}]],
    }
    kw_results = [
        {"id": "med_1", "content": "高分文档", "keyword_score": 10},
        {"id": "med_0", "content": "低分文档", "keyword_score": 1},
    ]
    fused = rag._rrf_fusion(vec_results, kw_results, top_k=2)
    assert fused[0]["id"] == "med_1"  # 高分排前
    assert fused[1]["id"] == "med_0"
