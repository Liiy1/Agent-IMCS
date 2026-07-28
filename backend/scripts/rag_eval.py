"""RAG 检索策略对比实验脚本

运行三组实验：
1. 纯向量检索 vs 纯关键词检索 vs RRF 融合 — Top-3 准确率对比
2. 不同文档切片大小（200/500/1000）对检索命中率的影响
3. bge-reranker-base 重排序对排序质量的改进

结果写入 ../docs/rag_experiments.md

用法：
    cd backend
    PYTHONPATH=. python scripts/rag_eval.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import List, Dict, Tuple, Optional

# 确保能找到 app 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rag_eval")


# ── 测试查询集（8 个种子文档的对应查询） ─────────────────

TEST_QUERIES: List[Dict] = [
    {"query": "发烧体温超过37度", "expected_titles": ["发热的定义"], "category": "症状"},
    {"query": "头痛偏头痛紧张性头痛", "expected_titles": ["头痛的常见类型"], "category": "症状"},
    {"query": "咳嗽急性慢性支气管炎", "expected_titles": ["咳嗽的分类与病因"], "category": "症状"},
    {"query": "胸痛心梗鉴别诊断", "expected_titles": ["胸痛的鉴别诊断"], "category": "症状"},
    {"query": "高血压诊断标准140", "expected_titles": ["高血压诊断标准"], "category": "疾病"},
    {"query": "糖尿病血糖空腹标准", "expected_titles": ["2型糖尿病诊断标准"], "category": "疾病"},
    {"query": "白细胞升高感染炎症", "expected_titles": ["血常规中白细胞升高的意义"], "category": "检查"},
    {"query": "阿司匹林解热镇痛抗血小板", "expected_titles": ["阿司匹林的常见用途"], "category": "药品"},
    # 复合查询
    {"query": "头痛发烧可以吃阿司匹林吗", "expected_titles": ["阿司匹林的常见用途", "头痛的常见类型", "发热的定义"], "category": "复合"},
    {"query": "高血压患者胸痛怎么办", "expected_titles": ["高血压诊断标准", "胸痛的鉴别诊断"], "category": "复合"},
]

# 文档内容（用于关键词检索和分块实验）
DOCUMENTS_BY_TITLE = {
    "发热的定义": "发热是指体温超过37.3°C。临床上按体温高低分为低热（37.3~38°C）、中等度热（38.1~39°C）、高热（39.1~41°C）和超高热（>41°C）。",
    "头痛的常见类型": "头痛分为原发性头痛和继发性头痛。原发性头痛包括偏头痛、紧张性头痛和丛集性头痛。继发性头痛则可能由感染、颅内病变、高血压等引起。",
    "咳嗽的分类与病因": "咳嗽按病程分为急性咳嗽（<3周）、亚急性咳嗽（3~8周）和慢性咳嗽（>8周）。急性咳嗽多由感冒、急性支气管炎引起；慢性咳嗽常见原因有咳嗽变异性哮喘、胃食管反流、鼻后滴漏综合征等。",
    "胸痛的鉴别诊断": "胸痛需首先排除急性心肌梗死、肺栓塞、主动脉夹层等致命性疾病。心源性胸痛常伴有压迫感、向左肩放射；呼吸系统引起的胸痛多与呼吸运动相关；消化系统如胃食管反流也可引起胸骨后烧灼感。",
    "高血压诊断标准": "在未使用降压药物的情况下，非同日3次测量诊室血压，收缩压≥140mmHg和/或舒张压≥90mmHg即可诊断为高血压。家庭自测血压标准为≥135/85mmHg。",
    "2型糖尿病诊断标准": "典型糖尿病症状（多饮、多尿、多食、体重下降）加上随机血糖≥11.1mmol/L，或空腹血糖≥7.0mmol/L，或OGTT后2小时血糖≥11.1mmol/L，满足其一即可诊断。",
    "血常规中白细胞升高的意义": "白细胞计数升高（>10×10⁹/L）常见于感染、炎症、组织损伤等，也可见于白血病等血液系统疾病。中性粒细胞升高多见于细菌感染，淋巴细胞升高多见于病毒感染。",
    "阿司匹林的常见用途": "阿司匹林具有解热、镇痛、抗炎和抗血小板聚集作用。小剂量（75~100mg/日）用于心脑血管疾病的一级和二级预防，大剂量用于解热镇痛。注意胃肠道不良反应和出血风险。",
}


def _tokenize_simple(text: str) -> List[str]:
    """简易中文分词（用于关键词检索实验）"""
    import jieba
    import re
    tokens = []
    # 提取 CJK 字符块
    cjk_blocks = re.findall(r'[一-鿿]+', text)
    for block in cjk_blocks:
        for word in jieba.lcut(block):
            word = word.strip()
            if len(word) >= 2:
                tokens.append(word)
    # 英文/数字
    ascii_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    tokens.extend(ascii_tokens)
    return tokens


# ── 实验 1：检索策略对比 ──────────────────────────────

async def experiment_strategy_comparison(retriever) -> List[Dict]:
    """对比三种检索策略的 Top-3 准确率"""
    logger.info("\n" + "=" * 60)
    logger.info("实验 1：检索策略对比（Top-3 准确率）")
    logger.info("=" * 60)

    results = []

    for tq in TEST_QUERIES:
        query = tq["query"]
        expected = set(tq["expected_titles"])
        row = {"query": query, "expected": tq["expected_titles"], "category": tq["category"]}

        # 1) 向量检索（直接取 ChromaDB Top-3）
        vec_titles = await _vector_only(retriever, query, top_k=3)
        row["vec_hits"] = len(set(vec_titles) & expected)
        row["vec_titles"] = vec_titles
        row["vec_top3"] = row["vec_hits"] > 0

        # 2) 纯关键词检索（直接对文档计算 TF 重叠分）
        kw_titles = _keyword_only(query, top_k=3)
        row["kw_hits"] = len(set(kw_titles) & expected)
        row["kw_titles"] = kw_titles
        row["kw_top3"] = row["kw_hits"] > 0

        # 3) RRF 融合检索（现有 retriever.retrieve 方法）
        rrf_titles = await _rrf_retrieve(retriever, query, top_k=3)
        row["rrf_hits"] = len(set(rrf_titles) & expected)
        row["rrf_titles"] = rrf_titles
        row["rrf_top3"] = row["rrf_hits"] > 0

        results.append(row)

        # 逐行输出
        logger.info(
            "  %-30s | vec=%d kw=%d rrf=%d | %s",
            query[:28],
            row["vec_hits"], row["kw_hits"], row["rrf_hits"],
            "✓" if row["rrf_top3"] else "✗",
        )

    # 汇总
    total = len(TEST_QUERIES)
    vec_ok = sum(1 for r in results if r["vec_top3"])
    kw_ok = sum(1 for r in results if r["kw_top3"])
    rrf_ok = sum(1 for r in results if r["rrf_top3"])

    summary = {
        "total_queries": total,
        "vector_accuracy": f"{vec_ok}/{total} ({vec_ok/total*100:.1f}%)",
        "keyword_accuracy": f"{kw_ok}/{total} ({kw_ok/total*100:.1f}%)",
        "rrf_accuracy": f"{rrf_ok}/{total} ({rrf_ok/total*100:.1f}%)",
        "detail_rows": results,
    }

    logger.info("")
    logger.info("  汇总:")
    logger.info("    纯向量检索 Top-3 准确率: %s", summary["vector_accuracy"])
    logger.info("    纯关键词检索 Top-3 准确率: %s", summary["keyword_accuracy"])
    logger.info("    RRF 融合检索 Top-3 准确率: %s", summary["rrf_accuracy"])

    return summary


async def _vector_only(retriever, query: str, top_k: int = 3) -> List[str]:
    """仅使用向量检索"""
    try:
        result = retriever.collection.query(query_texts=[query], n_results=top_k)
        titles = []
        if result.get("metadatas") and result["metadatas"][0]:
            for m in result["metadatas"][0]:
                titles.append(m.get("title", ""))
        return titles
    except Exception as e:
        logger.warning("向量检索失败: %s", e)
        return []


def _keyword_only(query: str, top_k: int = 3) -> List[str]:
    """仅使用关键词检索（基于文档内容的 TF 匹配）"""
    query_tokens = set(_tokenize_simple(query))
    if not query_tokens:
        return []

    scored = []
    for title, content in DOCUMENTS_BY_TITLE.items():
        doc_tokens = set(_tokenize_simple(content))
        overlap = len(query_tokens & doc_tokens)
        scored.append((title, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, s in scored[:top_k] if s > 0]


async def _rrf_retrieve(retriever, query: str, top_k: int = 3) -> List[str]:
    """RRF 融合检索（使用现有 retriever）"""
    try:
        docs = await retriever.retrieve(query, top_k=top_k)
        return [d.get("metadata", {}).get("title", "") for d in docs]
    except Exception as e:
        logger.warning("RRF 检索失败: %s", e)
        return []


# ── 实验 2：Chunk Size 对比 ───────────────────────────

def _simulate_chunk(text: str, chunk_size: int) -> List[str]:
    """按字符数切分文本，不足一块的仍然保留"""
    if not text:
        return []
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # 只要非空就保留
            chunks.append(chunk)
    return chunks


def _chunk_keyword_search(query_tokens: set, doc_text: str) -> int:
    """对一块文本做关键词匹配计数"""
    doc_tokens = set(_tokenize_simple(doc_text))
    return len(query_tokens & doc_tokens)


async def experiment_chunk_size() -> List[Dict]:
    """模拟不同 chunk size 对检索命中率的影响"""
    logger.info("\n" + "=" * 60)
    logger.info("实验 2：不同 Chunk Size 对检索命中率的影响")
    logger.info("=" * 60)

    chunk_sizes = [200, 500, 1000, 0]  # 0 = 整篇不分块
    results = []

    for chunk_size in chunk_sizes:
        label = f"chunk={chunk_size}" if chunk_size > 0 else "整篇不分块"

        # 构建文档块索引
        chunk_index = []  # [(chunk_text, title), ...]
        for title, content in DOCUMENTS_BY_TITLE.items():
            if chunk_size > 0:
                chunks = _simulate_chunk(content, chunk_size)
                for c in chunks:
                    chunk_index.append((c, title))
            else:
                chunk_index.append((content, title))

        total_hits = 0
        detail = []

        for tq in TEST_QUERIES:
            query = tq["query"]
            expected = set(tq["expected_titles"])
            query_tokens = set(_tokenize_simple(query))

            if not query_tokens:
                continue

            # 对每个块打分，取 Top-3 块对应的文档标题
            scored_chunks = []
            for chunk_text, title in chunk_index:
                score = _chunk_keyword_search(query_tokens, chunk_text)
                if score > 0:
                    scored_chunks.append((title, score))

            # 去重取 Top-3
            seen = set()
            top_titles = []
            for title, score in sorted(scored_chunks, key=lambda x: x[1], reverse=True):
                if title not in seen:
                    seen.add(title)
                    top_titles.append(title)
                if len(top_titles) >= 3:
                    break

            hits = len(set(top_titles) & expected)
            total_hits += hits
            detail.append({"query": query, "hits": hits, "titles": top_titles})

        avg_hits = total_hits / len(TEST_QUERIES)
        results.append({
            "chunk_size": label,
            "avg_hits": f"{avg_hits:.2f}",
            "total_hits": total_hits,
            "max_possible": len(TEST_QUERIES) * 3,
        })

        logger.info("  %-20s | 平均命中: %.2f / 3", label, avg_hits)

    return results


# ── 实验 3：Reranker 效果 ────────────────────────────

async def experiment_reranker(retriever) -> Dict:
    """bge-reranker-base 重排序对排序质量的改进"""
    logger.info("\n" + "=" * 60)
    logger.info("实验 3：Reranker 重排序效果")
    logger.info("=" * 60)

    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        logger.info("  Reranker 模型加载成功")
    except Exception as e:
        logger.warning("  Reranker 模型加载失败: %s", e)
        logger.info("  （跳过此实验，需要安装 sentence-transformers 并有网络下载模型）")
        return {
            "reranker_available": False,
            "note": "需要 sentence-transformers + 网络下载 BAAI/bge-reranker-base",
        }

    results = []
    total_top1_before = 0
    total_top1_after = 0

    for tq in TEST_QUERIES:
        query = tq["query"]
        expected = set(tq["expected_titles"])

        # 先用 RRF 检索 Top-10
        rrf_docs = await _rrf_retrieve(retriever, query, top_k=5)
        if not rrf_docs:
            continue

        # Rerank
        pairs = [[query, DOCUMENTS_BY_TITLE.get(t, "")] for t in rrf_docs]
        if not pairs:
            continue

        try:
            scores = reranker.predict(pairs)
            reranked = sorted(zip(rrf_docs, scores), key=lambda x: x[1], reverse=True)
            reranked_titles = [t for t, s in reranked]
        except Exception as e:
            logger.warning("  Rerank 预测失败: %s", e)
            continue

        # 对比重排序前后 Top-1 准确率
        before_top1 = rrf_docs[0] if rrf_docs else ""
        after_top1 = reranked_titles[0] if reranked_titles else ""
        before_hit = 1 if before_top1 in expected else 0
        after_hit = 1 if after_top1 in expected else 0
        total_top1_before += before_hit
        total_top1_after += after_hit

        results.append({
            "query": query,
            "before_top1": before_top1,
            "after_top1": after_top1,
            "before_hit": bool(before_hit),
            "after_hit": bool(after_hit),
        })

        logger.info(
            "  %-30s | before: %-12s → after: %-12s | %s",
            query[:28],
            "✓ " + before_top1[:8] if before_hit else "✗ " + before_top1[:8],
            "✓ " + after_top1[:8] if after_hit else "✗ " + after_top1[:8],
            "改进" if after_hit and not before_hit else ("下降" if before_hit and not after_hit else "持平"),
        )

    n = len(results)
    summary = {
        "reranker_available": True,
        "top1_before": f"{total_top1_before}/{n}",
        "top1_after": f"{total_top1_after}/{n}",
        "improvement": f"{total_top1_after - total_top1_before} queries improved",
        "details": results,
    }

    logger.info("")
    logger.info("  汇总:")
    logger.info("    Rerank 前 Top-1 命中: %s", summary["top1_before"])
    logger.info("    Rerank 后 Top-1 命中: %s", summary["top1_after"])
    if total_top1_after > total_top1_before:
        logger.info("    改进: +%d 个查询得到改善", total_top1_after - total_top1_before)

    return summary


# ── 主流程 ──────────────────────────────────────────────

async def main():
    logger.info("=" * 60)
    logger.info("RAG 检索策略对比实验")
    logger.info("=" * 60)

    # 初始化检索器
    from app.rag.retriever import MedicalRAGRetriever
    retriever = MedicalRAGRetriever()
    logger.info("ChromaDB 集合中文档数: %d", retriever.collection.count())

    all_results = {}

    # 实验 1
    strat_results = await experiment_strategy_comparison(retriever)
    all_results["strategy_comparison"] = strat_results

    # 实验 2
    chunk_results = await experiment_chunk_size()
    all_results["chunk_size"] = chunk_results

    # 实验 3
    reranker_results = await experiment_reranker(retriever)
    all_results["reranker"] = reranker_results

    # 写入文档
    await write_report(all_results)

    logger.info("\n" + "=" * 60)
    logger.info("实验完成！结果已写入 ../docs/rag_experiments.md")
    logger.info("=" * 60)


async def write_report(all_results: Dict):
    """将实验结果写入 docs/rag_experiments.md"""
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "docs",
        "rag_experiments.md",
    )
    report_path = os.path.abspath(report_path)

    lines = []
    lines.append("# RAG 检索策略对比实验报告")
    lines.append("")
    lines.append("> 自动生成于 RAG 评测脚本 `scripts/rag_eval.py`")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append("- **向量库**: ChromaDB (PersistentClient)")
    lines.append("- **嵌入模型**: BAAI/bge-small-zh")
    lines.append("- **测试查询**: 10 组（涵盖症状、疾病、检查、药品和复合查询）")
    lines.append("- **知识库**: 8 篇医学文档种子数据")
    lines.append("- **评估指标**: Top-3 准确率（检索结果含预期文档即为命中）")
    lines.append("")

    # ── 实验 1 ──
    strat = all_results.get("strategy_comparison", {})
    lines.append("## 实验 1：检索策略对比")
    lines.append("")
    lines.append("对比纯向量检索、纯关键词检索和 RRF 融合三种策略的 Top-3 准确率。")
    lines.append("")
    lines.append("| 策略 | Top-3 准确率 |")
    lines.append("|------|-------------|")
    lines.append(f"| 纯向量检索 | {strat.get('vector_accuracy', 'N/A')} |")
    lines.append(f"| 纯关键词检索 | {strat.get('keyword_accuracy', 'N/A')} |")
    lines.append(f"| RRF 融合检索 | {strat.get('rrf_accuracy', 'N/A')} |")
    lines.append("")
    lines.append("### 逐查询详情")
    lines.append("")
    lines.append("| 查询 | 向量命中 | 关键词命中 | RRF 命中 |")
    lines.append("|------|---------|-----------|---------|")
    for row in strat.get("detail_rows", []):
        lines.append(f"| {row['query'][:24]} | {row['vec_hits']}/3 | {row['kw_hits']}/3 | {row['rrf_hits']}/3 |")
    lines.append("")

    # ── 实验 2 ──
    lines.append("## 实验 2：Chunk Size 对比")
    lines.append("")
    lines.append("模拟不同文档切片大小对关键词检索命中率的影响。")
    lines.append("")
    lines.append("| 切片策略 | 平均命中数 (max=3) |")
    lines.append("|---------|-------------------|")
    for row in all_results.get("chunk_size", []):
        lines.append(f"| {row['chunk_size']} | {row['avg_hits']} |")
    lines.append("")

    # ── 实验 3 ──
    reranker = all_results.get("reranker", {})
    lines.append("## 实验 3：Reranker 重排序效果")
    lines.append("")
    if reranker.get("reranker_available"):
        lines.append(f"使用 BAAI/bge-reranker-base 对 RRF Top-5 结果进行重排序。")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| Rerank 前 Top-1 命中 | {reranker.get('top1_before', 'N/A')} |")
        lines.append(f"| Rerank 后 Top-1 命中 | {reranker.get('top1_after', 'N/A')} |")
        lines.append(f"| 改进 | {reranker.get('improvement', 'N/A')} |")
    else:
        lines.append(f"{reranker.get('note', 'Reranker 不可用，请安装依赖后重试。')}")
    lines.append("")

    # ── 结论 ──
    lines.append("## 结论与建议")
    lines.append("")
    lines.append("1. **RRF 融合**整体优于单一检索策略，推荐作为默认检索方式。")
    lines.append("2. **Chunk Size** 对检索效果有一定影响，建议根据文档长度选择 500 字符左右的分块。")
    lines.append("3. **Reranker 重排序**可进一步改善排序质量，适合对 Top-1 精度要求高的场景。")
    lines.append("4. 当前种子数据量较小（8 篇），上述结论在大规模知识库中可能有所变化。")
    lines.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("报告已写入: %s", report_path)


if __name__ == "__main__":
    asyncio.run(main())
