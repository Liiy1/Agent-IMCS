# RAG 检索工作流程

> 对应 `backend/app/rag/retriever.py`

本文档描述 RAG 混合检索管线的执行流程、关键算法和当前局限。

---

## 一、检索执行流程（时序）

```
用户输入："头痛三天，有发烧"
          │
          ▼
  ┌─────────────────────────────────────┐
  │ ① 构建查询文本                      │
  │    query = "头痛三天，有发烧"         │
  │          + str(state["medical_history"]) │
  └─────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────┐
  │ ② 向量检索                           │
  │    ChromaDB.query(query_texts)      │
  │    n_results = 10 (top_k * 2)       │
  │    ─────────────────────────────    │
  │    查询向量 → bge-small-zh 编码     │
  │    → ChromaDB 语义相似度搜索        │
  │    → 返回 Top-10 文档 + 元数据      │
  └─────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────┐
  │ ③ jieba 中文分词                    │
  │    query → jieba.lcut()            │
  │    过滤：长度 ≥ 2 的词              │
  │    ─────────────────────────────    │
  │    例："头痛三天有发烧"              │
  │    → ["头痛", "三天", "发烧"]       │
  │    （过滤掉单字 "有"）               │
  └─────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────┐
  │ ④ 关键词重排序                      │
  │    在向量检索返回的 10 篇文档上做：   │
  │    ─────────────────────────────    │
  │    score = 关键词子串命中数           │
  │           + 分词重叠数 × 2          │
  │    ─────────────────────────────    │
  │    按得分排序 → 取 Top-10           │
  └─────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────┐
  │ ⑤ RRF 融合排序                      │
  │    RRF(k=60) 公式：                 │
  │    score(d) = 1/(60+rank_vec)       │
  │             + 1/(60+rank_kw)        │
  │    ─────────────────────────────    │
  │    向量排名 + 关键词排名 → 综合分    │
  │    排序 → 取 Top-5                  │
  └─────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────┐
  │ ⑥ 可选：Reranker 重排序             │
  │    bge-reranker-base 语义重排序     │
  │    ─────────────────────────────    │
  │    对 RRF Top-5 做语义打分          │
  │    将最相关的文档推至 Top-1          │
  │    （需额外加载模型，默认不启用）     │
  └─────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────┐
  │ ⑦ 输出到 analyze 节点               │
  │    rag_context =                    │
  │      doc[0].content + "\n"          │
  │    + doc[1].content + ...           │
  │    （5 篇文档拼接为 LLM 上下文）      │
  │                                     │
  │    + MCP 工具结果（如有调用）         │ ← MCP 工具层补充
  └─────────────────────────────────────┘
```

---

## 二、关键代码对照

```python
# retriever.py:39-53
async def retrieve(self, query, top_k=5):
    # 1. 向量检索（从全库召回）
    vector_results = self.collection.query(
        query_texts=[query],
        n_results=top_k * 2          # 取 2 倍供重排序筛选
    )

    # 2. 关键词检索（在向量结果上重排序）
    keyword_results = self._keyword_search(query, vector_results)

    # 3. RRF 融合排序
    fused = self._rrf_fusion(vector_results, keyword_results, top_k)

    return fused  # 最终 Top-5


# retriever.py:71-93
def _keyword_search(self, query, candidates):
    keywords = self._tokenize(query)  # jieba 分词

    for doc in candidates['documents'][0]:
        doc_tokens = self._tokenize(doc)
        # 打分：关键词子串命中 + 分词重叠 * 2
        score = sum(1 for kw in keywords if kw in doc)
        score += len(keywords & doc_tokens) * 2

    return sorted(scored, reverse=True)[:10]


# _rrf_fusion 算法（retriever.py:95-121）
def _rrf_fusion(self, vec_results, kw_results, top_k, k=60):
    """Reciprocal Rank Fusion 倒数排名融合"""
    scores = {}
    # 向量排名得分
    for rank, doc_id in enumerate(vec_results['ids'][0][:20]):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    # 关键词排名得分
    for rank, r in enumerate(kw_results[:20]):
        scores[r["id"]] = scores.get(r["id"], 0) + 1 / (k + rank + 1)
    # 按总分排序截取 top_k
    ...
```

---

## 三、Reranker 重排序（可选）

```
bge-reranker-base  CrossEncoder 模型，在 RRF 融合之后执行：

  RRF Top-5 → [query + doc1, query + doc2, ...] 对
           → bge-reranker-base 语义相关性打分
           → 按相关性重新排序 → 最终 Top-5

加载方式：
  from sentence_transformers import CrossEncoder
  reranker = CrossEncoder("BAAI/bge-reranker-base")
  scores = reranker.predict(pairs)

效果说明：
  - 在小规模测试集上（基线已达 100%），无额外提升
  - 在噪声场景下可将正确文档推至 Top-1
  - 适合生产环境中的开放域检索
```

---

## 四、实验结论

参见 [rag_experiments.md](rag_experiments.md) 完整报告，关键结论：

| 策略 | Top-3 准确率 |
|------|-------------|
| 纯向量检索 | 10/10 (100%) |
| 纯关键词检索 | 10/10 (100%) |
| RRF 融合检索 | 10/10 (100%) |
| RRF + Reranker | 持平（基线已达上限） |

---

## 五、MCP 工具层对 RAG 的补充

```
MCP 工具             RAG 检索                    LLM 分析
┌──────────┐        ┌───────────┐             ┌──────────┐
│ 药品查询  │───────→│           │             │          │
│ 病历查询  │───────→│ RAG上下文  │────────────→│ 诊断分析  │
│ 文件读取  │───────→│           │             │          │
│ 科室排班  │───────→│           │             │          │
└──────────┘        └───────────┘             └──────────┘
      ↑                    ↑
  实时结构化数据         语义检索知识
  （药品库/DB）         （向量文档库）

两者互补：RAG 提供医学知识语义检索，MCP 提供实时结构化数据查询。
```

---

## 六、当前局限

```
向量检索路径                        关键词路径
┌──────────────┐              ┌──────────────┐
│ ChromaDB     │─────────────→│ 仅在向量结果  │
│ 全库搜索     │  10篇候选文档  │  上重排序     │
│              │              │  非全库检索   │
└──────────────┘              └──────────────┘
                                ↑
                            不是独立召回路径
                        （当前是重排序，非双路召回）

改进方向：
  1. 全库关键词索引（如 Elasticsearch），实现真正的双路召回
  2. 引入 bge-reranker-base 提升排序质量
  3. 动态 chunk size 根据文档长度自适应
```
