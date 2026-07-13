from typing import List, Dict, Set
import os
import re
import jieba
import chromadb
from chromadb.utils import embedding_functions
from app.config import settings


class MedicalRAGRetriever:
    """医学知识库RAG检索器"""

    def __init__(self):
        # 使用持久化 ChromaDB，确保种子数据跨进程可用
        persist_dir = settings.VECTOR_DB_PATH
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 优先使用环境变量指定的本地模型路径，否则使用 HuggingFace 模型名
        model_path = os.getenv("EMBEDDING_MODEL_PATH", "")
        if not model_path or not os.path.isdir(model_path):
            # 回退到项目根目录下的 bge-small-zh 子模块
            local_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "bge-small-zh"
            )
            if os.path.isdir(local_path):
                model_path = local_path
            else:
                model_path = "BAAI/bge-small-zh"  # 最后回退到在线模型名

        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_path
            )
        )

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """混合检索：向量检索 + 关键词检索 + RRF重排序"""
        # 1. 向量检索
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2
        )

        # 2. 关键词检索(BM25简易模拟)
        keyword_results = self._keyword_search(query, vector_results)

        # 3. RRF融合排序
        fused = self._rrf_fusion(vector_results, keyword_results, top_k)

        return fused

    def _tokenize(self, text: str) -> Set[str]:
        """对文本进行分词，兼容中文和英文"""
        # 英文按空格/标点分词，中文用 jieba 分词
        tokens = set()
        # 提取所有连续的 CJK 字符块
        cjk_blocks = re.findall(r'[一-鿿]+', text)
        for block in cjk_blocks:
            for word in jieba.lcut(block):
                word = word.strip()
                if len(word) >= 2:  # 过滤单字，减少噪音
                    tokens.add(word)
        # 英文/数字 token（小写）
        ascii_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        tokens.update(ascii_tokens)
        return tokens

    def _keyword_search(self, query: str, candidates: Dict) -> List[Dict]:
        """纯关键词文本匹配打分（支持中文分词）"""
        keywords = self._tokenize(query)
        # 如果分词后没有有效关键词，回退到字符级 bigram
        if not keywords:
            keywords = set(query.lower().replace(' ', ''))
        scored = []

        for i, doc in enumerate(candidates['documents'][0]):
            doc_tokens = self._tokenize(doc)
            # 计算关键词命中数
            score = sum(1 for kw in keywords if kw in doc)
            # 额外加分：分词匹配
            token_overlap = len(keywords & doc_tokens)
            score += token_overlap * 2
            scored.append({
                "id": candidates['ids'][0][i],
                "content": doc,
                "metadata": candidates['metadatas'][0][i] if candidates.get('metadatas') else {},
                "keyword_score": score
            })

        return sorted(scored, key=lambda x: x["keyword_score"], reverse=True)[:10]

    def _rrf_fusion(self, vec_results: Dict, kw_results: List, top_k: int, k: int = 60):
        """RRF倒数排名融合算法"""
        scores = {}

        # 向量检索结果打分
        for i, doc_id in enumerate(vec_results['ids'][0][:20]):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + i + 1)

        # 关键词检索结果打分
        for i, r in enumerate(kw_results[:20]):
            doc_id = r["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + i + 1)

        # 按总分排序截取topK
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        results = []
        for doc_id in sorted_ids:
            if doc_id in vec_results['ids'][0]:
                idx = vec_results['ids'][0].index(doc_id)
                results.append({
                    "id": doc_id,
                    "content": vec_results['documents'][0][idx],
                    "metadata": vec_results['metadatas'][0][idx] if vec_results.get('metadatas') else {},
                    "score": scores[doc_id]
                })
        return results

    async def add_medical_knowledge(self, title: str, content: str, category: str):
        """向向量库新增医学文档"""
        # 从已有记录数确定新 ID，避免重复遍历全量数据
        count = self.collection.count()
        self.collection.add(
            ids=[f"med_{count}"],
            documents=[content],
            metadatas=[{"title": title, "category": category}]
        )