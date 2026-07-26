# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv (Windows — run from backend/)
cd backend && source venv/Scripts/activate

# Run backend (dev with hot-reload)
uvicorn app.main:app --reload --port 8000

# Run tests
pip install pytest pytest-asyncio
python -m pytest tests/ -v

# Seed knowledge base (run after fresh ChromaDB)
PYTHONPATH=. python scripts/seed_knowledge.py

# Start infrastructure
docker compose up mysql redis -d

# Full stack
docker compose up --build -d

# Test API (use Python, NOT curl on Windows — see encoding note)
python -c "
import json, urllib.request
data = json.dumps({'session_id':'t','message':'头痛三天','user_token':'t'}).encode()
r = urllib.request.urlopen(urllib.request.Request('http://localhost:8000/api/v1/consultation/consult', data=data, headers={'Content-Type':'application/json'}))
print(json.loads(r.read()))
"
```

## ⚠️ Windows 编码注意事项

**不要在 Git Bash 中用 `curl` 发送含中文的 JSON！** Git Bash 在 Windows 上默认使用 cp1252 编码发送请求体，中文会被转成 `??????` 发给后端。

- ✅ 测试 API 用 **Python urllib/httpx** 或 **PowerShell 的 `Invoke-RestMethod`**
- ✅ 或者在 Git Bash 中先 `export PYTHONIOENCODING=utf-8` 后用 Python 脚本发请求

```powershell
# PowerShell (推荐)
$body = @{session_id="test"; message="头痛三天"; user_token="test"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8001/api/v1/consultation/consult -Method Post -Body $body -ContentType "application/json"
```

## Project Overview

**Agent-IMCS** — AI-powered medical consultation system. Patients describe symptoms, the system runs a multi-agent LangGraph workflow with RAG-enhanced medical knowledge retrieval, and produces a structured diagnosis report with urgency triage and department recommendation.

## Architecture

### Backend — FastAPI + LangGraph + RAG

```
backend/
├── app/
│   ├── main.py              # FastAPI app entry, lifespan init, global singletons
│   ├── config.py            # pydantic-settings (env/DEEPSEEK_API_KEY, DB_URL, etc.)
│   ├── database.py          # Async SQLAlchemy engine + sessionmaker
│   ├── models.py            # ORM: Conversation, Message, ConversationState
│   ├── agent/
│   │   ├── state.py         # ConsultationState TypedDict (LangGraph state schema)
│   │   ├── graph.py         # ConsultationGraph — 5-node with reflection loop
│   │   └── nodes.py         # collect → retrieve → analyze → reflect → generate_report
│   ├── rag/
│   │   └── retriever.py     # ChromaDB (PersistentClient) + bge-small-zh + RRF fusion
│   ├── routers/
│   │   ├── consultation.py  # POST /consult, GET /history
│   │   └── report.py        # GET /export
│   └── services/
│       ├── llm_service.py   # DeepSeek/OpenAI client, JSON prompt wrappers
│       ├── redis_cache.py   # Async Redis with graceful degradation
│       └── session_service.py  # State persistence (Redis cache + MySQL, optimistic locking)
├── scripts/
│   └── seed_knowledge.py    # 8 documents: symptoms, diseases, tests, drugs
├── requirements.txt
└── Dockerfile
```

### LangGraph Consultation Workflow (含反思闭环)

The core is a **5-node state graph with a reflection loop** in `app/agent/`:

1. **collect** — LLM extracts symptoms + medical history from user message, enriches accumulated state
2. **retrieve** — Hybrid RAG: vector + jieba keyword search → RRF fusion → top-5 medical docs
3. **analyze** — LLM produces diagnosis, urgency level (low/medium/high/emergency), recommended department
4. **reflect** — LLM-as-judge: scores analysis quality (1–5) across consistency, completeness, and medical accuracy
5. **generate_report** — LLM writes structured markdown report

**条件边 × 2：**
- `collect` 后：若 <3 症状且无病史 → `next_action: "ask"`（追问用户，结束本轮）；否则 → 继续检索
- `reflect` 后：若质量评分 ≥ 阈值（默认 3 分）→ 通过，生成报告；若不达标且轮次未耗尽 → 回 `analyze` 精炼（带上反思反馈）；轮次用尽仍不达标 → 强制出报告，追加"⚠️ 建议人工复核"声明

```
collect → (条件边: 症状是否充分?)
    ├── ask → END
    └── continue → retrieve → analyze → reflect → (条件边: 质量是否达标?)
                                        ↑             ├── refine ────┘
                                        │             └── continue → generate_report → END
                                        └── 最多 REFLECTION_MAX_ROUNDS 轮 ──┘
```

**配置项**（`config.py`）：
- `REFLECTION_ENABLED` — 总开关
- `REFLECTION_MAX_ROUNDS` — 最大精炼轮次（默认 2）
- `REFLECTION_PASS_THRESHOLD` — 最低通过分数（默认 3/5）

**故障降级**：LLM 反射调用失败时保守返回通过值，不阻塞工作流。

### Multi-Turn Conversation

State persists across turns via **SessionService**:

- Redis (fast cache, TTL=1800s) with MySQL (async + asyncmy) as backing store
- Optimistic locking (`version` column) prevents concurrent writes from overwriting
- `user_token` guards session ownership on history/report access

### RAG Knowledge Retrieval

- ChromaDB **PersistentClient** (not the ephemeral Client) at `./data/chroma/`
- Embedding: `BAAI/bge-small-zh` (local path override via `EMBEDDING_MODEL_PATH` env var)
- Dual-path retrieval: vector similarity + simple keyword overlap scoring
- RRF (Reciprocal Rank Fusion) merges both result sets
- **⚠️ 局限**: 关键词检索按空格分词（`query.lower().split()`），对中文无效。现仅靠向量检索工作。

### LLM Integration

- Configurable provider: DeepSeek (default) or OpenAI, selected via `LLM_PROVIDER` env var
- `temperature=0.3` for deterministic medical output
- Three prompt templates: symptom extraction, diagnosis analysis, report generation
- Response cleaner strips markdown fences (`_clean_json_response`)

### Key Environment Variables

| Variable           | Default                    | Purpose                                               |
| ------------------ | -------------------------- | ----------------------------------------------------- |
| `LLM_PROVIDER`     | `deepseek`                 | `deepseek` or `openai`                                |
| `DEEPSEEK_API_KEY` | —                          | DeepSeek API key (required for deepseek provider)     |
| `DATABASE_URL`     | `mysql+asyncmy://...`      | Async MySQL connection string                         |
| `REDIS_URL`        | `redis://localhost:6379/0` | Redis connection (gracefully degrades if unavailable) |
| `HF_ENDPOINT`      | `https://hf-mirror.com`    | HuggingFace mirror for model downloads                |
| `EMBEDDING_MODEL_PATH` | —                      | Local path to bge-small-zh model (avoids download)    |

### Infrastructure

- Docker Compose: MySQL 8.4, Redis 8.0, backend
- Backend Dockerfile pre-downloads bge-small-zh into the image
- Health endpoint: `GET /health` returns `{"status": "healthy"}``

## Known Issues

### 1. `redis_cache.py` — `delete_state` 未使用
`delete_state` 方法已定义但没有任何地方调用。保留作后续扩展用。

## Tests

测试文件在 `backend/tests/` 下，涵盖纯函数逻辑和 mock 驱动的节点测试。

### 运行

```bash
cd backend && source venv/Scripts/activate
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

### 测试文件说明

| 文件 | 测试内容 | 依赖 |
|------|----------|------|
| `test_retriever.py` | `_tokenize`、`_keyword_search`、`_rrf_fusion` | 无（纯函数） |
| `test_llm_service.py` | `_clean_json_response` | 无（纯函数） |
| `test_graph.py` | conditional edges、节点状态流转（mock LLM/RAG） | pytest-asyncio |
| `test_api.py` | `/health`、参数校验 | MySQL+Redis（默认 skip） |

### 设计说明

- **纯函数测试**（retriever / llm_service）不依赖任何外部服务，用 `__new__` 绕过 `__init__` 中的 ChromaDB/模型加载
- **节点测试**（graph）通过 `conftest.py` 的 `mock_llm_service` / `mock_rag_retriever` 隔离外部依赖
- **API 测试**（test_api.py）依赖真实 MySQL+Redis，默认 `@pytest.mark.skip`

## Fixed Issues (2026-07-13)

| # | 问题 | 修复方式 |
|---|------|----------|
| 1 | `consultation.py` `save_state` 重复调用 | 删除了第132-155行的重复块 |
| 2 | `state.py` `collected_symptoms` 类型为 `List[Dict]` | 改为 `List[str]` |
| 3 | 关键词检索对中文无效 | 添加 jieba 中文分词，替换空格分词 |
| 4 | `config.py` 中 `EMBEDDING_MODEL` 未使用 | 已移除 |
| 5 | `report.py` 的 `format` 参数未实现 | 已移除未使用的参数 |

## Test Results (2026-07-13)

### 端到端问诊测试结果 ✅

**输入**: "我头痛三天了，还有发烧、恶心和乏力"

| 节点 | 结果 | 说明 |
|------|------|------|
| collect | ✅ 提取4个症状 | 头痛、发烧、恶心、乏力 |
| retrieve | ✅ RAG命中 | 检索到头痛分类等知识库内容 |
| analyze | ✅ 紧急度 medium | 推荐科室：神经内科/感染科 |
| generate_report | ✅ 结构化Markdown | 含主诉、诊断分析、建议检查 |

**知识库种子**: 8条文档（发热、头痛、咳嗽、胸痛、高血压、糖尿病、白细胞、阿司匹林）— 全部成功写入持久化 ChromaDB。
