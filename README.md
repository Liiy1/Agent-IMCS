# Agent-IMCS 智能医疗问诊系统

基于 FastAPI + LangGraph + RAG 的 AI 医疗问诊后端服务。

## 技术栈

- **FastAPI** — 高性能异步 API 服务
- **LangGraph** — 多 Agent 问诊工作流编排（4 节点状态机）
- **ChromaDB** — 向量数据库，RAG 混合检索（向量 + 中文分词关键词 RRF 融合）
- **SQLAlchemy** (async) + MySQL — 会话状态持久化
- **Redis** — 会话缓存（优雅降级，不可用时自动切 MySQL-only）
- **LLM** — DeepSeek / OpenAI 双模型兼容

## 快速开始

### 前置条件

- Python 3.12+
- MySQL 8.0+ / Docker
- Redis 7+ / Docker

### 1. 配置环境

```bash
cd backend
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 等配置
```

### 2. 启动基础设施

```bash
docker compose up mysql redis -d
```

### 3. 安装依赖并启动

```bash
cd backend
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. 初始化知识库

```bash
cd backend
PYTHONPATH=. python scripts/seed_knowledge.py
```

### 5. 测试

```bash
# 使用 Python（不要用 Git Bash curl，见下方编码说明）
python -c "
import json, urllib.request
data = json.dumps({'session_id':'test','message':'头痛三天','user_token':'test'}).encode()
r = urllib.request.urlopen(urllib.request.Request('http://localhost:8000/api/v1/consultation/consult', data=data, headers={'Content-Type':'application/json'}))
print(json.loads(r.read()))
"
```

> **⚠️ Windows 用户注意**：Git Bash 的 `curl` 默认使用 cp1252 编码，发送中文会导致乱码。请使用 Python 或 PowerShell 测试 API。

## 项目结构

```
Agent-IMCS/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # 环境配置
│   │   ├── database.py          # 异步数据库引擎
│   │   ├── models.py            # ORM 模型
│   │   ├── agent/               # LangGraph 工作流
│   │   │   ├── state.py         # 状态定义
│   │   │   ├── graph.py         # 图编排（4 节点）
│   │   │   └── nodes.py         # 节点实现
│   │   ├── rag/
│   │   │   └── retriever.py     # ChromaDB + jieba 中文检索
│   │   ├── routers/
│   │   │   ├── consultation.py  # 问诊 API
│   │   │   └── report.py        # 报告导出 API
│   │   └── services/
│   │       ├── llm_service.py   # LLM 调用封装
│   │       ├── redis_cache.py   # Redis 缓存
│   │       └── session_service.py # 会话持久化
│   ├── scripts/
│   │   └── seed_knowledge.py    # 知识库种子
│   ├── tests/                   # 测试
│   ├── .env.example
│   └── requirements.txt
├── docker-compose.yml           # MySQL + Redis + 后端编排
├── .gitignore
└── README.md
```

## LangGraph 工作流

```
用户输入 → [collect] → [retrieve] → [analyze] → [generate_report] → 输出
                │                                            │
                └─ 症状不足 → 追问结束 ←───────────────────┘
```

1. **collect** — LLM 提取症状 + 病史，⩾3 个症状则继续，否则追问
2. **retrieve** — 向量检索 + jieba 中文关键词检索 → RRF 融合 → top-5
3. **analyze** — LLM 诊断分析、紧急分级、推荐科室
4. **generate_report** — 结构化 Markdown 问诊报告

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/consultation/consult` | POST | 问诊对话（支持多轮） |
| `/api/v1/consultation/history` | GET | 获取会话历史 |
| `/api/v1/report/export` | GET | 导出问诊报告 |

## 环境变量

参见 `backend/.env.example`。

## License

MIT
