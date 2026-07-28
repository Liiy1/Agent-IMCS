# Agent-IMCS 智能医疗问诊系统

基于 FastAPI + LangGraph + RAG + MCP 的 AI 医疗问诊后端服务。

## 技术栈

- **FastAPI** — 高性能异步 API 服务
- **LangGraph** — 多 Agent 问诊工作流编排（5 节点状态机 + 反思闭环）
- **ChromaDB** — 向量数据库，RAG 混合检索（向量 + 中文分词关键词 RRF 融合）
- **SQLAlchemy** (async) + MySQL — 会话状态持久化
- **Redis** — 会话缓存（优雅降级，不可用时自动切 MySQL-only）
- **LLM** — DeepSeek / OpenAI 双模型兼容
- **MCP** — Model Context Protocol 工具层（药品查询、病历查询、文件读取、科室排班）

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
│   │   ├── config.py            # 环境配置（含 MCP 设置）
│   │   ├── database.py          # 异步数据库引擎
│   │   ├── models.py            # ORM 模型
│   │   ├── agent/               # LangGraph 工作流
│   │   │   ├── state.py         # 状态定义
│   │   │   ├── graph.py         # 图编排（5 节点 + 反思闭环）
│   │   │   └── nodes.py         # 节点实现（含 MCP 工具调用子步骤）
│   │   ├── mcp/                 # MCP 工具层 ← 新增
│   │   │   ├── client.py        # MCP 客户端（本地/远程双模式）
│   │   │   └── server/
│   │   │       ├── drug_db.py         # 药品数据库查询工具
│   │   │       ├── patient_history.py # 患者历史病历查询工具
│   │   │       ├── file_reader.py     # 医学文件读取工具
│   │   │       └── scheduler.py       # 科室排班查询工具
│   │   ├── rag/
│   │   │   └── retriever.py     # ChromaDB + jieba 中文检索
│   │   ├── routers/
│   │   │   ├── consultation.py  # 问诊 API
│   │   │   └── report.py        # 报告导出 API
│   │   └── services/
│   │       ├── llm_service.py   # LLM 调用封装（含工具决策）
│   │       ├── redis_cache.py   # Redis 缓存
│   │       └── session_service.py # 会话持久化
│   ├── scripts/
│   │   ├── seed_knowledge.py    # 知识库种子
│   │   └── rag_eval.py          # RAG 对比实验脚本
│   ├── tests/                   # 测试
│   ├── .env / .env.example
│   └── requirements.txt
├── docs/
│   ├── rag_experiments.md       # RAG 检索策略实验报告
│   └── ...                      # 架构文档
├── docker-compose.yml           # MySQL + Redis + 后端编排
├── bge-small-zh/                # 本地嵌入模型
└── README.md
```

## LangGraph 工作流

```
用户输入 →
  [collect] ─┬─ (MCP 工具调用子步骤: 药品/病历/文件查询)
             │    LLM 判断是否需要外部数据 ─→ 是 → 调用 MCP → 注入上下文
             │                                  └→ 否 → 继续
             │
             ├─ 症状不足 → [ask] → 追问 → END
             │
             └─ 症状充分 → [retrieve] → [analyze] ─┬─ (MCP 工具调用子步骤)
                                                   │    药品/报告数据合并到 RAG 上下文
                                                   │
                                                   └─ → [reflect] ─┬─ 通过 → [generate_report] → END
                                                                     │
                                                                     └─ 不通过 → 精炼 → ⟳ analyze
                                                                          (最多 2 轮，超限强制出报告)
```

### 5 节点职责

| 节点 | 职责 | MCP 集成 |
|------|------|----------|
| `collect` | LLM 提取症状 + 病史，追问决策 | 关键词预检 + LLM 工具决策，调用药品/病历/文件工具 |
| `retrieve` | RAG 混合检索医学知识库 | — |
| `analyze` | LLM 诊断分析、紧急分级、推荐科室 | 分析前检测药品/报告需求，结果合并到 RAG 上下文 |
| `reflect` | LLM-as-Judge 自评分析质量（1-5 分） | — |
| `generate_report` | 生成结构化 Markdown 报告 | — |

### 反思闭环

- `reflect` 节点从 **一致性、完整性、医学准确性** 三维度评分
- 低于阈值（默认 3/5）且轮次未耗尽 → 带反馈返回 `analyze` 精炼
- 轮次耗尽仍不达标 → 强制出报告，追加「⚠️ 建议人工复核」

## MCP 工具层

项目集成了 4 个 MCP 工具，供 LLM 在问诊过程中按需调用：

| 工具 | 功能 | 数据源 |
|------|------|--------|
| `get_drug_info` | 查询药品适应症、副作用、禁忌 | SQLite 本地药典（50 种常用药） |
| `check_drug_interaction` | 检查两种药品的相互作用 | SQLite（15 条已知相互作用） |
| `get_patient_history` | 查询患者历史诊断记录 | SessionService / Mock 数据 |
| `read_lab_report` | 读取检验报告内容 | 本地文件（.txt/.pdf） |
| `extract_tables` | 提取结构化检验数据 | 本地文件（.txt/.pdf） |
| `get_department_schedule` | 查询科室排班信息 | Mock 数据 |
| `check_available_slots` | 查询挂号余号 | Mock 数据 |

**调用方式**：`collect` 节点中 LLM 通过 `tool_calls` 字段请求工具调用，节点执行后结果注入上下文重新提取。所有工具调用失败时优雅降级。

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/consultation/consult` | POST | 问诊对话（支持多轮 + MCP 工具调用） |
| `/api/v1/consultation/history` | GET | 获取会话历史 |
| `/api/v1/report/export` | GET | 导出问诊报告 |

## RAG 实验

参见 [docs/rag_experiments.md](docs/rag_experiments.md) 获取以下实验结果：

1. **检索策略对比**：纯向量 vs 纯关键词 vs RRF 融合的 Top-3 准确率
2. **Chunk Size 影响**：200/500/1000 字符分块对检索命中率的影响
3. **Reranker 效果**：bge-reranker-base 重排序的改进

运行实验：
```bash
cd backend
PYTHONPATH=. python scripts/rag_eval.py
```

## 环境变量

参见 `backend/.env.example`。

## License

MIT
