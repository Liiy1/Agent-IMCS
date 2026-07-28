# LLM 调用工作流程图

> 对应 `backend/app/services/llm_service.py`

本文档梳理系统内所有 LLM 调用点的输入输出结构、调用链路和异常处理。

---

## 一、调用总览

```
一次完整问诊的 LLM 调用序列（不含追问路径）：

collect:   extract_symptom (1次) [+ generate_followup(1次) 如症状不足]
          [+ MCP 工具调用后二次提取(1次)]     ← MCP 工具层新增
retrieve:  无 LLM 调用（仅 RAG 检索）
analyze:   analyze_diagnosis (1次)
reflect:   reflect_analysis (1次)
report:    generate_report (1次)

精炼循环追加：analyze → reflect 最多重复 1 次（共 2 轮）
MCP 工具调用：collect 节点中 LLM 通过 tool_calls 字段请求工具调用
```

---

## 二、各调用点详情

### ① extract_symptom（collect 节点触发）

```
输入：用户消息 + 累积症状 + 累积病史
     （如有 MCP 预检结果，额外包含药品信息/检验报告/历史病历）

Prompt 结构：
  "从以下用户输入中提取所有症状和病史，以 JSON 格式输出。
   - symptoms: 字符串列表
   - history: 字符串列表
   - next_ask: 固定返回空字符串
   - tool_calls: （可选）如需调用外部工具时返回

   可用工具：
   - get_drug_info: 用户询问药品作用、副作用时用
   - check_drug_interaction: 用户询问两种药能否一起吃时用
   - get_patient_history: 用户提到"上次""以前"时用
   - read_lab_report: 用户提到"报告""化验单"时用

   示例："我头痛，能吃阿司匹林吗？"
   输出：{"symptoms":["头痛"],"history":[],"tool_calls":[
         {"tool":"get_drug_info","arguments":{"drug_name":"阿司匹林"}}]}"

输出（JSON）：
  {"symptoms":["头痛","发烧"],"history":[],"tool_calls":[...]}  ← tool_calls 可选

消费方：
  - collect 节点追加到 state["collected_symptoms"]
  - 如有 tool_calls → 执行 MCP 工具 → 结果注入上下文 → 二次提取
```

---

### ② analyze_diagnosis（analyze 节点触发）

```
输入：症状列表 + 病史 + 知识库上下文（rag_context）
     （精炼轮次额外传入上一版分析 + 反思反馈）
     （如 MCP 工具在 analyze 中调用，结果也合并到 rag_context）

Prompt 结构（初版）：
  "你是一名专业内科医生。请根据以下信息进行分析，
    并以 JSON 格式返回诊断结果。

    参考医学知识库：{rag_context}
    用户症状：{symptoms}
    病史：{history}

    返回 JSON：
    {"content":"诊断分析...",
     "urgency_level":"low/medium/high/emergency",
     "department":"推荐就诊科室"}"

Prompt 结构（精炼轮次，追加）：
   上一版分析：{previous_analysis}
   改进反馈（请据此改进分析）：{reflection_feedback}

输出（JSON）：
  {"content":"...","urgency_level":"medium","department":"神经内科"}

消费方：analyze 节点更新 state 三个字段
```

---

### ③ reflect_analysis（reflect 节点触发）

```
输入：症状 + 病史 + RAG 上下文 + 诊断分析 + 紧急度 + 科室

Prompt 结构：
  "你是一位资深医学专家，负责对AI生成的诊断分析进行质量评审。
    请从以下三个维度严格评估：
    1. 一致性：紧急程度是否与症状和诊断匹配？科室推荐是否合理？
    2. 完整性：是否覆盖了所有症状和可能病因？有无重要遗漏？
    3. 医学准确性：分析是否与参考医学知识一致？有无明显错误？

    参考医学知识库：{rag_context}
    患者症状：{symptoms}
    病史：{history}
    当前诊断分析：{analysis}
    当前紧急程度：{urgency}
    当前推荐科室：{dept}

    返回 JSON：
    {"score": <整数 1-5>,
     "passed": <bool, score≥3 为通过>,
     "feedback": "具体的改进建议（供分析精炼使用）",
     "critical_issues": ["严重问题列表"],
     "minor_issues": ["次要问题列表"]}"

输出（JSON）：
  {"score":3,"passed":true,"feedback":"分析基本完整，建议补充鉴别诊断。"}

消费方：
  reflect 节点决定 next_action：
    - continue → generate_report（通过）
    - refine → 回 analyze 精炼（带上 feedback）
```

---

### ④ generate_report（report 节点触发）

```
输入：诊断分析 + 紧急程度 + 推荐科室

Prompt 结构：
  "你是一名专业内科医生。请根据以下诊断分析，
    生成一份结构化的问诊报告（Markdown格式）。

    诊断分析：{analysis}
    紧急程度：{urgency}
    推荐科室：{dept}

    报告应包含：
    1. 主诉 — 总结患者主要症状
    2. 诊断分析 — 详细分析可能的病因
    3. 紧急程度 — 评估是否需要立即就医
    4. 就诊建议 — 推荐科室及注意事项
    5. 建议检查 — 推荐的相关检查项目"

输出（Markdown 文本）：结构化报告文本

消费方：report 节点写入 state["report"]，最终返回给用户

特殊逻辑：
  若轮次用尽后评分仍不达标，报告开头追加 "⚠️ 建议人工复核" 声明
```

---

### ⑤ generate_followup（collect 节点触发，症状不足时）

```
输入：当前已收集症状 + 病史 + 用户消息原文

Prompt 结构：
  "你是一位耐心、细致的医生，正在通过问诊收集患者病情。
    目前已收集信息：
    - 症状：{symptoms}
    - 病史：{history}
    - 患者刚说：{user_message}

    请判断当前缺少哪些关键信息（发病时间、部位、性质等），
    生成一句有针对性的追问。
    返回 JSON：{"question":"追问话术","missing_aspect":"信息类别","priority":"high/medium/low"}"

输出（JSON）：
  {"question":"您的头痛是持续性的还是阵发性的？","missing_aspect":"疼痛性质","priority":"high"}

消费方：collect 节点返回追问文本给用户
```

---

## 三、LLM 调用封装层

```
┌─────────────────────────────────────────────────────┐
│                   LLMService                         │
├─────────────────────────────────────────────────────┤
│                                                      │
│  _call_llm(prompt)                                   │
│    ├─ 判断 self.provider                              │
│    │                                                  │
│    ├─ deepseek → POST {DEEPSEEK_BASE_URL}/chat/completions│
│    │              model: deepseek-chat                 │
│    │              temperature: 0.3                    │
│    │                                                  │
│    ├─ openai   → POST {OPENAI_BASE_URL}/chat/completions │
│    │              model: gpt-4-turbo-preview           │
│    │              temperature: 0.3                    │
│    │                                                  │
│    └─ 返回 response["choices"][0]["message"]["content"]│
│                                                      │
│  _clean_json_response(text)                          │
│    ├─ 去除 ```json ... ``` 包裹                       │
│    ├─ 去除 ``` ... ``` 包裹                           │
│    └─ 返回纯净 JSON 字符串                            │
│                                                      │
│  LLMCallTracker                                       │
│    ├─ 记录每次 LLM 调用的延迟 + 成功/失败              │
│    ├─ 分方法统计 P50/P95/P99 延迟                     │
│    └─ 输出：每轮问诊约 4~5 次 LLM 调用                 │
│        collect:   extract_symptom + 可能 generate_followup│
│        collect:   + 可能二次提取（MCP 工具后）         │ ← MCP 新增
│        analyze:   analyze_diagnosis                   │
│        reflect:   reflect_analysis                    │
│        report:    generate_report                     │
└─────────────────────────────────────────────────────┘
```

---

## 四、MCP 工具调用流程

```
collect 节点流程（含 MCP）：
  用户消息 → 关键词预检（药品/文件/历史检测）
    ├─ 命中 → MCP 工具调用 → 结果注入上下文
    └─ 未命中 → 跳过
  → LLM extract_symptom（含 tool_calls 字段）
    ├─ 返回 tool_calls → 执行 MCP 工具 → 二次提取
    └─ 无 tool_calls → 直接合并症状
  → 追问决策

analyze 节点流程（含 MCP）：
  → 检测症状+消息中是否有药品提及
    ├─ 有 → MCP get_drug_info → 结果合并到 rag_context
    └─ 无 → 跳过
  → LLM analyze_diagnosis
```

---

## 五、异常处理路径

```
A) LLM 调用失败                                     B) reflect 调用失败
                                                         │
  用户输入 → collect → ... → _call_llm → DeepSeek API    └─→ 降级返回通过值
                                              │              {"score":4,
                                 ┌────────────┤               "passed":true}
                                 ▼            ▼
                            HTTP 200     HTTP 非 200
                                 │            │
                                 ▼            ▼
                           JSON 解析    raise Exception
                                 │            │
                              KeyError?  consultation.py
                             （字段缺失）   HTTP 500 返回
                                 │           "问诊服务异常"
                                 ▼
                          consultation.py
                          except Exception
                              →
                          HTTP 500

C) MCP 工具调用失败
                             MCP Client.call_tool()
                                  │
                            try/except 包裹
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                            ▼
              调用成功                      调用失败
                    │                            │
                    ▼                            ▼
              结果注入上下文          is_error=True + 日志警告
                                     继续工作流，不阻塞
```

---

## 六、配置切换

```python
# .env 切换方式
LLM_PROVIDER=deepseek    # 使用 DeepSeek
# LLM_PROVIDER=openai    # 使用 OpenAI
# OPENAI_API_KEY=sk-xxx  # 切换到 OpenAI 时需配置

# 反思机制配置
REFLECTION_ENABLED=True       # 总开关
REFLECTION_MAX_ROUNDS=2       # 最大分析轮次（含初版）
REFLECTION_PASS_THRESHOLD=3   # 最低通过分数（1-5）

# MCP 工具层配置
MCP_ENABLED=True              # MCP 工具层总开关
MCP_SERVER_URL=http://localhost:8001  # MCP Server 地址（远程模式预留）
```
