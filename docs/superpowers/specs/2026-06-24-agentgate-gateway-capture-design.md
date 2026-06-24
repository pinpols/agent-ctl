# agentgate — 调用网关 + 全量捕获(设计)

> 工作名 `agentgate`(可改)。本文件是 **AgentOps 控制面** 的第一个子项目(脊柱)的设计。
> 日期:2026-06-24。状态:待用户复核 → 转 writing-plans。

## 1. 背景与北极星

已有两个"用模型"的项目:`vecstream`(RAG/检索)、`ops-agent`(应用层运维诊断 agent,含工具/记忆/HITL/eval/Langfuse/prod 安全闸)。再做"又一个 agent"或"又一个 RAG"都重复。

**真正不重复的方向**:不做"一个 agent",做 **agent 本身的工程化/运维平台(AgentOps 控制面)**——把"控制面"经验(注册/配置/治理/可观测/限流)用到 agent 的 **调用 → 管理 → 维护 → 开发** 全生命周期。它与 `ops-agent`(一个具体 agent)是上下层关系。

**北极星(完整控制面,四根支柱)**,本子项目只做第 1 根:

```
        [运行治理/控制面]  注册表 + 配置 + 可观测大盘            ← 子项目 3
              ▲
   [质量维护]  [开发体验]  eval门禁/漂移/回放 ; prompt版本/脚手架  ← 子项目 2 / 4
              ▲
   ┌──────────┴───────────┐
   │  调用网关(脊柱) ★本期 │  路由/回退/超时重试/成本/缓存 + 全量捕获
   └──────────────────────┘
              ▲
         ops-agent / 任何 agent
```

**为什么脊柱先做**:它是所有模型调用的唯一咽喉——既是治理点,又顺带**捕获每一次调用**,而这份捕获数据是上面三根支柱(eval/漂移/回放/大盘)的唯一数据源。没有它,上层皆空中楼阁。

## 2. 范围与非目标

**本期做(调用网关 + 捕获)**
- 统一调用入口:**路由 → 回退 → 超时/重试 → 成本核算 → 缓存 → 全量捕获**。
- 每次调用落一条结构化 `CallRecord`(脱敏后)。
- `ops-agent` 接入:仅改其 `llm.py` 走网关。

**非目标(留后续子项目 / 后置)**
- ❌ eval 门禁 / 漂移检测 / trace 回放 → 子项目 2
- ❌ 注册表 / 配置中心 / 可视化大盘 → 子项目 3
- ❌ prompt 版本管理 UI / 脚手架 → 子项目 4(但 `CallRecord.prompt_version` 字段本期预留)
- ❌ 限流 → v2(第一刀聚焦 路由/回退/成本/缓存/捕获)
- ❌ 多租户 / 鉴权 / 分布式部署(单机、单用户 showcase 起步)
- ❌ 语义缓存(拉进 embedding 会与 vecstream 重叠)→ v2

## 3. 架构与接入形态

**核心 = 传输无关的治理引擎**(纯逻辑:routing/fallback/retry/cost/cache/capture)。接入形态决策:**方案 C —— 核心 + 双形态**:本期出"库形态",代理形态留接口/骨架不实现。兼顾出活速度与"任意 agent 接入"的 showcase 卖点。

```
agentgate/
  core/        治理引擎(传输无关):router / fallback / retry / cost / cache / capture 编排
  providers/   provider 适配(本期仅 Anthropic;接口预留 OpenAI/Ollama)
  store/       捕获存储(本期 SQLite + repository 接口;预留 PG)
  client/      库形态接入(包住 client,ops-agent 用这个)
  server/      代理形态(留接口/骨架,本期不实现)  ← 给"任意 agent 接入"留门
  config.py    配置(路由表 / 价表 / 缓存 / profile,统一从 yaml + env)
  cli.py       命令行(查看捕获、成本汇总、配置自检)
```

### 组件边界(每个:做什么 / 接口 / 依赖)

- **`core.Gateway`** — 编排者。入参=归一化请求,出参=归一化响应。内部按序:查缓存 → 路由选目标 → 调 provider(带超时/重试)→ 失败回退下一目标 → 算成本 → 写捕获 → 返回。依赖:Router、Provider、CostMeter、Cache、CaptureStore(全经接口注入)。
- **`core.Router`** — 逻辑名 → 有序目标链(`dict[str, list[Target]]`,不写 if-chain)。纯函数式选择,无副作用。
- **`providers.Provider`(Protocol)** — `invoke(target, request, timeout) -> ProviderResult`;抛分类异常(`RetriableError` / `TerminalError`)。本期 `AnthropicProvider` + 测试用 `FakeProvider`。
- **`core.CostMeter`** — 价表(`dict[model, (in_price, out_price)]`)→ 由 token 算 `cost_usd`;未知模型返回 None + warn。
- **`core.Cache`(Protocol)** — 精确匹配 key=(model, 归一化 messages, params),TTL。默认 SQLite(测试用内存实现);可经配置关闭。
- **`store.CaptureStore`(Protocol)** — `save(CallRecord)`;本期 `SqliteCaptureStore`,预留 `PgCaptureStore`。
- **`store.redaction`** — 落库前脱敏(token/DSN 口令/JWT/邮箱/手机号),端口 ops-agent 的范式。
- **`client.GatewayClient`** — 库形态门面,签名贴近原生 client,内部转 `Gateway.invoke`。ops-agent 的 `llm.py` 指向它。

## 4. 数据模型:`CallRecord`

落库前脱敏。一条调用一条记录。

| 字段组 | 字段 | 说明 |
|---|---|---|
| 标识 | `id` `ts` `latency_ms` | UUID + 起始时间 + 端到端耗时 |
| 来源 | `consumer` `call_site` `trace_id` | 谁调(如 `ops-agent`)、调用点、链路 id(对接 trace/Langfuse) |
| 请求 | `model_requested` `params` `messages_redacted` `prompt_version` | 逻辑模型名、温度/max_tokens/有无工具、脱敏输入、**prompt 版本(预留)** |
| 路由 | `model_resolved` `attempts[]` | 实际命中目标 + 每次尝试 `{provider,model,outcome,latency,error}` |
| 响应 | `output_redacted` `finish_reason` `tool_calls` | 脱敏输出、结束原因、工具调用数 |
| 计量 | `input_tokens` `output_tokens` `cost_usd` | token + 成本(未知模型 None) |
| 缓存 | `cache_hit` `cache_key` | 是否命中 |
| 状态 | `status` `error_type` | `success / fallback_success / error` + 错误归类 |

**存储**:`CaptureStore` 接口 → 先 `SqliteCaptureStore`(零依赖、本地)。原始请求体默认**只存摘要**(避免体积与隐私);需要完整体留 v2 开关。

## 5. 治理行为(第一刀五件事)

1. **路由** — 声明式配置:逻辑名 → 有序目标链。例 `default → [anthropic/opus-4-8, anthropic/sonnet-4-6]`。
2. **回退** — 主目标遇可重试故障(429/5xx/overloaded/超时)→ 链上下一个;每次尝试入 `attempts[]`,最终 `status=fallback_success`。
3. **重试/超时** — 每次尝试带超时 + 有界退避重试;**区分可重试(429/5xx/超时)vs 终态(4xx 鉴权/参数 → 不重试,直接透传)**。
4. **成本** — 可版本化价表 → 实时算 `cost_usd` + 汇总;未知模型告警不阻断。
5. **缓存** — 可选响应缓存,精确匹配 + TTL;语义缓存后置 v2。

### 容错原则(成熟度的核心体现)

**治理侧故障绝不打断真实调用**:缓存挂、捕获写失败 → 记 warn、放行(fail-open,沿用 ops-agent 的 redaction/Langfuse no-op 范式)。但 **LLM 调用自身的终态错误照常透传给消费者**(不吞错)。

## 6. ops-agent 接入(首个真实消费者)

- 改动面**仅 `ops-agent/llm.py`**(它本就是"共享 client 工厂 + 集中重试"的单一接入点)+ 少量配置。
- ops-agent 继续是它;但从此每次调用被网关治理 + 捕获。
- 重试逻辑从 ops-agent 上移到网关(去重),`llm.py` 变薄。

## 7. 错误处理

- provider 异常分类:`RetriableError`(429/5xx/timeout/overloaded)、`TerminalError`(4xx auth/validation)。
- 全链路重试耗尽 / 全目标回退失败 → 抛聚合错误给消费者,`status=error` + `attempts[]` 全留痕。
- 治理组件(cache/store/cost)异常 → fail-open + warn,绝不影响主调用。

## 8. 测试策略(TDD)

- **离线 `FakeProvider`**:可脚本化产出成功/失败/超时/限流,**无需真 key** 即可测全部治理逻辑。
- 覆盖:路由选择、回退切换、重试退避与终态不重试、成本计算(含未知模型)、缓存命中/失效/TTL、捕获落库与脱敏、治理 fail-open。
- 沿用 ops-agent 工具链:pytest + ruff + mypy;`agentgate doctor` 配置自检。

## 9. 北极星路线(本期 = 第 1 根支柱)

1. **调用网关 + 捕获(本期)**
2. 质量维护:eval 门禁 + 漂移检测 + trace 回放(吃本期捕获数据)
3. 运行治理:注册表 + 配置 + 大盘(可视化 #1/#2)
4. 开发体验:prompt 版本管理 + 脚手架(与 #2 穿插)

## 10. 待定 / 后续

- 价表来源与更新机制(手工 yaml 起步;后续可拉取)。
- 代理形态(server/)落地时的协议(Anthropic 兼容 vs OpenAI 兼容)。
- 捕获存储换 PG 的时机(showcase 单机 SQLite 足够)。
