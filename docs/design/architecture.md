# agent-ctl 架构设计(当前)

> 本文是系统**当前状态**的单一权威描述(取代两份 MVP 设计 spec 的"它是什么"部分;
> 决策与取舍的"为什么"见 [ADR-0001](../adr/0001-gateway-maturity-and-hardening.md) /
> [ADR-0002](../adr/0002-governance-and-audit-hardening.md))。配置字段见
> [configuration.md](../configuration.md),运维见 [operations.md](../operations.md)。
> 最后更新:2026-06-26。

## 1. 定位

`agent-ctl` 是一个**单实例的、受治理的 LLM 调用出口**:给 agent/RAG 一条统一路径,
在这条路径上做路由、回退、重试、墙钟 deadline、按 provider 熔断、成本预算闸、响应缓存、
全量脱敏捕获,并提供 Prometheus 指标。支持 5 家 provider(claude 原生 + openai/deepseek/
qwen/glm 走 OpenAI 兼容),对外有 **OpenAI 兼容 HTTP server** 与 **进程内库** 两种形态。

诚实边界:这是"单信任域、单进程"的治理代理,**不是**多租户控制面。预算/熔断/缓存均进程内、
重启清零、多副本不共享;存储是 SQLite;鉴权是单 token。这些是显式取舍(见 §10)。

## 2. 两种形态

```
消费者(rag / ops-agent / 任意 OpenAI SDK 客户端)
        │ 库形态:GatewayClient.messages()/embed()        │ server 形态:HTTP /v1/*
        ▼                                                 ▼
   client/gateway_client.py                        server/app.py(FastAPI)
        └───────────────┬─────────────────────────────────┘
                        ▼
                 core/gateway.py  ── Gateway(治理编排)
                        │  invoke / invoke_stream / embed
        ┌───────────────┼───────────────────────────────┐
        ▼               ▼                ▼               ▼
   Router          CircuitBreaker   BudgetGuard      Capturer ──► CaptureStore + Metrics
   (路由/别名/直连)  (按 provider)    (USD 上限)        (成本/脱敏/落库/指标)
        │
        ▼
   providers/*  ── AnthropicProvider(原生) / OpenAIProvider(×4,仅 base_url 不同)
                   工具/消息经 tooltrans 在边界互译(内部 canonical = Anthropic 形)
```

server 是对外主形态;库形态供同进程嵌入。两者共用同一个 `Gateway`,**治理逻辑只有一套**。

## 3. 治理流水线(一次 `invoke`)

1. **缓存**:精确匹配 key=(model, messages, params, tools, system, tool_choice);命中即返回(cost=0)。
2. **预算闸**:`BudgetGuard.check(consumer)`,达上限(含一次"典型调用"预留余量)→ 打 provider 前短路抛 `BudgetExceeded`。
3. **路由**:`Router.resolve(model)` → 有序目标链(逻辑名→链 / 别名→单目标 / 含 `/` 直连)。
4. **回退循环**(三 runner 共享形状):逐目标 —— deadline 守卫 → 未注册 provider 守卫 → 熔断 `allow` 跳过 → 调 provider(带 per-target 重试 + 超时压到剩余预算)→ 成功 `record_success`+捕获+返回;终态 `record_failure`+捕获+透传;可重试 `record_failure`+回退下一目标。
5. **捕获**:`Capturer` 算成本 → 计预算 → 上报指标 → 脱敏落库 → 结构化日志。全程 fail-open(捕获侧任何异常只告警,绝不打断真实调用)。
6. 全目标耗尽 → `AllTargetsFailed`。

`invoke_stream` 与 `embed` 复用同一形状(见 §5),差异在执行与捕获细节。

## 4. 模块职责

| 模块 | 职责 |
|---|---|
| `core/gateway.py` | `Gateway` 控制流编排 + 共享守卫(`_deadline_exceeded`/`_circuit_blocked`)+ per-target 重试 + 缓存接线 |
| `core/stream_runner.py` / `embedding_runner.py` | `Gateway` 的 mixin:`invoke_stream` / `embed` 的执行循环(共用 Gateway 的守卫/状态)|
| `core/capture.py` | `Capturer` 协作者:成本/预算/指标/脱敏落库/日志(横切关注,从 Gateway 抽出)|
| `core/{router,circuit,budget,cache,cost}.py` | 路由、按 provider 熔断、USD 预算闸、有界 LRU 缓存、价表 |
| `providers/{base,catalog,tooltrans}.py` | Provider 协议、5 家目录+能力推导、跨 provider 工具/消息互译 |
| `providers/{anthropic,openai}_provider.py` | 原生 Anthropic / OpenAI 兼容适配(invoke/stream/embed)|
| `store/{sqlite_store,async_store,redaction}.py` | SQLite 捕获存储、异步写装饰器、脱敏 |
| `server/app.py` | OpenAI 兼容 FastAPI:`/v1/chat/completions`(含 SSE)、`/v1/embeddings`、`/v1/models`、`/metrics`、`/healthz`;鉴权/限流/体积/直连白名单 |
| `client/gateway_client.py` | 库门面 + `from_config` 装配 |
| `obs/metrics.py` | Prometheus 指标(可选依赖,未装则 no-op)|
| `cli.py` | `serve` / `doctor` / `captures` / `cost` / `export` / `config-schema` / `version` |

## 5. Runner 架构与决策

`Gateway(StreamRunnerMixin, EmbeddingRunnerMixin)` —— 三条执行路径(invoke 内联 / stream / embed
mixin)**共享** Gateway 的守卫(deadline/circuit)、`_invoke_target`、`Capturer`、`Router`、状态。
守卫抽成单一真相源(此前 deadline/circuit 判断散布多处导致同类 bug 要多处修)。

**已知取舍**(深审指出,见 ADR-0002):mixin 直接访问 `self._budget/_circuit/...`,带
`# mypy: disable-error-code=attr-defined`——是"文件级提取"而非可独立实例化的真解耦,
mypy 沉默掩盖了潜在类型错误。**复评触发**:出现第 3 条执行路径,或 mixin 共享的私有 API
面继续膨胀时,改为协作者/策略对象。当前规模(三个小文件)下可维护。

## 6. 数据模型(`models.py`)

- **NormalizedRequest**:model, messages, max_tokens, temperature, tools, system, tool_choice, metadata(含 consumer)。内部 canonical 消息/工具是 **Anthropic 风格**。
- **NormalizedResponse**:text, finish_reason, tool_calls(计数), input/output_tokens, raw(Anthropic 风格完整响应,供消费者还原 tool_use)。
- **StreamChunk**:text 增量;`done=True` 终块带 finish_reason + 计量 + 重组后的 tool_calls。
- **EmbeddingResponse**:vectors, input_tokens。
- **CallRecord**(落库,**脱敏后**):标识/来源/请求(脱敏 messages)/路由(model_resolved + attempts[])/响应(脱敏 output、tool_calls 计数)/计量(tokens、cost)/缓存/状态/错误。**不落 raw**(避免 tool_use input 里的密钥落地)。

## 7. Provider 模型

- 三个协议:`Provider`(必备 `invoke`)、`StreamingProvider`(可选 `stream`)、`EmbeddingProvider`(可选 `embed`),后两者 `@runtime_checkable`。
- `PROVIDER_CATALOG`:5 家 = {kind, key_env, base_url};仅有 key 的被构造。能力按 kind 静态推导(`provider_capabilities`),`doctor` 据此打能力矩阵并对回退链能力不一致告警。
- **工具/消息互译**(`tooltrans`):内部 canonical = Anthropic 形(首消费者 ops-agent 原生、tool_use 表达全)。OpenAI 家族在边界双向翻译。**代价**:占 4/5 的 OpenAI 兼容 provider 每次双翻译;而 server 已是对外主形态 → 复评条件已触发,以契约测试守翻译正确性(见 ADR-0002)。

## 8. 治理与安全模型

| 面 | 当前 | 边界 |
|---|---|---|
| 鉴权 | 单 server token(`hmac.compare_digest`);非本地 serve 强制非默认 token | 无 per-consumer key;多人共用即信任崩 → **单信任域** |
| consumer 身份 | 取自 OpenAI `user` 字段(供 per-consumer 预算/归因)| `user` 可伪造;真隔离需 per-token(后置)|
| 直连白名单 | `allow_direct_models`(默认禁):`provider/model` 直连未登记目标被拒,堵成本治理绕过 | 库形态默认放行(可信调用方)|
| 预算 | per-consumer + 全局 USD 上限,达限短路 402;预留一次典型成本余量收紧并发越界 | 进程内、重启清零、多副本不共享 |
| 成本 | 价表;`profile: prod` 下未知价 fail-closed(`UnknownPriceError`)| dev 下未知价告警不阻断 |
| deadline | 单次调用墙钟总预算;封顶 重试×回退×超时;**流式开流后也逐块约束**(超预算截断)| 单个已阻塞 next() 读取靠 provider SDK read timeout |
| 熔断 | 按 provider 开/半开/闭;回退链跳过开路目标 | 进程内 |
| 限流 | 按 IP(可选信任 XFF),客户端数有界 LRU | XFF 信任需在可信代理后 |
| 脱敏 | 落库前递归脱敏 key/token/JWT/DSN/邮箱/手机号(含嵌套 content 与 tool payload)| 正则法,分段/编码绕过仍可能 |

## 9. 存储与可观测

- **SqliteCaptureStore**:一行一 CallRecord(JSON 整存 + 关键列冗余);WAL + 写锁;`iter_all` 另开只读连接流式导出。
- **AsyncCaptureStore**:把落库移出请求主路径(后台线程 + 有界队列),fail-open + atexit 兜底落尾。
- **/metrics**:Prometheus(requests/duration/tokens/cost/errors/cache_hits);未装 prometheus_client 则 no-op。
- **export**:`agent-ctl export` 流式 JSONL(时序),供 eval/replay——但消费侧(eval 门禁/漂移/回放/dashboard)尚未实现,见 §10。

## 10. 已知边界 / 非目标(显式)

随"分布式部署 + 捕获库 PG 化"专项一并推进,非遗漏:
- 分布式/共享的预算、熔断、缓存(当前进程内)。
- 预算持久化 / 滚动时间窗(当前重启清零)——这是"拦得住超支"叙事的最大缺口。
- 捕获存储 PG 化(`CaptureStore` 接口待与实现对齐,见 ADR-0002)。
- 多租户鉴权 / per-consumer key。
- 上层"控制面"消费(eval 门禁 / 漂移检测 / 回放 / dashboard):捕获只产出原料,消费层未建。
- 内部 canonical = Anthropic 形的双翻译(复评条件已触发,以契约测试守正确性,暂不翻转)。

## 11. 扩展点

- **加 OpenAI 兼容后端**:`PROVIDER_CATALOG` 加一条(name/base_url/key_env),零适配代码。
- **加原生 SDK provider**:新建 `xxx_provider.py` + catalog `kind` + `build_providers` 分支 + `provider_capabilities` 分支(4 处)。
- **加能力(rerank/vision 等)**:扩 Provider 协议;vision 需补 tooltrans 的图片块翻译(当前只处理 text/tool_use/tool_result)。
