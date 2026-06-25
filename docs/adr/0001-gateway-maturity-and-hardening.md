# ADR-0001:网关成熟度扩张与硬化(对齐文档与实现)

> 日期:2026-06-25。状态:已采纳(Accepted)。
> 关联:supersedes 子项目1设计 §2、子项目2设计 §3 的若干"非目标"。

## 背景

子项目1(网关+捕获)、子项目2(OpenAI 兼容多 provider server)落地后,系统在两轮
迭代中超出了两份原始 spec 的"非目标"边界:先补齐 5 项成熟度(跨 provider 工具归一、
指标、熔断、流式、embeddings),再做 5 项架构硬化(异步捕获、请求 deadline、预算闸、
真流式、文档回写)。原 spec 的"不做"清单已不再描述真实系统——本 ADR 统一记录这些
决策,并把对应 spec 段标注为被取代,消除"控制面文档描述不了自己"的债。

## 决策

### D1 流式:缓冲式 → 真·passthrough(取代 子项目1 §2「流式→v2」、子项目2 §3「不做 streaming」)
`POST /v1/chat/completions` 的 `stream=true` 走 `Gateway.invoke_stream`,逐块下发。
- 治理一致:预算闸/路由/熔断/deadline 照旧;捕获在流结束后按累计文本+计量落一条记录
  (复用 `_capture`,成本计入预算)——这正是当初推迟流式时担心的"chunk 聚合后才能算
  成本/捕获",解法是**累计到末块再落**。
- 回退语义:**开流前**(server 预拉首块)失败可回退下一目标 / 降级为普通 HTTP 状态
  (400/402/502);**一旦首块已出**即提交该目标不再回退(已发字节无法回退,与业界一致)。
- 兼容:无原生 `stream()` 能力的 provider 退化为缓冲式(跑非流式再切块,保留全部治理)。

### D2 embeddings:实现 `/v1/embeddings`(取代 子项目2 §3「不做 embeddings」)
`EmbeddingProvider` 可选能力协议;OpenAI 兼容 provider 实现 `embed`(按 data.index 排序),
无此能力者(Anthropic)在回退链留痕 `no_embed` 跳过。走与 invoke 同一治理。

### D3 跨 provider 工具归一:实现(取代 子项目2 §3「先透传」)
`tooltrans` 双向互译(Anthropic tools/tool_use ↔ OpenAI function/tool_calls),含多轮
工具循环消息翻译。内部规范格式 = Anthropic 风格(见 D7)。

### D4 可观测:Prometheus `/metrics`
`Metrics`/`MetricsRegistry`;每次调用上报 requests/duration/tokens/cost/errors/cache。

### D5 弹性:按 provider 熔断 + 单次调用 deadline
- `CircuitBreaker`:连续失败达阈值开路冷却,回退链跳过(半开试探)。进程内。
- `request_deadline_s`:每次尝试超时压到 min(配置, 剩余预算),封顶"目标×重试×超时"最坏延迟。

### D6 治理:成本预算闸(从"看得见花费"到"拦得住超支")
`BudgetGuard`:per-consumer / 全局 USD 上限,调用前 `check()` 达限即 `BudgetExceeded`
(打 provider 前短路),调用后 `add()` 实际成本。server 映射 **402 budget_exceeded**。

### D7 捕获落库异步化(移出请求主路径)
`AsyncCaptureStore` 装饰任意 store:`save()` 入队即返回,后台单线程落库。fail-open
(队列满丢弃+计数告警);读路径先 flush 保证写后即读;`Config.capture_async` 控制。

### D8 内部规范格式 = Anthropic 风格(显式记录,暂不翻转)
首消费者 ops-agent 为 Anthropic 原生、且 tool_use 表达更全,故内部 canonical = Anthropic 形。
**代价**:占 4/5 的 OpenAI 兼容 provider 每次做两次翻译(OpenAI 入→Anthropic 内部→OpenAI 出)。
当前规模下翻转收益<风险(YAGNI),保持现状;**触发复评条件**:OpenAI 兼容面成为主导且
该层翻译进入热点 profile。

## 仍然保留的非目标(显式,勿当缺陷)

- 分布式/共享的熔断与缓存(当前进程内,多副本各算各的)。
- 捕获存储 PG 化(当前单机 SQLite;接口已留 `CaptureStore`)。
- 预算的持久化 / 滚动时间窗 / 跨副本共享(当前进程内、进程生命周期窗口,重启清零)。
- 成本分层计价 / prompt-caching 计价建模(价表手工 yaml,未知模型告警不阻断)。
- 多租户鉴权 / per-consumer API key / key 轮换(当前单 server token + 网络管控)。

以上随"分布式部署 + 捕获库 PG 化"专项一并推进,非本期遗漏。

## 影响

- 两份 spec 的相关"非目标"段已加被取代标注,README 同步更新到真实能力面。
- 新增可选依赖:`prometheus-client`(server/dev extra)。
- 对外面新增:SSE 流式、`/v1/embeddings`、`/metrics`、402 预算信号。
