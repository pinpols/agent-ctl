# ADR-0002:治理硬化与深度审核整改

> 日期:2026-06-26。状态:已采纳(Accepted)。
> 承接 [ADR-0001](0001-gateway-maturity-and-hardening.md);系统当前架构见
> [architecture.md](../design/architecture.md)。

## 背景

ADR-0001 之后,系统经过 3 轮代码审核(挖修 F1–F6 / G1–G5 / H1–H7,共 17 项真 bug)、
一轮多维度深度审核,以及一次执行路径重构。本 ADR 统一记录这批决策——重点是**把"招牌
功能(治理)在主路径上被绕过"的几处缝补上**,以及一次架构重构的取舍归档。

## 决策

### D1 执行路径拆为 mixin + 共享守卫 + Capturer 协作者
`invoke`/`invoke_stream`/`embed` 此前各自重抄"路由→deadline→熔断→捕获"骨架,导致同类
修复要在多 runner 重复改。整改:抽 `_deadline_exceeded`/`_circuit_blocked` 守卫为单一真相源;
把"成本/预算/指标/脱敏落库/日志"抽到 `Capturer` 协作者;`invoke_stream`/`embed` 落入
`StreamRunnerMixin`/`EmbeddingRunnerMixin`(`Gateway` 多继承)。
**已知取舍**:mixin 直接访问 `Gateway` 私有成员、带 `# mypy: disable=attr-defined`——是
文件级提取而非可独立测试的真解耦,mypy 沉默掩盖潜在类型错误(深审 A1)。**复评触发**:
出现第 3 条执行路径,或共享私有 API 面继续膨胀时,改协作者/策略对象。

### D2 流式 deadline 与中途语义
- **开流后 deadline 也生效**:此前 deadline 只约束开流前,长流/卡顿流可无视预算跑满。
  整改:native 流的逐块迭代里加 deadline 检查,超预算即截断(已发部分保留 + `finish_reason=length`
  + `status=deadline` 落库)。单个已阻塞 `next()` 读取仍靠 provider SDK read timeout。
- **中途失败语义**:首块已出即提交该 target 不回退;中途断流 `record_failure`(反复中途失败
  能正确累计开路,不"开流即自我赦免");客户端断流(GeneratorExit)落 `status=aborted` 捕获。

### D3 堵成本治理的三处绕过(深审横切发现)
治理的硬卖点是"拦得住超支",但在 HTTP 主形态上曾三重可绕过,整改:
- **consumer 不再硬编码**:取自 OpenAI `user` 字段 → per-consumer 预算/归因对诚实调用方生效。
  (真隔离仍需 per-token 身份,后置。)
- **直连白名单 `allow_direct_models`(server 默认禁)**:禁止 `provider/model` 直连未在
  routes/aliases 登记的目标,堵"调任意已注册 provider 任意模型"绕过路由白名单。库形态默认放行。
- **prod 未知价 fail-closed**:`profile: prod` 下目标缺价 → `UnknownPriceError` 终态拒绝
  (而非静默 cost=None 放行);dev 仍告警不阻断。

### D4 契约健壮性整改
- OpenAI 兼容 server 流式补 **usage 帧**(`stream_options.include_usage` 约定),否则流式
  路径客户端永远拿不到 token 计量。
- 多条 `system` 消息**合并**(漏合并会把第 2 条起当普通消息发给 Anthropic 触发 role 报错)。
- `OpenAIProvider` 空 `choices`(内容过滤等)→ 显式 `TerminalError`,而非裸 `choices[0]`
  的未捕获 `IndexError` 冒 500。

### D5 资源有界化(防长驻泄漏)
LRU 有界缓存(`cache_max_entries`)、限流客户端数有界淘汰、异步捕获 atexit 兜底、捕获存储
独立只读连接流式导出——长驻 server 不再"只增不减"。

### D6 不落 raw
`CallRecord` 不持久化 `NormalizedResponse.raw`,避免 tool_use input 等原始字段里的密钥落入
SQLite(深审复核:raw 本就未落库,此为显式确认而非新改)。

## 仍然保留 / 仍开口(显式)

承 ADR-0001 的非目标外,深审确认以下仍开口,按对采用的杀伤力排:
1. **预算持久化(扛重启)** —— "拦得住超支"叙事的最大缺口;最小动作=预算落 SQLite + 滚动窗。
2. **per-token consumer 身份** —— 当前 `user` 可伪造,多人共用即治理崩。
3. **`CaptureStore` Protocol 与实现对齐** —— 当前 Protocol 签名过时、缺 `iter_all`,PgStore 插入无法类型校验。
4. **真 provider 契约进 CI** —— provider 解析全打 fake SDK;nightly 无 key 时 skip-to-green。建议用真实录制样本作 fixture + 无 provider 跑时显式告警而非绿。
5. **双翻译复评条件已触发**(OpenAI 兼容是对外主形态)—— 暂以工具调用往返契约测试守正确性,不翻转 canonical。

## 影响

- 对外面新增:流式 usage 帧、402 预算信号(已存)、`allow_direct_models` 行为、prod fail-closed 价格。
- 默认行为变化:**server 默认禁直连未登记模型**(库形态不变)、**prod 缺价直接拒绝**。升级者需确认 routes/aliases 与 prices 覆盖了要暴露的模型。
- 文档:[architecture.md](../design/architecture.md) 成为"当前架构"权威;两份 MVP spec 仅作历史。
