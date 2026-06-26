# agent-ctl 子项目2:OpenAI 兼容多 provider 网关 server(设计)

> 日期:2026-06-24。延续子项目1(网关+捕获)。本文件定义"对齐到业界标准"的 server 形态。
>
> ⚠️ **历史文档(仅存 MVP 设计意图)**。系统**当前架构**以
> [architecture.md](../../design/architecture.md) 为权威;决策演进见
> [ADR-0001](../../adr/0001-gateway-maturity-and-hardening.md) 与
> [ADR-0002](../../adr/0002-governance-and-audit-hardening.md)。本文 §3「不做」清单
> (streaming/embeddings/工具归一/鉴权 + /metrics/熔断/deadline/预算)均已实现。

## 1. 背景与对齐决策

业界多 provider LLM 接入的事实标准:**对消费者暴露 OpenAI 兼容 `/v1/chat/completions`,前面架一个网关**(LiteLLM / OpenRouter / Portkey / Cloudflare AI Gateway 都是此形态)。网关内部做路由/回退/成本/缓存/key 管理,只在需要 provider 专属特性(Anthropic prompt caching / extended thinking)时走原生 SDK。

**决策**:agent-ctl 的 server 形态 = **OpenAI 兼容网关**(取代子项目1 里含糊的"代理形态"骨架)。对外唯一标准面 = `POST /v1/chat/completions`。消费者(rag / ops-agent / 任意 agent)只设 `OPENAI_BASE_URL=http://agent-ctl:PORT/v1`,用现成 openai SDK,零定制代码,即获网关治理。

```
rag / ops-agent / 任意 agent  ──(openai SDK, base_url 指过来)──►  agent-ctl server (/v1/chat/completions)
                                                                      │ NormalizedRequest
                                                                      ▼
                                                          Gateway(路由/回退/超时/成本/缓存/捕获)
                                                                      │
                              ┌───────────────────────┬──────────────┴───────────────┐
                              ▼                        ▼                              ▼
                       AnthropicProvider         OpenAIProvider(base_url=…)     OpenAIProvider(base_url=…)
                       (Claude 原生)             openai / deepseek             qwen(通义) / glm(智谱)
```

子项目1 的库形态 `GatewayClient.messages()` 降为"进程内嵌"次要入口;OpenAI 兼容 server 是对外标准面。

## 2. 支持的 provider(5 家)

| provider 名 | 适配器 | base_url | api key env |
|---|---|---|---|
| `anthropic`(Claude) | AnthropicProvider(原生) | — | `ANTHROPIC_API_KEY` |
| `openai` | OpenAIProvider | 默认 api.openai.com | `OPENAI_API_KEY` |
| `deepseek` | OpenAIProvider | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `qwen`(通义千问) | OpenAIProvider | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| `glm`(智谱) | OpenAIProvider | `https://open.bigmodel.cn/api/paas/v4` | `GLM_API_KEY` |

4 家 OpenAI 兼容的复用同一个 `OpenAIProvider`,仅注入不同 base_url + key。仅 Claude 走原生(保留 thinking / prompt caching 等)。

## 3. 范围

**做**
- **Provider 目录 + 工厂**:`build_providers(config) -> dict[str,Provider]`,按上表从 env 读 key + base_url 构造;**只纳入 key 存在的 provider**(缺 key 的不注册,doctor 会提示)。SDK 延迟 import。
- **OpenAI 兼容 server**(FastAPI):`POST /v1/chat/completions`、`GET /v1/models`、`GET /healthz`。请求体 OpenAI 形 → NormalizedRequest(system 从 messages 提取 / 透传)→ `gateway.invoke` → NormalizedResponse → OpenAI 形响应。
- **模型路由**:请求 `model` 字段解析为 target——含 `/` 直接当 `provider/model`;否则查 `config.routes`(逻辑名→目标链,继承子项目1)或 `config.model_aliases`。
- **CLI**:`agent-ctl serve --port` 起 server。

**不做(YAGNI / 后置)**
- ❌ streaming(`stream:true`)→ 先非流式;OpenAI 形先返完整 response(后置 SSE)
- ❌ embeddings 端点(`/v1/embeddings`)→ 后置(vec-stream 有独立 embed-service)
- ❌ tools/function-calling 的跨 provider 归一(Anthropic vs OpenAI tools schema 不同)→ 先透传,深度归一后置
- ❌ 鉴权(网关自己的 API key 校验)→ 本地/可信网络起步,后置

## 4. 组件

- `providers/catalog.py`:`PROVIDER_CATALOG`(上表的 base_url + key-env 静态目录)+ `build_providers(config)`。
- `server/app.py`:`build_server(gateway, models)`→ FastAPI app(取代 NotImplemented 骨架)。请求/响应翻译纯函数 `to_normalized(openai_req)` / `to_openai_response(normalized, model)` 便于单测。
- `config.py`:加 `model_aliases: dict[str,str]`(可选,逻辑名→provider/model);provider base_url 覆盖可选。
- `cli.py`:`serve` 子命令。

## 5. 容错 / 一致性(承子项目1)

- 治理 fail-open、失败 attempts 留痕、缓存命中 cost=0、落库脱敏 —— 全部继承 Gateway,不变。
- server 层错误 → 返回 OpenAI 形 error 体(`{"error":{"message","type","code"}}`)+ 合适 HTTP 码(终态 4xx 透传,全失败 502)。

## 6. 测试(TDD)

- 翻译纯函数单测(OpenAI req↔NormalizedRequest / Normalized↔OpenAI resp)。
- server 用**注入的 fake gateway**(不连真 provider、无需 key)测 `/v1/chat/completions` 返 OpenAI 形、model 路由、错误映射、`/healthz`、`/v1/models`。
- `build_providers` 用 monkeypatch env 测"仅注册有 key 的 provider"。
- 沿用离线 FakeProvider;真 provider 调用不进单测。

## 7. 接入收益(对齐后)

- **rag**:现有 openai 路径代码不改,`OPENAI_BASE_URL` 指向 agent-ctl → 自动获路由/回退/成本/缓存/捕获。
- **ops-agent**:bespoke `messages()` 可退役,改 openai SDK 指向 agent-ctl。
- 任意新 agent:零定制接入。
