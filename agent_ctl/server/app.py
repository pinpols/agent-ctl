from __future__ import annotations

import hmac
import json
import os
import time
import uuid
from collections import OrderedDict, deque
from ipaddress import IPv4Network, IPv6Network, ip_address, ip_network
from json import JSONDecodeError

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    GatewayError,
    TerminalError,
)
from agent_ctl.models import NormalizedRequest, NormalizedResponse
from agent_ctl.providers.tooltrans import stop_reason_to_finish, validate_local_content

_DEFAULT_TRUSTED_PROXY_CIDRS = ("127.0.0.1/32", "::1/128")


def _consumer_of(body: dict) -> str:
    """调用方身份 = OpenAI 约定的 `user` 字段(供 per-consumer 预算/归因生效);
    缺省回退固定名。注:HTTP 端鉴权仍是单 token,user 可伪造——这是把 per-consumer
    预算从"完全无效"提到"诚实调用方可用",真正隔离需 per-token 身份(见 ADR 后置)。
    """
    user = body.get("user")
    if isinstance(user, str) and user.strip():
        return user.strip()[:128]
    return "openai-compat-server"


def to_normalized(body: dict, default_max_tokens: int = 1024) -> NormalizedRequest:
    """OpenAI /v1/chat/completions 请求体 → NormalizedRequest。

    OpenAI 允许多条 role=system;全部抽出按序合并到 NormalizedRequest.system
    (AnthropicProvider 需独立 system 字段——漏合并会把第 2 条起当普通消息发给
    Anthropic 触发 role 报错)。
    """
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("field 'messages' must be a list")
    system_parts: list[str] = []
    rest = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        if m.get("role") == "system":
            c = m.get("content")
            system_parts.append(c if isinstance(c, str) else json.dumps(c))
        else:
            rest.append(m)
    system = "\n\n".join(system_parts) if system_parts else None
    max_tokens = body["max_tokens"] if "max_tokens" in body else default_max_tokens
    try:
        if isinstance(max_tokens, bool):
            raise ValueError
        max_tokens = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError("field 'max_tokens' must be an integer") from exc
    if max_tokens <= 0:
        raise ValueError("field 'max_tokens' must be a positive integer")
    temperature = body.get("temperature")
    if temperature is not None:
        try:
            if isinstance(temperature, bool):
                raise ValueError
            temperature = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("field 'temperature' must be a number") from exc
        if not 0 <= temperature <= 2:
            raise ValueError("field 'temperature' must be between 0 and 2")
    if body.get("tools") is not None and not isinstance(body["tools"], list):
        raise ValueError("field 'tools' must be a list")
    if body.get("tool_choice") is not None and not isinstance(
        body["tool_choice"], (dict, str)
    ):
        raise ValueError("field 'tool_choice' must be an object or string")
    # 本地可判定的终态校验(多模态/tool_choice 字符串)在 HTTP 边界直接 400,
    # 不进 gateway → 不产生捕获记录、更不可能污染熔断。
    try:
        validate_local_content(rest, body.get("tool_choice"))
    except TerminalError as exc:
        raise ValueError(str(exc)) from exc
    return NormalizedRequest(
        model=body["model"],
        messages=rest,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=body.get("tools"),
        tool_choice=body.get("tool_choice"),
        metadata={"consumer": _consumer_of(body)},
    )


def _raw_tool_uses_to_openai(resp: NormalizedResponse) -> list[dict]:
    """从 raw 的 Anthropic 风格 content 还原 OpenAI message.tool_calls(非流式工具调用)。"""
    blocks = (resp.raw or {}).get("content") or []
    out = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "tool_use":
            out.append(
                {
                    "id": b.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                    },
                }
            )
    return out


def to_openai_response(
    resp: NormalizedResponse, requested_model: str, created: int
) -> dict:
    tool_calls = _raw_tool_uses_to_openai(resp)
    # 有工具调用且无文本时 content=null(OpenAI 约定);否则原样输出文本。
    content: str | None = resp.text
    if tool_calls and not resp.text:
        content = None
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                # Anthropic 后端透出的 stop_reason(end_turn/max_tokens/tool_use)
                # 映射回 OpenAI finish_reason;已是 OpenAI 值则原样。
                "finish_reason": stop_reason_to_finish(resp.finish_reason) or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": resp.input_tokens,
            "completion_tokens": resp.output_tokens,
            "total_tokens": resp.input_tokens + resp.output_tokens,
        },
    }


def to_openai_embeddings(resp, requested_model: str) -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(resp.vectors)
        ],
        "model": requested_model,
        "usage": {
            "prompt_tokens": resp.input_tokens,
            "total_tokens": resp.input_tokens,
        },
    }


def _sse_from_chunks(first, gen, requested_model: str, created: int):
    """把网关的 StreamChunk 流逐块编码为 OpenAI 兼容 SSE 帧(真·流式,逐块下发)。

    first 是 server 预拉的首块(用于把"开流前"错误降级为普通 HTTP 状态);其余从 gen
    续取。中途异常会终止流(已发字节无法改 HTTP 状态)。客户端断开时 StreamingResponse
    会 close 本生成器 → finally 显式关闭网关流(触发其 aborted 捕获与资源释放)。
    """
    try:
        yield from _sse_frames(first, gen, requested_model, created)
    finally:
        gen.close()


def _sse_frames(first, gen, requested_model: str, created: int):
    import itertools
    import json as _json

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    def frame(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"

    yield frame({"role": "assistant"})
    final_fr = "stop"
    in_tok = out_tok = 0
    for chunk in itertools.chain([] if first is None else [first], gen):
        if chunk is None:
            continue
        if chunk.done:
            final_fr = stop_reason_to_finish(chunk.finish_reason) or "stop"
            in_tok, out_tok = chunk.input_tokens, chunk.output_tokens
            if chunk.tool_calls:
                # 工具调用合并为一帧下发(OpenAI 形;客户端可正常重组)
                yield frame(
                    {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for i, tc in enumerate(chunk.tool_calls)
                        ]
                    }
                )
        elif chunk.text:
            yield frame({"content": chunk.text})
    yield frame({}, finish_reason=final_fr)
    # usage 帧(OpenAI stream_options.include_usage 约定:choices 为空 + usage)。
    # 客户端据此在流式下拿到 token 计量,否则成本追踪在流式路径失效。
    if in_tok or out_tok:
        usage_payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [],
            "usage": {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
        }
        yield f"data: {_json.dumps(usage_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


_NO_FIRST = object()  # 流式预拉:生成器空(StopIteration)的哨兵


def _pull_first(gen):
    """在线程池里安全预拉首块:空生成器返回哨兵,网关异常照常抛出(供 await 处映射)。"""
    try:
        return next(gen)
    except StopIteration:
        return _NO_FIRST


def _error_body(message: str, err_type: str) -> dict:
    return {"error": {"message": message, "type": err_type, "code": err_type}}


def _request_too_large_response() -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content=_error_body("request too large", "request_too_large"),
    )


def _invalid_json_response() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=_error_body("invalid JSON body", "invalid_request_error"),
    )


async def _read_json_body(request: Request, max_request_bytes: int):
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > max_request_bytes:
            return None, _request_too_large_response()
        chunks.extend(chunk)
    try:
        return json.loads(chunks), None
    except (JSONDecodeError, UnicodeDecodeError):
        return None, _invalid_json_response()


def _gateway_error_response(exc: GatewayError) -> JSONResponse:
    """网关异常 → OpenAI 形 error 体 + 合适 HTTP 码(终态 400 / 预算 402 / 全失败 502)。"""
    if isinstance(exc, TerminalError):
        return JSONResponse(
            status_code=400, content=_error_body(str(exc), "terminal_error")
        )
    if isinstance(exc, BudgetExceeded):
        return JSONResponse(
            status_code=402, content=_error_body(str(exc), "budget_exceeded")
        )
    if isinstance(exc, AllTargetsFailed):
        return JSONResponse(
            status_code=502, content=_error_body(str(exc), "upstream_error")
        )
    return JSONResponse(status_code=400, content=_error_body(str(exc), "gateway_error"))


def build_server(
    gateway,
    models: list[str] | None = None,
    now=None,
    *,
    api_token: str | None = None,
    metrics_token: str | None = None,
    max_request_bytes: int = 1_000_000,
    rate_limit_per_minute: int = 120,
    trust_proxy_headers: bool = False,
    trusted_proxy_cidrs: list[str] | None = None,
    allow_direct_models: bool = True,
    default_max_tokens: int | None = None,
):
    """构造 OpenAI 兼容网关 FastAPI app。

    gateway: 已装配的 Gateway(注入,便于测试)。
    models: /v1/models 列出的模型名(可选)。
    now: 可注入的时间戳函数(测试用),默认 time.time。
    default_max_tokens: 请求未带 max_tokens 时的默认值;None 时读环境变量
        AGENT_CTL_DEFAULT_MAX_TOKENS,再缺省 1024(Anthropic 后端必填该字段,
        不能不设;但 1024 对长回答太小,故开放配置)。
    """
    from fastapi import FastAPI

    if default_max_tokens is None:
        default_max_tokens = int(os.getenv("AGENT_CTL_DEFAULT_MAX_TOKENS", "1024"))
    if default_max_tokens <= 0:
        raise ValueError("default_max_tokens must be a positive integer")
    clock = now or (lambda: int(time.time()))
    app = FastAPI(title="agent-ctl OpenAI-compatible gateway")
    listed = list(models or [])
    listed_set = set(listed)
    proxy_cidrs = (
        _DEFAULT_TRUSTED_PROXY_CIDRS
        if trust_proxy_headers and trusted_proxy_cidrs is None
        else trusted_proxy_cidrs or []
    )
    trusted_proxy_networks: tuple[IPv4Network | IPv6Network, ...] = tuple(
        ip_network(cidr, strict=False) for cidr in proxy_cidrs
    )

    def _direct_model_blocked(model: str):
        """禁止用 "provider/model" 直连未登记目标(绕过路由白名单 = 绕过成本治理)。"""
        if allow_direct_models or "/" not in model or model in listed_set:
            return None
        return JSONResponse(
            status_code=400,
            content=_error_body(
                f"direct 'provider/model' target {model!r} not allowed; "
                "use a configured route or alias",
                "invalid_request_error",
            ),
        )

    # LRU 有界:仅按时间裁剪每个 bucket 不清理空闲 IP 的 key → 公网多 IP 会无界增长。
    # 用 OrderedDict 钉住被追踪客户端数,超界淘汰最久未见的 IP。
    request_buckets: OrderedDict[str, deque[float]] = OrderedDict()
    auth_failure_buckets: OrderedDict[str, deque[float]] = OrderedDict()
    max_clients = 10_000

    @app.middleware("http")
    async def safety_middleware(request: Request, call_next):
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content=_error_body("invalid Content-Length", "invalid_request_error"),
            )
        if length > max_request_bytes:
            return JSONResponse(
                status_code=413,
                content=_error_body("request too large", "request_too_large"),
            )
        auth_error = _auth_error(request, api_token, metrics_token)
        if auth_error is not None:
            if rate_limit_per_minute > 0 and _rate_limited(
                auth_failure_buckets,
                _rate_limit_key(request, trust_proxy_headers, trusted_proxy_networks),
                rate_limit_per_minute,
                max_clients,
            ):
                return JSONResponse(
                    status_code=429,
                    content=_error_body("rate limit exceeded", "rate_limit_exceeded"),
                )
            return JSONResponse(status_code=401, content=auth_error)
        if (
            rate_limit_per_minute > 0
            and request.url.path not in {"/healthz", "/metrics"}
            and _rate_limited(
                request_buckets,
                _rate_limit_key(request, trust_proxy_headers, trusted_proxy_networks),
                rate_limit_per_minute,
                max_clients,
            )
        ):
            return JSONResponse(
                status_code=429,
                content=_error_body("rate limit exceeded", "rate_limit_exceeded"),
            )
        return await call_next(request)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics_endpoint():
        from fastapi.responses import Response

        from agent_ctl.obs import metrics

        content_type, body = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "owned_by": "agent-ctl"} for m in listed
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body, error_response = await _read_json_body(request, max_request_bytes)
        if error_response is not None:
            return error_response
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "request body must be an object", "invalid_request_error"
                ),
            )
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            return JSONResponse(
                status_code=400,
                content=_error_body("field 'model' required", "invalid_request_error"),
            )
        blocked = _direct_model_blocked(body["model"])
        if blocked is not None:
            return blocked
        try:
            req = to_normalized(body, default_max_tokens)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content=_error_body(str(exc), "invalid_request_error"),
            )
        created = clock()
        # 网关调用是阻塞 I/O(provider SDK 同步)→ 卸到线程池,避免卡死事件循环、
        # 让 server 在一次 LLM 调用期间仍能并发服务其他连接。
        if body.get("stream"):
            from fastapi.responses import StreamingResponse

            # 预拉首块(线程池):把"开流前"错误(预算/路由/终态/全失败)降级为普通 HTTP
            # 状态;首块已出后再发生的中途错误只能终止流(已发字节无法改状态码)。
            gen = gateway.invoke_stream(req)
            try:
                first = await run_in_threadpool(_pull_first, gen)
            except GatewayError as exc:
                return _gateway_error_response(exc)
            return StreamingResponse(
                _sse_from_chunks(
                    None if first is _NO_FIRST else first, gen, body["model"], created
                ),
                media_type="text/event-stream",
            )
        try:
            resp = await run_in_threadpool(gateway.invoke, req)
        except GatewayError as exc:
            return _gateway_error_response(exc)
        return to_openai_response(resp, body["model"], created)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        body, error_response = await _read_json_body(request, max_request_bytes)
        if error_response is not None:
            return error_response
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("model"), str)
            or not body["model"].strip()
        ):
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "fields 'model' and 'input' required", "invalid_request_error"
                ),
            )
        blocked = _direct_model_blocked(body["model"])
        if blocked is not None:
            return blocked
        raw_input = body.get("input")
        if isinstance(raw_input, str):
            inputs = [raw_input]
        elif isinstance(raw_input, list) and all(isinstance(x, str) for x in raw_input):
            inputs = raw_input
        else:
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "field 'input' must be a string or array of strings",
                    "invalid_request_error",
                ),
            )
        if not inputs:
            return JSONResponse(
                status_code=400,
                content=_error_body("field 'input' is empty", "invalid_request_error"),
            )
        try:
            resp = await run_in_threadpool(
                gateway.embed,
                body["model"],
                inputs,
                {"consumer": _consumer_of(body)},
            )
        except GatewayError as exc:
            return _gateway_error_response(exc)
        return to_openai_embeddings(resp, body["model"])

    return app


def _auth_error(
    request, api_token: str | None, metrics_token: str | None = None
) -> dict | None:
    if request.url.path == "/healthz":
        return None
    token = (
        (metrics_token or api_token) if request.url.path == "/metrics" else api_token
    )
    if not token:
        return None
    expected = f"Bearer {token}"
    actual = request.headers.get("authorization") or ""
    if not hmac.compare_digest(actual.encode(), expected.encode()):
        return _error_body("missing or invalid bearer token", "unauthorized")
    return None


def _rate_limited(
    buckets: OrderedDict[str, deque[float]],
    client: str,
    limit: int,
    max_clients: int,
) -> bool:
    now_ts = time.monotonic()
    bucket = buckets.get(client)
    if bucket is None:
        bucket = deque()
        buckets[client] = bucket
    buckets.move_to_end(client)  # 最近见过
    while bucket and now_ts - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= limit:
        return True
    bucket.append(now_ts)
    while len(buckets) > max_clients:
        buckets.popitem(last=False)  # 淘汰最久未见的客户端
    return False


def _rate_limit_key(
    request: Request,
    trust_proxy_headers: bool,
    trusted_proxy_networks: tuple[IPv4Network | IPv6Network, ...] = (),
) -> str:
    """限流键 = 真实客户端标识。

    XFF 最左值是客户端可任意伪造的(攻击者每请求换一个即绕过限流);可信的是
    **可信反代自己追加的右端条目**。因此从右往左跳过可信 CIDR 内的地址,取第一个
    不可信地址作为真实客户端;整条链全可信(内网直连)则回退 socket 对端。"""
    if trust_proxy_headers and _client_is_trusted_proxy(
        request, trusted_proxy_networks
    ):
        forwarded_for = request.headers.get("x-forwarded-for") or ""
        hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
        for hop in reversed(hops):
            if not _ip_in_networks(hop, trusted_proxy_networks):
                return f"xff:{hop}"
    return request.client.host if request.client else "unknown"


def _ip_in_networks(
    value: str, networks: tuple[IPv4Network | IPv6Network, ...]
) -> bool:
    try:
        addr = ip_address(value)
    except ValueError:
        return False  # 非法地址不可能是我们的可信反代 → 视为不可信(即真实客户端键)
    return any(addr in network for network in networks)


def _client_is_trusted_proxy(
    request: Request, trusted_proxy_networks: tuple[IPv4Network | IPv6Network, ...]
) -> bool:
    if not request.client:
        return False
    try:
        client_ip = ip_address(request.client.host)
    except ValueError:
        return False
    return any(client_ip in network for network in trusted_proxy_networks)
