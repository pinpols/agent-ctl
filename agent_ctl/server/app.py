from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from json import JSONDecodeError

from fastapi import Request
from fastapi.responses import JSONResponse

from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    GatewayError,
    TerminalError,
)
from agent_ctl.models import NormalizedRequest, NormalizedResponse


def to_normalized(body: dict) -> NormalizedRequest:
    """OpenAI /v1/chat/completions 请求体 → NormalizedRequest。

    OpenAI 把 system 作为 messages 里 role=system 的一条;抽出首条 system 放
    NormalizedRequest.system(AnthropicProvider 需独立 system,OpenAIProvider 会再塞回)。
    """
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("field 'messages' must be a list")
    system = None
    rest = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        if m.get("role") == "system" and system is None:
            system = m.get("content")
        else:
            rest.append(m)
    max_tokens = body.get("max_tokens") or 1024
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError("field 'max_tokens' must be an integer") from exc
    return NormalizedRequest(
        model=body["model"],
        messages=rest,
        system=system,
        max_tokens=max_tokens,
        temperature=body.get("temperature"),
        tools=body.get("tools"),
        tool_choice=body.get("tool_choice"),
        metadata={"consumer": "openai-compat-server"},
    )


def to_openai_response(
    resp: NormalizedResponse, requested_model: str, created: int
) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": resp.text},
                "finish_reason": resp.finish_reason or "stop",
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


def _sse_stream(resp: NormalizedResponse, requested_model: str, created: int):
    """把已完成的响应切成 OpenAI 兼容 SSE chunk(缓冲式:无 TTFB 收益,仅协议兼容)。

    真·逐 token 流式需各 provider 原生 streaming 且绕过治理层(成本/捕获/重试均依赖完整
    响应),代价远大于收益;此处以一段 content delta 还原 stream=true 客户端契约。
    """
    import json as _json

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    def chunk(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant"})
    if resp.text:
        yield chunk({"content": resp.text})
    yield chunk({}, finish_reason=resp.finish_reason or "stop")
    yield "data: [DONE]\n\n"


def _error_body(message: str, err_type: str) -> dict:
    return {"error": {"message": message, "type": err_type, "code": err_type}}


def build_server(
    gateway,
    models: list[str] | None = None,
    now=None,
    *,
    api_token: str | None = None,
    max_request_bytes: int = 1_000_000,
    rate_limit_per_minute: int = 120,
):
    """构造 OpenAI 兼容网关 FastAPI app。

    gateway: 已装配的 Gateway(注入,便于测试)。
    models: /v1/models 列出的模型名(可选)。
    now: 可注入的时间戳函数(测试用),默认 time.time。
    """
    from fastapi import FastAPI

    clock = now or (lambda: int(time.time()))
    app = FastAPI(title="agent-ctl OpenAI-compatible gateway")
    listed = list(models or [])
    request_times: defaultdict[str, deque[float]] = defaultdict(deque)

    @app.middleware("http")
    async def safety_middleware(request: Request, call_next):
        auth_error = _auth_error(request, api_token)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        length = int(request.headers.get("content-length") or 0)
        if length > max_request_bytes:
            return JSONResponse(
                status_code=413,
                content=_error_body("request too large", "request_too_large"),
            )
        if rate_limit_per_minute > 0:
            client = request.client.host if request.client else "unknown"
            now_ts = time.monotonic()
            bucket = request_times[client]
            while bucket and now_ts - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= rate_limit_per_minute:
                return JSONResponse(
                    status_code=429,
                    content=_error_body("rate limit exceeded", "rate_limit_exceeded"),
                )
            bucket.append(now_ts)
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
        raw = await request.body()
        if len(raw) > max_request_bytes:
            return JSONResponse(
                status_code=413,
                content=_error_body("request too large", "request_too_large"),
            )
        try:
            body = await request.json()
        except JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content=_error_body("invalid JSON body", "invalid_request_error"),
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "request body must be an object", "invalid_request_error"
                ),
            )
        if not body.get("model"):
            return JSONResponse(
                status_code=400,
                content=_error_body("field 'model' required", "invalid_request_error"),
            )
        try:
            req = to_normalized(body)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content=_error_body(str(exc), "invalid_request_error"),
            )
        # invoke 在流式前完成(治理:捕获/成本/熔断/重试都依赖完整响应),
        # 故所有错误仍以普通 HTTP 状态返回,流式仅切分已完成响应。
        try:
            resp = gateway.invoke(req)
        except TerminalError as exc:
            return JSONResponse(
                status_code=400, content=_error_body(str(exc), "terminal_error")
            )
        except BudgetExceeded as exc:
            return JSONResponse(
                status_code=402, content=_error_body(str(exc), "budget_exceeded")
            )
        except AllTargetsFailed as exc:
            return JSONResponse(
                status_code=502, content=_error_body(str(exc), "upstream_error")
            )
        except GatewayError as exc:
            return JSONResponse(
                status_code=400, content=_error_body(str(exc), "gateway_error")
            )
        if body.get("stream"):
            from fastapi.responses import StreamingResponse

            return StreamingResponse(
                _sse_stream(resp, body["model"], clock()),
                media_type="text/event-stream",
            )
        return to_openai_response(resp, body["model"], clock())

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        try:
            body = await request.json()
        except JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content=_error_body("invalid JSON body", "invalid_request_error"),
            )
        if not isinstance(body, dict) or not body.get("model"):
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "fields 'model' and 'input' required", "invalid_request_error"
                ),
            )
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
            resp = gateway.embed(
                body["model"], inputs, {"consumer": "openai-compat-server"}
            )
        except TerminalError as exc:
            return JSONResponse(
                status_code=400, content=_error_body(str(exc), "terminal_error")
            )
        except BudgetExceeded as exc:
            return JSONResponse(
                status_code=402, content=_error_body(str(exc), "budget_exceeded")
            )
        except AllTargetsFailed as exc:
            return JSONResponse(
                status_code=502, content=_error_body(str(exc), "upstream_error")
            )
        except GatewayError as exc:
            return JSONResponse(
                status_code=400, content=_error_body(str(exc), "gateway_error")
            )
        return to_openai_embeddings(resp, body["model"])

    return app


def _auth_error(request, api_token: str | None) -> dict | None:
    if not api_token:
        return None
    expected = f"Bearer {api_token}"
    if request.headers.get("authorization") != expected:
        return _error_body("missing or invalid bearer token", "unauthorized")
    return None
