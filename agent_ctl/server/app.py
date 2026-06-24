from __future__ import annotations

import time
import uuid

from agent_ctl.errors import AllTargetsFailed, GatewayError, TerminalError
from agent_ctl.models import NormalizedRequest, NormalizedResponse


def to_normalized(body: dict) -> NormalizedRequest:
    """OpenAI /v1/chat/completions 请求体 → NormalizedRequest。

    OpenAI 把 system 作为 messages 里 role=system 的一条;抽出首条 system 放
    NormalizedRequest.system(AnthropicProvider 需独立 system,OpenAIProvider 会再塞回)。
    """
    messages = list(body.get("messages") or [])
    system = None
    rest = []
    for m in messages:
        if m.get("role") == "system" and system is None:
            system = m.get("content")
        else:
            rest.append(m)
    return NormalizedRequest(
        model=body["model"],
        messages=rest,
        system=system,
        max_tokens=int(body.get("max_tokens") or 1024),
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


def _error_body(message: str, err_type: str) -> dict:
    return {"error": {"message": message, "type": err_type, "code": err_type}}


def build_server(gateway, models: list[str] | None = None, now=None):
    """构造 OpenAI 兼容网关 FastAPI app。

    gateway: 已装配的 Gateway(注入,便于测试)。
    models: /v1/models 列出的模型名(可选)。
    now: 可注入的时间戳函数(测试用),默认 time.time。
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    clock = now or (lambda: int(time.time()))
    app = FastAPI(title="agent-ctl OpenAI-compatible gateway")
    listed = list(models or [])

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "owned_by": "agent-ctl"} for m in listed
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict):
        if not body.get("model"):
            return JSONResponse(
                status_code=400,
                content=_error_body("field 'model' required", "invalid_request_error"),
            )
        if body.get("stream"):
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "streaming not supported yet", "invalid_request_error"
                ),
            )
        req = to_normalized(body)
        try:
            resp = gateway.invoke(req)
        except TerminalError as exc:
            return JSONResponse(
                status_code=400, content=_error_body(str(exc), "terminal_error")
            )
        except AllTargetsFailed as exc:
            return JSONResponse(
                status_code=502, content=_error_body(str(exc), "upstream_error")
            )
        except GatewayError as exc:
            return JSONResponse(
                status_code=400, content=_error_body(str(exc), "gateway_error")
            )
        return to_openai_response(resp, body["model"], clock())

    return app
