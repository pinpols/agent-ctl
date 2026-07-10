from fastapi.testclient import TestClient

from agent_ctl.errors import AllTargetsFailed, BudgetExceeded, TerminalError
from agent_ctl.models import (
    EmbeddingResponse,
    NormalizedRequest,
    NormalizedResponse,
    StreamChunk,
)
from agent_ctl.server.app import build_server, to_normalized, to_openai_response


class FakeGateway:
    """记录收到的 NormalizedRequest,按预设返回/抛错。"""

    def __init__(
        self,
        resp=None,
        exc=None,
        embed_resp=None,
        embed_exc=None,
        stream_chunks=None,
        stream_exc=None,
        stream_tool_calls=None,
    ):
        self._resp = resp
        self._exc = exc
        self._embed_resp = embed_resp
        self._embed_exc = embed_exc
        self._stream_chunks = stream_chunks  # list[str] 文本增量
        self._stream_exc = stream_exc
        self._stream_tool_calls = stream_tool_calls
        self.last_request = None
        self.last_embed = None

    def invoke(self, request: NormalizedRequest) -> NormalizedResponse:
        self.last_request = request
        if self._exc:
            raise self._exc
        return self._resp

    def invoke_stream(self, request: NormalizedRequest):
        self.last_request = request
        if self._stream_exc:
            raise self._stream_exc
        for text in self._stream_chunks or ([self._resp.text] if self._resp else []):
            if text:
                yield StreamChunk(text=text)
        r = self._resp
        yield StreamChunk(
            done=True,
            finish_reason=(r.finish_reason if r else None),
            input_tokens=(r.input_tokens if r else 0),
            output_tokens=(r.output_tokens if r else 0),
            tool_calls=self._stream_tool_calls,
        )

    def embed(self, model, inputs, metadata=None) -> EmbeddingResponse:
        self.last_embed = (model, inputs)
        if self._embed_exc:
            raise self._embed_exc
        return self._embed_resp


def _client(gateway, models=None):
    return TestClient(build_server(gateway, models=models, now=lambda: 1234))


# ── 翻译纯函数 ──────────────────────────────────────────────


def test_to_normalized_extracts_system():
    req = to_normalized(
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 99,
            "temperature": 0.3,
        }
    )
    assert req.model == "deepseek-chat"
    assert req.system == "你是助手"
    assert req.messages == [{"role": "user", "content": "hi"}]
    assert req.max_tokens == 99
    assert req.temperature == 0.3


def test_to_openai_response_surfaces_tool_calls():
    """H7:非流式响应从 raw 还原 OpenAI message.tool_calls(否则 HTTP 工具调用拿不到)。"""
    import json

    resp = NormalizedResponse(
        text="",
        finish_reason="tool_calls",
        raw={
            "content": [
                {"type": "tool_use", "id": "t1", "name": "diagnose", "input": {"x": 1}}
            ]
        },
    )
    out = to_openai_response(resp, "m", created=1)
    msg = out["choices"][0]["message"]
    assert msg["content"] is None  # 有工具调用且无文本 → content=null
    assert msg["tool_calls"][0]["id"] == "t1"
    assert msg["tool_calls"][0]["function"]["name"] == "diagnose"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_to_openai_response_no_tool_calls_unchanged():
    """无工具调用时形状不变(content 原样,无 tool_calls 键)。"""
    resp = NormalizedResponse(text="答案", finish_reason="stop")
    msg = to_openai_response(resp, "m", created=1)["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "答案"}


def test_to_openai_response_shape():
    resp = NormalizedResponse(
        text="答案",
        finish_reason="stop",
        input_tokens=7,
        output_tokens=3,
    )
    out = to_openai_response(resp, "deepseek-chat", created=1234)
    assert out["object"] == "chat.completion"
    assert out["model"] == "deepseek-chat"  # 回显请求的 model
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "答案"}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


# ── server 端点 ─────────────────────────────────────────────


def test_healthz():
    c = _client(FakeGateway(resp=NormalizedResponse(text="")))
    assert c.get("/healthz").json() == {"status": "ok"}


def test_models_list():
    c = _client(
        FakeGateway(resp=NormalizedResponse(text="")),
        models=["claude-opus-4-8", "glm-4"],
    )
    data = c.get("/v1/models").json()
    assert data["object"] == "list"
    assert {m["id"] for m in data["data"]} == {"claude-opus-4-8", "glm-4"}


def test_chat_completions_success_and_routes_model():
    gw = FakeGateway(
        resp=NormalizedResponse(
            text="hello", finish_reason="stop", input_tokens=5, output_tokens=2
        )
    )
    c = _client(gw)
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "qwen/qwen-max",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    # 网关收到的 request.model 即请求的 model(由 Router 解析 provider/model)
    assert gw.last_request.model == "qwen/qwen-max"


def test_chat_completions_missing_model_400():
    c = _client(FakeGateway(resp=NormalizedResponse(text="")))
    r = c.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_chat_completions_malformed_messages_400():
    c = _client(FakeGateway(resp=NormalizedResponse(text="")))
    r = c.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": ["bad"]},
    )
    assert r.status_code == 400
    assert "message" in r.json()["error"]["message"]


def test_chat_completions_bad_max_tokens_400():
    c = _client(FakeGateway(resp=NormalizedResponse(text="")))
    r = c.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [], "max_tokens": "abc"},
    )
    assert r.status_code == 400
    assert "max_tokens" in r.json()["error"]["message"]


def test_chat_completions_rejects_non_positive_max_tokens():
    c = _client(FakeGateway(resp=NormalizedResponse(text="")))
    for value in (0, -1):
        r = c.post(
            "/v1/chat/completions",
            json={"model": "openai/gpt-4o", "messages": [], "max_tokens": value},
        )
        assert r.status_code == 400


def test_chat_completions_rejects_bad_temperature():
    c = _client(FakeGateway(resp=NormalizedResponse(text="")))
    r = c.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [], "temperature": 3},
    )
    assert r.status_code == 400
    assert "temperature" in r.json()["error"]["message"]


def test_chat_completions_streaming_emits_multiple_sse_chunks():
    """真流式:多段文本增量 → 多个 content 帧逐块下发。"""
    gw = FakeGateway(
        resp=NormalizedResponse(
            text="", finish_reason="stop", input_tokens=5, output_tokens=2
        ),
        stream_chunks=["你", "好", "世界"],
    )
    c = _client(gw)
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert '"object": "chat.completion.chunk"' in text
    assert '"role": "assistant"' in text
    assert text.count('"content":') == 3  # 三段增量各一帧
    assert '"finish_reason": "stop"' in text
    assert text.rstrip().endswith("data: [DONE]")


def test_streaming_pre_open_error_returns_http_status_not_stream():
    """开流前的错误(预拉首块时抛)降级为普通 HTTP 状态(502),而非半截流。"""
    c = _client(FakeGateway(stream_exc=AllTargetsFailed("all down")))
    r = c.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [], "stream": True},
    )
    assert r.status_code == 502


def test_streaming_budget_error_maps_402():
    c = _client(FakeGateway(stream_exc=BudgetExceeded("over budget")))
    r = c.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [], "stream": True},
    )
    assert r.status_code == 402


def test_handlers_offload_blocking_calls_to_threadpool():
    """H1:阻塞网关调用卸到线程池 → 并发请求不被事件循环串行化。"""
    import asyncio
    import time as _time

    import httpx
    from httpx import ASGITransport

    class SlowGateway:
        def invoke(self, req):
            _time.sleep(0.2)  # 模拟阻塞 I/O
            return NormalizedResponse(text="ok")

    app = build_server(SlowGateway(), now=lambda: 1, rate_limit_per_minute=0)

    async def _run():
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            start = _time.monotonic()
            await asyncio.gather(
                *[
                    client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                    for _ in range(5)
                ]
            )
            return _time.monotonic() - start

    elapsed = asyncio.run(_run())
    # 5 个各 0.2s 阻塞调用:线程池并发 ~0.2-0.4s;若卡事件循环则串行 ~1.0s+
    assert elapsed < 0.7


def test_streaming_emits_tool_calls_frame():
    """G5:末块带 tool_calls → SSE 下发 OpenAI 形 tool_calls 帧 + finish_reason=tool_calls。"""
    gw = FakeGateway(
        resp=NormalizedResponse(text="", finish_reason="tool_calls"),
        stream_chunks=[],
        stream_tool_calls=[{"id": "c1", "name": "diagnose", "arguments": '{"x":1}'}],
    )
    c = _client(gw)
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "诊断"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    text = r.text
    assert '"tool_calls"' in text
    assert '"name": "diagnose"' in text
    assert '"finish_reason": "tool_calls"' in text
    assert text.rstrip().endswith("data: [DONE]")


# ── /v1/embeddings ──────────────────────────────────────────


def test_embeddings_success_string_input():
    gw = FakeGateway(
        embed_resp=EmbeddingResponse(vectors=[[0.1, 0.2, 0.3]], input_tokens=4)
    )
    c = _client(gw)
    r = c.post("/v1/embeddings", json={"model": "deepseek/x", "input": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert body["data"][0]["index"] == 0
    assert body["usage"]["prompt_tokens"] == 4
    assert gw.last_embed == ("deepseek/x", ["hello"])  # str 归一为单元素列表


def test_embeddings_success_list_input_preserves_order():
    gw = FakeGateway(
        embed_resp=EmbeddingResponse(vectors=[[1.0], [2.0]], input_tokens=8)
    )
    c = _client(gw)
    r = c.post("/v1/embeddings", json={"model": "deepseek/x", "input": ["a", "b"]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1]
    assert [d["embedding"] for d in data] == [[1.0], [2.0]]


def test_embeddings_missing_input_400():
    c = _client(FakeGateway())
    r = c.post("/v1/embeddings", json={"model": "deepseek/x"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_embeddings_empty_input_400():
    c = _client(FakeGateway())
    r = c.post("/v1/embeddings", json={"model": "deepseek/x", "input": []})
    assert r.status_code == 400


def test_embeddings_non_string_input_400():
    c = _client(FakeGateway())
    r = c.post("/v1/embeddings", json={"model": "deepseek/x", "input": [1, 2]})
    assert r.status_code == 400


def test_embeddings_upstream_failure_maps_502():
    c = _client(FakeGateway(embed_exc=AllTargetsFailed("no embed provider")))
    r = c.post("/v1/embeddings", json={"model": "deepseek/x", "input": "hi"})
    assert r.status_code == 502


def test_chat_completions_terminal_error_maps_400():
    c = _client(FakeGateway(exc=TerminalError("bad key")))
    r = c.post("/v1/chat/completions", json={"model": "openai/gpt-4o", "messages": []})
    assert r.status_code == 400
    assert "bad key" in r.json()["error"]["message"]


def test_chat_completions_all_targets_failed_maps_502():
    c = _client(FakeGateway(exc=AllTargetsFailed("all down")))
    r = c.post("/v1/chat/completions", json={"model": "openai/gpt-4o", "messages": []})
    assert r.status_code == 502


def test_chat_completions_budget_exceeded_maps_402():
    c = _client(FakeGateway(exc=BudgetExceeded("budget exhausted")))
    r = c.post("/v1/chat/completions", json={"model": "openai/gpt-4o", "messages": []})
    assert r.status_code == 402
    assert r.json()["error"]["type"] == "budget_exceeded"


def test_server_requires_bearer_token_when_configured():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            api_token="secret",
        )
    )
    r = c.get("/v1/models")
    assert r.status_code == 401
    assert c.get("/healthz").status_code == 200
    ok = c.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_metrics_can_use_separate_token():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            api_token="api-secret",
            metrics_token="metrics-secret",
            rate_limit_per_minute=0,
        )
    )
    assert (
        c.get("/metrics", headers={"Authorization": "Bearer api-secret"}).status_code
        == 401
    )
    assert (
        c.get(
            "/metrics", headers={"Authorization": "Bearer metrics-secret"}
        ).status_code
        == 200
    )


def test_metrics_auth_failures_consume_rate_limit():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            api_token="api-secret",
            metrics_token="metrics-secret",
            rate_limit_per_minute=1,
        )
    )
    assert (
        c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    )
    assert (
        c.get("/metrics", headers={"Authorization": "Bearer wrong2"}).status_code == 429
    )


def test_metrics_successes_skip_business_rate_limit():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            api_token="api-secret",
            metrics_token="metrics-secret",
            rate_limit_per_minute=1,
        )
    )
    headers = {"Authorization": "Bearer metrics-secret"}
    assert c.get("/metrics", headers=headers).status_code == 200
    assert c.get("/metrics", headers=headers).status_code == 200


def test_metrics_falls_back_to_api_token_when_no_metrics_token():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            api_token="api-secret",
            rate_limit_per_minute=0,
        )
    )
    assert c.get("/metrics").status_code == 401
    assert (
        c.get("/metrics", headers={"Authorization": "Bearer api-secret"}).status_code
        == 200
    )


def test_server_rejects_large_request():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            max_request_bytes=10,
        )
    )
    r = c.post(
        "/v1/chat/completions",
        headers={"Content-Length": "11"},
        json={"model": "openai/gpt-4o", "messages": []},
    )
    assert r.status_code == 413


def test_server_rejects_invalid_content_length():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
        )
    )
    r = c.post(
        "/v1/chat/completions",
        content=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": "bad"},
    )
    assert r.status_code == 400


def test_server_rejects_large_request_even_with_low_content_length_header():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            max_request_bytes=30,
        )
    )
    body = b'{"model":"m","messages":[],"pad":"' + b"x" * 100 + b'"}'
    r = c.post(
        "/v1/chat/completions",
        content=body,
        headers={"Content-Type": "application/json", "Content-Length": "1"},
    )
    assert r.status_code == 413


def test_embeddings_rejects_large_request_even_with_low_content_length_header():
    c = TestClient(
        build_server(
            FakeGateway(embed_resp=EmbeddingResponse(vectors=[[0.1]], input_tokens=1)),
            now=lambda: 1234,
            max_request_bytes=30,
        )
    )
    body = b'{"model":"m","input":"' + b"x" * 100 + b'"}'
    r = c.post(
        "/v1/embeddings",
        content=body,
        headers={"Content-Type": "application/json", "Content-Length": "1"},
    )
    assert r.status_code == 413


def test_server_rate_limits_by_client():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            rate_limit_per_minute=1,
        )
    )
    assert c.get("/v1/models").status_code == 200
    assert c.get("/v1/models").status_code == 429


def test_auth_failures_consume_rate_limit():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            api_token="secret",
            rate_limit_per_minute=1,
        )
    )
    assert (
        c.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert (
        c.get("/v1/models", headers={"Authorization": "Bearer wrong2"}).status_code
        == 429
    )
    assert (
        c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code
        == 200
    )


def test_healthz_skips_auth_and_rate_limit():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            api_token="secret",
            rate_limit_per_minute=1,
        )
    )
    assert c.get("/healthz").status_code == 200
    assert c.get("/healthz").status_code == 200


def test_rate_limit_can_trust_forwarded_for():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            rate_limit_per_minute=1,
            trust_proxy_headers=True,
        ),
        client=("127.0.0.1", 12345),
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 200
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 200
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 429
    )


def test_rate_limit_ignores_forwarded_for_from_untrusted_proxy():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            rate_limit_per_minute=1,
            trust_proxy_headers=True,
            trusted_proxy_cidrs=["10.0.0.0/8"],
        ),
        client=("192.0.2.10", 12345),
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "198.51.100.1"}).status_code
        == 200
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "198.51.100.2"}).status_code
        == 429
    )


# ── 深审 5 修回归 ───────────────────────────────────────────


def test_to_normalized_merges_multiple_system_messages():
    """⑤ 多条 system 全部合并(漏合并会把第 2 条当普通消息发给 Anthropic 报错)。"""
    req = to_normalized(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "B"},
            ],
        }
    )
    assert req.system == "A\n\nB"
    assert req.messages == [
        {"role": "user", "content": "hi"}
    ]  # system 不再漏进 messages


def test_to_normalized_consumer_from_user_field():
    """② consumer 取自 OpenAI `user` 字段 → per-consumer 预算/归因生效。"""
    assert (
        to_normalized({"model": "m", "messages": [], "user": "alice"}).metadata[
            "consumer"
        ]
        == "alice"
    )
    assert (
        to_normalized({"model": "m", "messages": []}).metadata["consumer"]
        == "openai-compat-server"
    )


def test_streaming_emits_usage_frame():
    """① 流式末尾发 usage 帧(stream_options.include_usage 客户端据此拿 token)。"""
    gw = FakeGateway(
        resp=NormalizedResponse(
            text="hi", finish_reason="stop", input_tokens=7, output_tokens=3
        ),
        stream_chunks=["hi"],
    )
    c = _client(gw)
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    text = r.text
    assert '"usage"' in text
    assert '"prompt_tokens": 7' in text
    assert '"completion_tokens": 3' in text
    assert '"total_tokens": 10' in text
    assert text.rstrip().endswith("data: [DONE]")


def test_direct_model_blocked_when_disallowed():
    """③ allow_direct_models=False 时,provider/model 直连未登记目标被拒(防绕过路由白名单)。"""
    gw = FakeGateway(resp=NormalizedResponse(text="ok"))
    app = build_server(gw, models=["chat"], now=lambda: 1, allow_direct_models=False)
    c = TestClient(app)
    # 直连未登记目标 → 400
    r = c.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 400
    assert "not allowed" in r.json()["error"]["message"]
    # 登记的逻辑名 → 放行
    ok = c.post(
        "/v1/chat/completions",
        json={"model": "chat", "messages": [{"role": "user", "content": "x"}]},
    )
    assert ok.status_code == 200


def test_direct_model_allowed_by_default():
    """默认 allow_direct_models=True(库形态 build_server)不影响现有直连用法。"""
    gw = FakeGateway(resp=NormalizedResponse(text="ok"))
    c = _client(gw)  # 默认 True
    r = c.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 200


# ── 深审 round4 ────────────────────────────────────────────


def test_str_tool_choice_passes_end_to_end():
    """P1-1:"auto"/"required"/"none" 是合法 OpenAI tool_choice,须穿过 server→gateway。"""
    gw = FakeGateway(resp=NormalizedResponse(text="ok"))
    c = _client(gw)
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required",
        },
    )
    assert r.status_code == 200
    assert gw.last_request.tool_choice == "required"


def test_anthropic_stop_reason_mapped_to_finish_reason():
    """P1-3:Anthropic 后端透出的 stop_reason 须映射回 OpenAI finish_reason。"""
    for stop_reason, expected in [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
    ]:
        gw = FakeGateway(resp=NormalizedResponse(text="t", finish_reason=stop_reason))
        r = _client(gw).post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.json()["choices"][0]["finish_reason"] == expected


def test_stream_anthropic_stop_reason_mapped():
    import json as _json

    gw = FakeGateway(
        resp=NormalizedResponse(text="hi", finish_reason="end_turn"),
        stream_chunks=["hi"],
    )
    r = _client(gw).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    frames = [
        _json.loads(line[len("data: ") :])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    finishes = [
        f["choices"][0]["finish_reason"]
        for f in frames
        if f.get("choices") and f["choices"][0]["finish_reason"]
    ]
    assert finishes == ["stop"]  # end_turn → stop


def test_openai_server_to_anthropic_backend_end_to_end():
    """P1-3 端到端:OpenAI 形请求穿 server → Gateway → AnthropicProvider(fake client),
    工具/消息在边界翻成 Anthropic 形,anthropic 形响应翻回 OpenAI 形。"""
    from agent_ctl.config import RetryConfig
    from agent_ctl.core.cost import CostMeter
    from agent_ctl.core.gateway import Gateway
    from agent_ctl.core.router import Router
    from agent_ctl.providers.anthropic_provider import AnthropicProvider

    captured = {}

    class _Msg:
        content = [
            type("B", (), {"type": "tool_use", "id": "t1", "name": "f", "input": {}})()
        ]
        stop_reason = "tool_use"
        usage = type("U", (), {"input_tokens": 7, "output_tokens": 3})()

        def model_dump(self, mode="python"):
            return {
                "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
                "stop_reason": "tool_use",
            }

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Msg()

    gw = Gateway(
        router=Router({"m": ["anthropic/claude-x"]}),
        providers={
            "anthropic": AnthropicProvider(type("C", (), {"messages": _Messages()})())
        },
        cost_meter=CostMeter({}),
        store=None,
        retry=RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=5.0),
    )
    c = TestClient(build_server(gw, now=lambda: 1))
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto",
        },
    )
    assert r.status_code == 200
    # 请求方向:OpenAI 形 → Anthropic 形
    assert captured["tools"][0] == {
        "name": "f",
        "description": "",
        "input_schema": {"type": "object", "properties": {}},
    }
    assert captured["tool_choice"] == {"type": "auto"}
    # 响应方向:stop_reason=tool_use → finish_reason=tool_calls,tool_use 块 → tool_calls
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"


def _anthropic_gateway(create_fn):
    from agent_ctl.config import RetryConfig
    from agent_ctl.core.cost import CostMeter
    from agent_ctl.core.gateway import Gateway
    from agent_ctl.core.router import Router
    from agent_ctl.providers.anthropic_provider import AnthropicProvider

    messages_api = type("M", (), {"create": staticmethod(create_fn)})()
    return Gateway(
        router=Router({"m": ["anthropic/claude-x"]}),
        providers={
            "anthropic": AnthropicProvider(type("C", (), {"messages": messages_api})())
        },
        cost_meter=CostMeter({}),
        store=None,
        retry=RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=5.0),
    )


def test_image_url_message_converted_and_forwarded():
    """P1-c 端到端:image_url 不再 400——翻成 Anthropic image 块直通后端。"""
    captured = {}

    class _Msg:
        content = [type("B", (), {"type": "text", "text": "看到了"})()]
        stop_reason = "end_turn"
        usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()

    def create(**kwargs):
        captured.update(kwargs)
        return _Msg()

    c = TestClient(build_server(_anthropic_gateway(create), now=lambda: 1))
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "看图"},
                        {"type": "image_url", "image_url": {"url": "https://x/1.png"}},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 200
    assert captured["messages"][0]["content"][1] == {
        "type": "image",
        "source": {"type": "url", "url": "https://x/1.png"},
    }


def test_input_audio_message_returns_400_without_reaching_provider():
    """P1-b 端到端:本地拒绝在 HTTP 边界 400,不打 provider(更不污染熔断)。"""

    def create(**kwargs):
        raise AssertionError("不应打到 provider")

    c = TestClient(build_server(_anthropic_gateway(create), now=lambda: 1))
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {}}],
                }
            ],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "input_audio" in r.json()["error"]["message"]


def test_unknown_tool_choice_string_returns_400():
    def create(**kwargs):
        raise AssertionError("不应打到 provider")

    c = TestClient(build_server(_anthropic_gateway(create), now=lambda: 1))
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "weird",
        },
    )
    assert r.status_code == 400
    assert "tool_choice" in r.json()["error"]["message"]


def test_sse_generator_close_closes_gateway_stream():
    """P2-10:客户端断开(StreamingResponse close SSE 生成器)须连带关闭网关流。"""
    from agent_ctl.server.app import _sse_from_chunks

    closed = []

    def gw_stream():
        try:
            yield StreamChunk(text="a")
            yield StreamChunk(text="b")
            yield StreamChunk(done=True, finish_reason="stop")
        finally:
            closed.append(True)

    gen = gw_stream()
    first = next(gen)
    sse = _sse_from_chunks(first, gen, "m", created=1)
    next(sse)  # role 帧
    next(sse)  # 首块内容帧
    sse.close()  # 模拟客户端断开
    assert closed == [True]


def test_xff_leftmost_spoof_cannot_evade_rate_limit():
    """P1-5:XFF 最左值客户端可伪造;须从右往左取第一个不在可信 CIDR 的地址。
    攻击者每次换最左伪造值,真实客户端(右侧)不变 → 仍应命中同一限流桶。"""
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            rate_limit_per_minute=1,
            trust_proxy_headers=True,
        ),
        client=("127.0.0.1", 12345),
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "1.1.1.1, 9.9.9.9"}).status_code
        == 200
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "2.2.2.2, 9.9.9.9"}).status_code
        == 429
    )


def test_xff_skips_trusted_intermediate_hops():
    """多跳:右起跳过可信反代地址,取第一个不可信地址作为真实客户端。"""
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            rate_limit_per_minute=1,
            trust_proxy_headers=True,
            trusted_proxy_cidrs=["127.0.0.1/32", "10.0.0.0/8"],
        ),
        client=("127.0.0.1", 12345),
    )
    hdr1 = {"X-Forwarded-For": "spoofed-a, 9.9.9.9, 10.0.0.5"}
    hdr2 = {"X-Forwarded-For": "spoofed-b, 9.9.9.9, 10.0.0.7"}
    assert c.get("/v1/models", headers=hdr1).status_code == 200
    assert (
        c.get("/v1/models", headers=hdr2).status_code == 429
    )  # 同一真实客户端 9.9.9.9


def test_xff_all_hops_trusted_falls_back_to_socket_peer():
    """整条链全是可信地址(内网直连)→ 回退用 socket 对端,不因空选择而崩。"""
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            rate_limit_per_minute=1,
            trust_proxy_headers=True,
            trusted_proxy_cidrs=["127.0.0.1/32", "10.0.0.0/8"],
        ),
        client=("127.0.0.1", 12345),
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 200
    )
    assert (
        c.get("/v1/models", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 429
    )


def test_default_max_tokens_configurable(monkeypatch):
    """P2-8:强加的 max_tokens=1024 默认可经参数/环境变量配置;显式请求值仍优先。"""
    gw = FakeGateway(resp=NormalizedResponse(text=""))
    c = TestClient(build_server(gw, now=lambda: 1, default_max_tokens=4096))
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    c.post("/v1/chat/completions", json=body)
    assert gw.last_request.max_tokens == 4096
    c.post("/v1/chat/completions", json={**body, "max_tokens": 7})
    assert gw.last_request.max_tokens == 7  # 显式值优先

    monkeypatch.setenv("AGENT_CTL_DEFAULT_MAX_TOKENS", "2048")
    c2 = TestClient(build_server(gw, now=lambda: 1))  # 未传参 → 读环境变量
    c2.post("/v1/chat/completions", json=body)
    assert gw.last_request.max_tokens == 2048
