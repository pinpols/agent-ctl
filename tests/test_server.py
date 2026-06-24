from fastapi.testclient import TestClient

from agent_ctl.errors import AllTargetsFailed, TerminalError
from agent_ctl.models import NormalizedRequest, NormalizedResponse
from agent_ctl.server.app import build_server, to_normalized, to_openai_response


class FakeGateway:
    """记录收到的 NormalizedRequest,按预设返回/抛错。"""

    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.last_request = None

    def invoke(self, request: NormalizedRequest) -> NormalizedResponse:
        self.last_request = request
        if self._exc:
            raise self._exc
        return self._resp


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


def test_chat_completions_streaming_rejected():
    c = _client(FakeGateway(resp=NormalizedResponse(text="")))
    r = c.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [], "stream": True},
    )
    assert r.status_code == 400


def test_chat_completions_terminal_error_maps_400():
    c = _client(FakeGateway(exc=TerminalError("bad key")))
    r = c.post("/v1/chat/completions", json={"model": "openai/gpt-4o", "messages": []})
    assert r.status_code == 400
    assert "bad key" in r.json()["error"]["message"]


def test_chat_completions_all_targets_failed_maps_502():
    c = _client(FakeGateway(exc=AllTargetsFailed("all down")))
    r = c.post("/v1/chat/completions", json={"model": "openai/gpt-4o", "messages": []})
    assert r.status_code == 502


def test_server_requires_bearer_token_when_configured():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            api_token="secret",
        )
    )
    r = c.get("/healthz")
    assert r.status_code == 401
    ok = c.get("/healthz", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


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


def test_server_rate_limits_by_client():
    c = TestClient(
        build_server(
            FakeGateway(resp=NormalizedResponse(text="")),
            now=lambda: 1234,
            rate_limit_per_minute=1,
        )
    )
    assert c.get("/healthz").status_code == 200
    assert c.get("/healthz").status_code == 429
