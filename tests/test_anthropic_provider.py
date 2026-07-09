import pytest

from agent_ctl.errors import RetriableError, TerminalError
from agent_ctl.providers.anthropic_provider import AnthropicProvider, classify_status
from agent_ctl.models import Target, NormalizedRequest

REQ = NormalizedRequest(
    model="default", messages=[{"role": "user", "content": "hi"}], max_tokens=64
)
T = Target(provider="anthropic", model="claude-opus-4-8")


class _FakeMessages:
    def __init__(self, behavior):
        self._b = behavior
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._b == "ok":

            class R:
                content = [type("B", (), {"type": "text", "text": "hello"})()]
                stop_reason = "end_turn"
                usage = type("U", (), {"input_tokens": 7, "output_tokens": 3})()

            return R()
        raise RuntimeError(self._b)


class _FakeClient:
    def __init__(self, behavior):
        self.messages = _FakeMessages(behavior)


def test_invoke_maps_response():
    p = AnthropicProvider(_FakeClient("ok"))
    resp = p.invoke(T, REQ, timeout=5.0)
    assert resp.text == "hello"
    assert resp.input_tokens == 7
    assert resp.finish_reason == "end_turn"


def test_invoke_passes_timeout_to_sdk():
    client = _FakeClient("ok")
    AnthropicProvider(client).invoke(T, REQ, timeout=12.5)
    assert client.messages.last_kwargs["timeout"] == 12.5


def test_invoke_passes_system_and_tool_choice():
    client = _FakeClient("ok")
    req = NormalizedRequest(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        system="你是 SRE",
        tool_choice={"type": "tool", "name": "report"},
        tools=[{"name": "report"}],
    )
    AnthropicProvider(client).invoke(T, req, timeout=5.0)
    kw = client.messages.last_kwargs
    assert kw["system"] == "你是 SRE"
    assert kw["tool_choice"] == {"type": "tool", "name": "report"}
    assert kw["tools"] == [{"name": "report"}]


def test_invoke_omits_system_tool_choice_when_absent():
    # 纯文本路由消费者:不传 system/tool_choice → kwargs 里不出现(向后兼容)
    client = _FakeClient("ok")
    AnthropicProvider(client).invoke(T, REQ, timeout=5.0)
    assert "system" not in client.messages.last_kwargs
    assert "tool_choice" not in client.messages.last_kwargs


def test_invoke_populates_raw_when_sdk_supports_model_dump():
    class _Msg:
        content = [type("B", (), {"type": "text", "text": "hi"})()]
        stop_reason = "end_turn"
        usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()

        def model_dump(self, mode="python"):
            return {
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
            }

    class _Client:
        messages = type("M", (), {"create": staticmethod(lambda **k: _Msg())})()

    resp = AnthropicProvider(_Client()).invoke(T, REQ, timeout=5.0)
    assert resp.raw == {
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
    }


def _ev(etype, **attrs):
    return type("Ev", (), {"type": etype, **attrs})()


class _StreamMessages:
    def create(self, **kwargs):
        assert kwargs["stream"] is True
        msg = type("M", (), {"usage": type("U", (), {"input_tokens": 9})()})()
        text_delta = type("D", (), {"type": "text_delta", "text": "Hi "})()
        text_delta2 = type("D", (), {"type": "text_delta", "text": "there"})()
        msg_delta = type("D", (), {"stop_reason": "end_turn"})()
        return iter(
            [
                _ev("message_start", message=msg),
                _ev("content_block_delta", delta=text_delta),
                _ev("content_block_delta", delta=text_delta2),
                _ev(
                    "message_delta",
                    delta=msg_delta,
                    usage=type("U", (), {"output_tokens": 6})(),
                ),
                _ev("message_stop"),
            ]
        )


def test_stream_parses_anthropic_events():
    client = type("C", (), {"messages": _StreamMessages()})()
    chunks = list(AnthropicProvider(client).stream(T, REQ, timeout=5.0))
    assert [c.text for c in chunks if not c.done] == ["Hi ", "there"]
    done = chunks[-1]
    assert done.done and done.finish_reason == "end_turn"
    assert done.input_tokens == 9 and done.output_tokens == 6


def test_stream_connect_error_is_typed():
    p = AnthropicProvider(_FakeClientStatus(503))
    with pytest.raises(RetriableError):
        list(p.stream(T, REQ, timeout=5.0))


def test_classify_status():
    assert classify_status(429) == "retriable"
    assert classify_status(529) == "retriable"
    assert classify_status(500) == "retriable"
    assert classify_status(401) == "terminal"
    assert classify_status(400) == "terminal"


class _StatusError(Exception):
    """携带 status_code 属性的假 SDK 异常。"""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeMessagesStatus:
    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    def create(self, **kwargs):
        raise _StatusError(self._status_code)


class _FakeClientStatus:
    def __init__(self, status_code: int) -> None:
        self.messages = _FakeMessagesStatus(status_code)


def test_status_401_raises_terminal_error():
    """401 应映射为 TerminalError(终态,不重试)。"""
    p = AnthropicProvider(_FakeClientStatus(401))
    with pytest.raises(TerminalError):
        p.invoke(T, REQ, timeout=5.0)


def test_status_503_raises_retriable_error():
    """503 应映射为 RetriableError(可重试)。"""
    p = AnthropicProvider(_FakeClientStatus(503))
    with pytest.raises(RetriableError):
        p.invoke(T, REQ, timeout=5.0)


# ── 深审 round4 P1-3:OpenAI 形请求在边界翻成 Anthropic 形 ──


def test_invoke_translates_openai_shaped_tools_and_messages():
    client = _FakeClient("ok")
    req = NormalizedRequest(
        model="default",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"x":1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
        max_tokens=64,
        tools=[
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="required",
    )
    AnthropicProvider(client).invoke(T, req, timeout=5.0)
    kw = client.messages.last_kwargs
    assert kw["tools"] == [
        {"name": "f", "description": "", "input_schema": {"type": "object"}}
    ]
    assert kw["tool_choice"] == {"type": "any"}
    assert kw["messages"][1]["content"][0]["type"] == "tool_use"
    assert kw["messages"][2]["content"][0]["type"] == "tool_result"


def test_invoke_anthropic_shaped_request_passes_through_unchanged():
    client = _FakeClient("ok")
    req = NormalizedRequest(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        tools=[{"name": "report", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "report"},
    )
    AnthropicProvider(client).invoke(T, req, timeout=5.0)
    kw = client.messages.last_kwargs
    assert kw["tools"] == [{"name": "report", "input_schema": {"type": "object"}}]
    assert kw["tool_choice"] == {"type": "tool", "name": "report"}
    assert kw["messages"] == [{"role": "user", "content": "hi"}]
