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
