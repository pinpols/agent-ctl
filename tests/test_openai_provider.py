from agent_ctl.providers.openai_provider import OpenAIProvider, classify_status
from agent_ctl.models import Target, NormalizedRequest
from agent_ctl.errors import RetriableError, TerminalError
import pytest

REQ = NormalizedRequest(
    model="default",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=64,
    system="你是助手",
)
T = Target(provider="openai", model="gpt-4o")


class _FakeCompletions:
    def __init__(self, behavior):
        self._b = behavior
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if isinstance(self._b, Exception):
            raise self._b
        msg = type("M", (), {"content": "hello", "tool_calls": None})()
        choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
        usage = type("U", (), {"prompt_tokens": 7, "completion_tokens": 3})()
        return type("R", (), {"choices": [choice], "usage": usage})()


class _FakeClient:
    def __init__(self, behavior):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(behavior)})()


def test_invoke_maps_response():
    p = OpenAIProvider(_FakeClient("ok"))
    resp = p.invoke(T, REQ, timeout=5.0)
    assert resp.text == "hello"
    assert resp.input_tokens == 7
    assert resp.output_tokens == 3
    assert resp.finish_reason == "stop"


def test_system_prepended_as_message_and_timeout_passed():
    client = _FakeClient("ok")
    OpenAIProvider(client).invoke(T, REQ, timeout=12.5)
    sent = client.chat.completions.last_kwargs
    assert sent["messages"][0] == {"role": "system", "content": "你是助手"}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    assert sent["timeout"] == 12.5


def test_classify_status():
    assert classify_status(429) == "retriable"
    assert classify_status(503) == "retriable"
    assert classify_status(401) == "terminal"
    assert classify_status(400) == "terminal"


def test_status_based_exception_routing():
    err401 = type("E", (Exception,), {"status_code": 401})("auth")
    with pytest.raises(TerminalError):
        OpenAIProvider(_FakeClient(err401)).invoke(T, REQ, timeout=5.0)
    err503 = type("E", (Exception,), {"status_code": 503})("overloaded")
    with pytest.raises(RetriableError):
        OpenAIProvider(_FakeClient(err503)).invoke(T, REQ, timeout=5.0)
