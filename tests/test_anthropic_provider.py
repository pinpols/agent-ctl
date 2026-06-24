from agentctl.providers.anthropic_provider import AnthropicProvider, classify_status
from agentctl.models import Target, NormalizedRequest

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


def test_classify_status():
    assert classify_status(429) == "retriable"
    assert classify_status(529) == "retriable"
    assert classify_status(500) == "retriable"
    assert classify_status(401) == "terminal"
    assert classify_status(400) == "terminal"
