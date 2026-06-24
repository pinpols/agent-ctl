import pytest
from agentctl.models import Target, NormalizedRequest
from agentctl.providers.fake import FakeProvider
from agentctl.errors import RetriableError, TerminalError

REQ = NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}])
T = Target(provider="fake", model="m")


def test_fake_ok():
    p = FakeProvider(["ok"])
    resp = p.invoke(T, REQ, timeout=1.0)
    assert resp.text == "fake-ok"
    assert resp.input_tokens == 10


def test_fake_retriable_then_ok():
    p = FakeProvider(["retriable", "ok"])
    with pytest.raises(RetriableError):
        p.invoke(T, REQ, timeout=1.0)
    assert p.invoke(T, REQ, timeout=1.0).text == "fake-ok"


def test_fake_terminal():
    p = FakeProvider(["terminal"])
    with pytest.raises(TerminalError):
        p.invoke(T, REQ, timeout=1.0)
