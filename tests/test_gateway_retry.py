# tests/test_gateway_retry.py
import pytest
from agentctl.core.gateway import Gateway
from agentctl.core.router import Router
from agentctl.core.cost import CostMeter
from agentctl.config import RetryConfig
from agentctl.providers.fake import FakeProvider
from agentctl.models import Target, NormalizedRequest
from agentctl.errors import RetriableError, TerminalError

REQ = NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}])
T = Target(provider="fake", model="m")


def _gw(
    provider,
    retry=RetryConfig(max_attempts_per_target=2, base_backoff_s=0.0, timeout_s=1.0),
):
    return Gateway(
        router=Router({"default": ["fake/m"]}),
        providers={"fake": provider},
        cost_meter=CostMeter({}),
        store=None,
        cache=None,
        retry=retry,
    )


def test_retriable_then_success_within_target():
    gw = _gw(FakeProvider(["retriable", "ok"]))
    attempts = []
    resp = gw._invoke_target(FakeProvider(["retriable", "ok"]), T, REQ, attempts)
    assert resp.text == "fake-ok"
    assert [a.outcome for a in attempts] == ["retriable", "success"]


def test_terminal_not_retried_but_attempt_recorded():
    p = FakeProvider(["terminal", "ok"])
    gw = _gw(p)
    attempts = []
    with pytest.raises(TerminalError):
        gw._invoke_target(p, T, REQ, attempts)
    assert len(p.calls) == 1  # 终态不重试
    assert [a.outcome for a in attempts] == ["terminal"]  # 失败也留痕


def test_retriable_exhausted_records_all_attempts():
    p = FakeProvider(["retriable", "retriable"])
    gw = _gw(p)
    attempts = []
    with pytest.raises(RetriableError):
        gw._invoke_target(p, T, REQ, attempts)
    assert len(p.calls) == 2
    assert [a.outcome for a in attempts] == ["retriable", "retriable"]  # 失败也留痕
