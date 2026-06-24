# tests/test_gateway_retry.py
import pytest
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.core.cost import CostMeter
from agent_ctl.config import RetryConfig
from agent_ctl.providers.fake import FakeProvider
from agent_ctl.models import Target, NormalizedRequest
from agent_ctl.errors import RetriableError, TerminalError

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


def test_backoff_applies_bounded_jitter(monkeypatch):
    monkeypatch.setattr("agent_ctl.core.gateway.random.uniform", lambda low, high: high)
    gw = _gw(
        FakeProvider(["ok"]),
        retry=RetryConfig(
            max_attempts_per_target=2,
            base_backoff_s=1.0,
            timeout_s=1.0,
            jitter_ratio=0.25,
        ),
    )
    assert gw._backoff_s(1) == 2.5
