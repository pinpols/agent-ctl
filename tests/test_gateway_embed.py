# tests/test_gateway_embed.py
import time

import pytest

from agent_ctl.config import RetryConfig
from agent_ctl.core.budget import BudgetGuard
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import EmbeddingResponse
from agent_ctl.providers.fake import FakeProvider
from agent_ctl.store.sqlite_store import SqliteCaptureStore

RETRY = RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0)


class FakeEmbedProvider:
    """带 embed 能力的离线 provider;按脚本逐次产出 ok/retriable/terminal。"""

    def __init__(self, script=None, dim=3):
        self._script = list(script or ["ok"])
        self._i = 0
        self._dim = dim
        self.calls = []

    def invoke(self, target, request, timeout):  # 满足 Provider 协议(本测试不用)
        raise NotImplementedError

    def embed(self, target, inputs, timeout) -> EmbeddingResponse:
        self.calls.append((target, list(inputs)))
        action = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        if action == "ok":
            return EmbeddingResponse(
                vectors=[[0.1] * self._dim for _ in inputs],
                input_tokens=len(inputs) * 2,
            )
        if action == "retriable":
            raise RetriableError("fake embed retriable")
        if action == "terminal":
            raise TerminalError("fake embed terminal")
        raise ValueError(action)


def _gw(providers, routes, store, circuit=None):
    return Gateway(
        router=Router(routes),
        providers=providers,
        cost_meter=CostMeter({"m": (5.0, 25.0)}),
        store=store,
        retry=RETRY,
        circuit=circuit,
    )


def test_embed_success_records_capture(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    p = FakeEmbedProvider(["ok"])
    gw = _gw({"emb": p}, {"default": ["emb/m"]}, store)
    resp = gw.embed("default", ["a", "b"], {"consumer": "t"})
    assert len(resp.vectors) == 2
    assert resp.input_tokens == 4
    rec = store.list_recent(1)[0]
    assert rec.status == "success"
    assert rec.model_resolved == "emb/m"
    assert rec.params["embed"] is True
    assert rec.cost_usd is not None  # model 'm' 有价表


def test_embed_skips_provider_without_capability(tmp_path):
    """无 embed 能力的 provider(如 FakeProvider≈Anthropic)被跳过,回退到能 embed 的目标。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    no_embed = FakeProvider(["ok"])  # 无 embed 方法
    emb = FakeEmbedProvider(["ok"])
    gw = _gw(
        {"plain": no_embed, "emb": emb},
        {"default": ["plain/m", "emb/m"]},
        store,
    )
    resp = gw.embed("default", ["x"], {"consumer": "t"})
    assert len(resp.vectors) == 1
    assert len(emb.calls) == 1
    rec = store.list_recent(1)[0]
    assert rec.status == "fallback_success"
    assert rec.attempts[0].outcome == "no_embed"


def test_embed_strict_pricing_skips_no_embed_before_price_check(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    no_embed = FakeProvider(["ok"])
    emb = FakeEmbedProvider(["ok"])
    gw = Gateway(
        router=Router({"default": ["plain/unpriced-chat", "emb/m"]}),
        providers={"plain": no_embed, "emb": emb},
        cost_meter=CostMeter({"m": (5.0, 25.0)}, fail_unknown=True),
        store=store,
        retry=RETRY,
    )
    resp = gw.embed("default", ["x"], {"consumer": "t"})
    assert len(resp.vectors) == 1
    rec = store.list_recent(1)[0]
    assert rec.status == "fallback_success"
    assert rec.attempts[0].outcome == "no_embed"


def test_embed_open_circuit_skips_provider(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    pa = FakeEmbedProvider(["retriable", "retriable", "retriable"])
    pb = FakeEmbedProvider(["ok", "ok", "ok"])
    gw = _gw(
        {"pa": pa, "pb": pb},
        {"default": ["pa/m", "pb/m"]},
        store,
        circuit=CircuitBreaker(failure_threshold=2, cooldown_s=30.0),
    )
    assert len(gw.embed("default", ["x"]).vectors) == 1
    assert len(gw.embed("default", ["x"]).vectors) == 1
    assert len(pa.calls) == 2  # 两次失败后开路
    gw.embed("default", ["x"])
    assert len(pa.calls) == 2  # 第三次:pa 开路被跳过


def test_embed_all_targets_fail_raises(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    pa = FakeEmbedProvider(["retriable"])
    gw = _gw({"pa": pa}, {"default": ["pa/m"]}, store)
    with pytest.raises(AllTargetsFailed):
        gw.embed("default", ["x"], {"consumer": "t"})
    rec = store.list_recent(1)[0]
    assert rec.status == "error"
    assert rec.error_type == "all_failed"


def test_embed_terminal_error_propagates(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    pa = FakeEmbedProvider(["terminal"])
    gw = _gw({"pa": pa}, {"default": ["pa/m"]}, store)
    with pytest.raises(TerminalError):
        gw.embed("default", ["x"], {"consumer": "t"})
    rec = store.list_recent(1)[0]
    assert rec.status == "error"
    assert rec.error_type == "terminal"


def test_embed_unknown_model_raises_routing(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    pa = FakeEmbedProvider(["ok"])
    gw = _gw({"pa": pa}, {"default": ["pa/m"]}, store)
    with pytest.raises(GatewayError):
        gw.embed("missing", ["x"], {"consumer": "t"})
    rec = store.list_recent(1)[0]
    assert rec.error_type == "routing"


# ── 覆盖补强:embed 内预算/未注册直连/deadline ──────────────────────────────


def test_embed_budget_exceeded_before_call(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["emb/m"]}),
        providers={"emb": FakeEmbedProvider(["ok"])},
        cost_meter=CostMeter({}),
        store=store,
        retry=RETRY,
        budget=BudgetGuard(per_consumer={"t": 0.0}),  # cap 0 → 首次即超
    )
    with pytest.raises(BudgetExceeded):
        gw.embed("default", ["x"], {"consumer": "t"})
    rec = store.list_recent(1)[0]
    assert rec.status == "error" and rec.error_type == "budget"


def test_embed_direct_unregistered_provider(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = _gw({"emb": FakeEmbedProvider(["ok"])}, {"default": ["emb/m"]}, store)
    with pytest.raises(GatewayError):
        gw.embed("nope/x", ["x"], {"consumer": "t"})
    rec = store.list_recent(1)[0]
    assert rec.error_type == "provider"


def test_embed_deadline_break_stops_fallback(tmp_path):
    """第一目标耗时超 deadline → 第二目标在循环顶被 deadline 守卫跳过。"""

    class SlowEmbedFail:
        def embed(self, target, inputs, timeout):
            time.sleep(0.06)
            raise RetriableError("slow")

    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["a/m", "b/m"]}),
        providers={"a": SlowEmbedFail(), "b": FakeEmbedProvider(["ok"])},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=1, timeout_s=60.0),
        request_deadline_s=0.05,
    )
    with pytest.raises(AllTargetsFailed):
        gw.embed("default", ["x"], {"consumer": "t"})
    rec = store.list_recent(1)[0]
    assert rec.attempts[-1].outcome == "deadline"
