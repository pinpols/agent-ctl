# tests/test_gateway_invoke.py
import pytest
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.cache import MemoryCache
from agent_ctl.config import RetryConfig
from agent_ctl.providers.fake import FakeProvider
from agent_ctl.store.sqlite_store import SqliteCaptureStore
from agent_ctl.models import NormalizedRequest
from agent_ctl.errors import AllTargetsFailed, GatewayError

REQ = NormalizedRequest(
    model="default",
    messages=[{"role": "user", "content": "hi"}],
    metadata={"consumer": "t"},
)
RETRY = RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0)


def _gw(provider, store, cache=None):
    return Gateway(
        router=Router({"default": ["fake/a", "fake/b"]}),
        providers={"fake": provider},
        cost_meter=CostMeter({"a": (5.0, 25.0)}),
        store=store,
        cache=cache,
        retry=RETRY,
    )


def test_primary_success(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    resp = _gw(FakeProvider(["ok"]), store).invoke(REQ)
    assert resp.text == "fake-ok"
    rec = store.list_recent(1)[0]
    assert rec.status == "success"
    assert rec.model_resolved == "fake/a"
    assert rec.cost_usd is not None  # model 'a' 有价表


def test_fallback_to_second_target(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    # 第一目标 retriable(耗尽 1 次)→ 回退第二目标 ok
    resp = _gw(FakeProvider(["retriable", "ok"]), store).invoke(REQ)
    assert resp.text == "fake-ok"
    rec = store.list_recent(1)[0]
    assert rec.status == "fallback_success"
    assert len(rec.attempts) == 2


def test_all_targets_fail_records_error_with_all_attempts(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    with pytest.raises(AllTargetsFailed):
        _gw(FakeProvider(["retriable", "retriable"]), store).invoke(REQ)
    rec = store.list_recent(1)[0]
    assert rec.status == "error"
    # 修复验证:两个目标各 1 次尝试都留痕(此前会丢)
    assert len(rec.attempts) == 2
    assert all(a.outcome == "retriable" for a in rec.attempts)


def test_cache_hit_skips_provider_and_costs_zero(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    cache = MemoryCache()
    p = FakeProvider(["ok", "ok"])
    gw = _gw(p, store, cache)
    gw.invoke(REQ)
    gw.invoke(REQ)
    assert len(p.calls) == 1  # 第二次命中缓存
    hit = store.list_recent(1)[0]
    assert hit.cache_hit is True
    assert hit.cost_usd == 0.0  # 命中=省下的开销


def test_tool_requests_are_not_cached_by_default(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    cache = MemoryCache()
    p = FakeProvider(["ok", "ok"])
    gw = _gw(p, store, cache)
    req = REQ.model_copy(update={"tools": [{"name": "x"}]})
    gw.invoke(req)
    gw.invoke(req)
    assert len(p.calls) == 2


def test_tool_requests_can_be_cached_when_enabled(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    cache = MemoryCache()
    p = FakeProvider(["ok", "ok"])
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": p},
        cost_meter=CostMeter({"a": (5.0, 25.0)}),
        store=store,
        cache=cache,
        retry=RETRY,
        cache_tool_responses=True,
    )
    req = REQ.model_copy(update={"tools": [{"name": "x"}]})
    gw.invoke(req)
    gw.invoke(req)
    assert len(p.calls) == 1


def test_unregistered_provider_raises_gateway_error():
    """路由指向未注册 provider 时,Gateway.__init__ 应抛 GatewayError 而非 KeyError。"""
    with pytest.raises(GatewayError, match="unregistered provider"):
        Gateway(
            router=Router({"default": ["missing_provider/model-x"]}),
            providers={"fake": FakeProvider(["ok"])},
            cost_meter=CostMeter({}),
            retry=RETRY,
        )


def test_unknown_model_records_routing_error(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": FakeProvider(["ok"])},
        cost_meter=CostMeter({}),
        store=store,
        retry=RETRY,
    )
    with pytest.raises(GatewayError, match="unknown model"):
        gw.invoke(
            NormalizedRequest(
                model="missing",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"consumer": "t"},
            )
        )
    rec = store.list_recent(1)[0]
    assert rec.status == "error"
    assert rec.error_type == "routing"
    assert rec.error_message_redacted is not None
    assert rec.model_requested == "missing"


def test_direct_unregistered_provider_records_error(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": FakeProvider(["ok"])},
        cost_meter=CostMeter({}),
        store=store,
        retry=RETRY,
    )
    with pytest.raises(GatewayError, match="unregistered provider"):
        gw.invoke(
            NormalizedRequest(
                model="missing_provider/model-x",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"consumer": "t"},
            )
        )
    rec = store.list_recent(1)[0]
    assert rec.status == "error"
    assert rec.error_type == "provider"
    assert rec.model_resolved == "missing_provider/model-x"


def test_cost_failure_still_saves_capture_record(tmp_path):
    class BadCost:
        def cost(self, model, input_tokens, output_tokens):
            raise RuntimeError("price table broken")

    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": FakeProvider(["ok"])},
        cost_meter=BadCost(),
        store=store,
        retry=RETRY,
    )
    resp = gw.invoke(REQ)
    assert resp.text == "fake-ok"
    rec = store.list_recent(1)[0]
    assert rec.status == "success"
    assert rec.cost_usd is None


def test_store_failure_is_fail_open(tmp_path):
    class BadStore:
        def save(self, record):
            raise RuntimeError("disk full")

    resp = _gw(FakeProvider(["ok"]), BadStore()).invoke(REQ)
    assert resp.text == "fake-ok"  # 捕获写失败不影响主调用
