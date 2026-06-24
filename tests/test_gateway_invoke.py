# tests/test_gateway_invoke.py
import pytest
from agentctl.core.gateway import Gateway
from agentctl.core.router import Router
from agentctl.core.cost import CostMeter
from agentctl.core.cache import MemoryCache
from agentctl.config import RetryConfig
from agentctl.providers.fake import FakeProvider
from agentctl.store.sqlite_store import SqliteCaptureStore
from agentctl.models import NormalizedRequest
from agentctl.errors import AllTargetsFailed

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


def test_store_failure_is_fail_open(tmp_path):
    class BadStore:
        def save(self, record):
            raise RuntimeError("disk full")

    resp = _gw(FakeProvider(["ok"]), BadStore()).invoke(REQ)
    assert resp.text == "fake-ok"  # 捕获写失败不影响主调用
