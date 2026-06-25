# tests/test_budget.py
import pytest

from agent_ctl.config import RetryConfig
from agent_ctl.core.budget import BudgetGuard
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.errors import BudgetExceeded
from agent_ctl.models import NormalizedRequest
from agent_ctl.providers.fake import FakeProvider
from agent_ctl.store.sqlite_store import SqliteCaptureStore

RETRY = RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0)


# ── BudgetGuard 单元 ────────────────────────────────────────


def test_disabled_guard_never_blocks():
    g = BudgetGuard()
    assert g.enabled is False
    g.add("c", 999.0)
    g.check("c")  # 不抛


def test_per_consumer_cap_blocks_after_exhausted():
    g = BudgetGuard(per_consumer={"c": 0.01})
    g.check("c")  # 起始可过
    g.add("c", 0.02)  # 超额
    with pytest.raises(BudgetExceeded):
        g.check("c")
    g.check("other")  # 其他 consumer 不受影响


def test_global_cap_blocks_across_consumers():
    g = BudgetGuard(global_cap=0.05)
    g.add("a", 0.03)
    g.add("b", 0.03)  # 全局累计 0.06 > 0.05
    with pytest.raises(BudgetExceeded):
        g.check("a")
    with pytest.raises(BudgetExceeded):
        g.check("b")


def test_add_ignores_none_and_zero():
    g = BudgetGuard(per_consumer={"c": 0.01})
    g.add("c", None)
    g.add("c", 0.0)
    g.check("c")  # 仍可过
    assert g.spent("c") == 0.0


# ── 网关级:预算闸短路 ──────────────────────────────────────


def _gw(store, budget):
    return Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": FakeProvider(["ok", "ok", "ok"])},
        cost_meter=CostMeter({"a": (1000.0, 1000.0)}),  # 故意高价,一次就累计可观成本
        store=store,
        retry=RETRY,
        budget=budget,
    )


def _req():
    return NormalizedRequest(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"consumer": "rag"},
    )


def test_gateway_blocks_when_consumer_over_budget(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    # FakeProvider ok 用 10 in / 5 out token,价 1000/1M → 每次约 0.000015 USD
    guard = BudgetGuard(per_consumer={"rag": 0.00001})
    gw = _gw(store, guard)
    gw.invoke(_req())  # 第一次通过并累计成本(超过 0.00001 上限)
    with pytest.raises(BudgetExceeded):
        gw.invoke(_req())  # 第二次被短路
    p = gw._providers["fake"]
    assert len(p.calls) == 1  # 短路在打 provider 之前 → 只调过一次
    rec = store.list_recent(1)[0]
    assert rec.status == "error"
    assert rec.error_type == "budget"


def test_gateway_unlimited_budget_passes(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = _gw(store, BudgetGuard())
    for _ in range(3):
        gw.invoke(_req())  # 不限 → 全通过
    assert len(gw._providers["fake"].calls) == 3
