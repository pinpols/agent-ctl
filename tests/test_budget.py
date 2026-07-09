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


def test_reserve_blocks_one_call_before_cap():
    """F5:预留一次"典型调用"余量 → 仍差约一次就触顶时即拒绝,收紧并发越界窗口。"""
    g = BudgetGuard(per_consumer={"c": 0.01})
    g.add("c", 0.004)  # spent=0.004,last_cost=0.004
    # spent(0.004) < cap(0.01),但 spent+reserve(0.004) = 0.008 < 0.01 → 仍可过
    g.check("c")
    g.add("c", 0.004)  # spent=0.008
    # spent(0.008) 仍 < cap,但 +reserve(0.004)=0.012 >= 0.01 → 提前拒绝(防越界)
    with pytest.raises(BudgetExceeded):
        g.check("c")


def test_first_call_has_no_reserve():
    """无历史成本时预留为 0,首call不被误拒。"""
    g = BudgetGuard(per_consumer={"c": 0.01})
    g.check("c")  # spent 0 + reserve 0 → 通过


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


# ── 深审 round4 P2-11:按 consumer 的累计表有界(consumer 名客户端可控)──


def test_spent_tracking_bounded_by_lru():
    g = BudgetGuard(global_cap=1000.0, max_consumers=3)
    for i in range(10):
        g.add(f"c{i}", 0.01)
    # 无界会留 10 个;有界(LRU)只留最近 3 个
    assert g.tracked_consumers() == 3
    assert g.spent("c9") > 0  # 最近的仍在
    assert g.spent("c0") == 0.0  # 最老的被淘汰(仅丢明细,全局累计不受影响)
    assert abs(g._global_spent - 0.1) < 1e-9


def test_capped_consumers_never_evicted():
    g = BudgetGuard(per_consumer={"vip": 5.0}, global_cap=1000.0, max_consumers=2)
    g.add("vip", 1.0)
    for i in range(10):
        g.add(f"noise{i}", 0.01)
    assert g.spent("vip") == 1.0  # 有 cap 的 consumer 淘汰会毁预算强制,必须钉住
