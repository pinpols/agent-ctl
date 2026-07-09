# tests/test_circuit.py
from agent_ctl.config import RetryConfig
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.models import NormalizedRequest
from agent_ctl.providers.fake import FakeProvider
from agent_ctl.store.sqlite_store import SqliteCaptureStore


class _Clock:
    """可控时钟,供熔断冷却测试推进时间。"""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_disabled_when_threshold_zero():
    cb = CircuitBreaker(failure_threshold=0, cooldown_s=30.0)
    assert cb.enabled is False
    for _ in range(10):
        cb.record_failure("p")
    assert cb.allow("p") is True  # 关闭 → 恒放行


def test_disabled_when_cooldown_zero():
    cb = CircuitBreaker(failure_threshold=3, cooldown_s=0.0)
    assert cb.enabled is False
    for _ in range(5):
        cb.record_failure("p")
    assert cb.allow("p") is True


def test_opens_after_threshold_failures():
    cb = CircuitBreaker(failure_threshold=3, cooldown_s=30.0, now=_Clock())
    assert cb.allow("p") is True
    cb.record_failure("p")
    cb.record_failure("p")
    assert cb.allow("p") is True  # 未达阈值仍放行
    cb.record_failure("p")  # 第 3 次 → 开路
    assert cb.allow("p") is False


def test_failures_are_per_provider():
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30.0, now=_Clock())
    cb.record_failure("a")
    cb.record_failure("a")
    assert cb.allow("a") is False
    assert cb.allow("b") is True  # b 不受 a 影响


def test_cooldown_half_opens_then_allows_one_probe():
    clock = _Clock()
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30.0, now=clock)
    cb.record_failure("p")
    cb.record_failure("p")
    assert cb.allow("p") is False
    clock.t = 29.9
    assert cb.allow("p") is False  # 冷却未到
    clock.t = 30.0
    assert cb.allow("p") is True  # 冷却到 → 半开放行


def test_success_resets_failures():
    cb = CircuitBreaker(failure_threshold=3, cooldown_s=30.0, now=_Clock())
    cb.record_failure("p")
    cb.record_failure("p")
    cb.record_success("p")  # 清零
    cb.record_failure("p")
    cb.record_failure("p")
    assert cb.allow("p") is True  # 只累计 2 次,未开路


def test_failure_after_half_open_reopens():
    clock = _Clock()
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30.0, now=clock)
    cb.record_failure("p")
    cb.record_failure("p")
    clock.t = 30.0
    assert cb.allow("p") is True  # 半开:试探放行并清零
    cb.record_failure("p")
    cb.record_failure("p")  # 再次达阈值 → 重新开路
    assert cb.allow("p") is False


# ---- 网关级:开路 provider 被跳过,回退到下一目标 ----

REQ = NormalizedRequest(
    model="default",
    messages=[{"role": "user", "content": "hi"}],
    metadata={"consumer": "t"},
)
RETRY = RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0)


def test_open_circuit_skips_provider_and_uses_fallback(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    pa = FakeProvider(["retriable", "retriable", "retriable"])
    pb = FakeProvider(["ok", "ok", "ok"])
    gw = Gateway(
        router=Router({"default": ["pa/m", "pb/m"]}),
        providers={"pa": pa, "pb": pb},
        cost_meter=CostMeter({}),
        store=store,
        retry=RETRY,
        circuit=CircuitBreaker(failure_threshold=2, cooldown_s=30.0),
    )
    # 前两次:pa 失败回退 pb,第 2 次后 pa 熔断开路
    assert gw.invoke(REQ).text == "fake-ok"
    assert gw.invoke(REQ).text == "fake-ok"
    assert pa.calls and len(pa.calls) == 2
    # 第三次:pa 开路被跳过,不再被打,直接走 pb
    assert gw.invoke(REQ).text == "fake-ok"
    assert len(pa.calls) == 2  # 未新增 → 确认被跳过
    rec = store.list_recent(1)[0]
    assert rec.status == "fallback_success"
    assert rec.attempts[0].outcome == "circuit_open"


def test_open_circuit_records_success_resets(tmp_path):
    """pa 恢复后 record_success 应让其重新可用(此处验证未开路时正常计数清零)。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    pa = FakeProvider(["retriable", "ok", "ok"])
    pb = FakeProvider(["ok", "ok", "ok"])
    gw = Gateway(
        router=Router({"default": ["pa/m", "pb/m"]}),
        providers={"pa": pa, "pb": pb},
        cost_meter=CostMeter({}),
        store=store,
        retry=RETRY,
        circuit=CircuitBreaker(failure_threshold=3, cooldown_s=30.0),
    )
    gw.invoke(REQ)  # pa retriable → 回退 pb;pa fails=1
    gw.invoke(REQ)  # pa ok → record_success 清零
    gw.invoke(REQ)  # pa ok
    # pa 三次都被尝试(从未开路),且第 2/3 次成功
    assert len(pa.calls) == 3


# ── 深审 round4 P2-6:半开态显式建模 ──────────────────────────


def test_half_open_grants_single_probe():
    """冷却到期只放行一个探测,并发调用不得同时涌入。"""
    clock = _Clock()
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30.0, now=clock)
    cb.record_failure("p")
    cb.record_failure("p")
    clock.t = 30.0
    assert cb.allow("p") is True  # 探测名额
    assert cb.allow("p") is False  # 探测在途,其余仍拒
    assert cb.allow("p") is False


def test_half_open_probe_failure_reopens_immediately():
    """探测失败 → 立即回开路,不需要重新累计满 threshold(与 docstring 一致)。"""
    clock = _Clock()
    cb = CircuitBreaker(failure_threshold=3, cooldown_s=30.0, now=clock)
    for _ in range(3):
        cb.record_failure("p")
    clock.t = 30.0
    assert cb.allow("p") is True  # 半开探测
    cb.record_failure("p")  # 探测失败(仅 1 次)
    assert cb.allow("p") is False  # 立即回开路
    clock.t = 59.9
    assert cb.allow("p") is False  # 新一轮冷却
    clock.t = 60.0
    assert cb.allow("p") is True  # 冷却再到 → 再放一个探测


def test_half_open_probe_success_closes():
    clock = _Clock()
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30.0, now=clock)
    cb.record_failure("p")
    cb.record_failure("p")
    clock.t = 30.0
    assert cb.allow("p") is True
    cb.record_success("p")
    assert cb.allow("p") is True  # 闭合,全放行
    assert cb.allow("p") is True


def test_half_open_probe_slot_expires_if_never_reported():
    """探测方既没 record_success 也没 record_failure(如被 deadline 跳过)→
    名额过一个冷却期后重新可授,避免永久卡死。"""
    clock = _Clock()
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30.0, now=clock)
    cb.record_failure("p")
    cb.record_failure("p")
    clock.t = 30.0
    assert cb.allow("p") is True  # 探测名额被拿走且未回报
    clock.t = 45.0
    assert cb.allow("p") is False
    clock.t = 60.0
    assert cb.allow("p") is True  # 名额过期重授
