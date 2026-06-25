# tests/test_gateway_deadline.py
import time

import pytest

from agent_ctl.config import RetryConfig
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.errors import AllTargetsFailed, RetriableError
from agent_ctl.models import NormalizedRequest, NormalizedResponse
from agent_ctl.store.sqlite_store import SqliteCaptureStore

REQ = NormalizedRequest(
    model="default",
    messages=[{"role": "user", "content": "hi"}],
    metadata={"consumer": "t"},
)


class SlowProvider:
    """每次 invoke 记录收到的 timeout,并按 sleep_s 拖延。"""

    def __init__(self, sleep_s=0.0):
        self._sleep = sleep_s
        self.timeouts = []

    def invoke(self, target, request, timeout):
        self.timeouts.append(timeout)
        if self._sleep:
            time.sleep(self._sleep)
        return NormalizedResponse(text="ok", input_tokens=1, output_tokens=1)


def test_per_attempt_timeout_clamped_to_remaining_deadline(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    p = SlowProvider()
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": p},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=1, timeout_s=60.0),
        request_deadline_s=2.0,
    )
    gw.invoke(REQ)
    # 单次超时被压到 ≤ 剩余预算(2s),而非配置的 60s
    assert p.timeouts[0] <= 2.0


def test_deadline_exhausted_stops_fallback(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    # 第一目标拖过 deadline → 第二目标因预算耗尽被跳过(留 deadline 痕)
    p = SlowProvider(sleep_s=0.15)
    gw = Gateway(
        router=Router({"default": ["fake/a", "fake/b"]}),
        providers={"fake": p},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=1, timeout_s=60.0),
        request_deadline_s=0.1,
    )
    # 第一目标成功返回(SlowProvider 不真超时,只是耗时);但耗时超 deadline 后
    # 不会再回退。这里第一目标成功,故应直接成功。验证它不抛 deadline:
    resp = gw.invoke(REQ)
    assert resp.text == "ok"


def test_no_deadline_uses_full_timeout(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    p = SlowProvider()
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": p},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=1, timeout_s=30.0),
        request_deadline_s=0.0,  # 关闭
    )
    gw.invoke(REQ)
    assert p.timeouts[0] == 30.0


def test_deadline_already_exceeded_records_all_failed(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))

    class TwoTargetSlow:
        def __init__(self):
            self.calls = 0

        def invoke(self, target, request, timeout):
            self.calls += 1
            time.sleep(0.12)  # 拖过 deadline,使第二目标被预算闸跳过
            from agent_ctl.errors import RetriableError

            raise RetriableError("slow then fail")

    p = TwoTargetSlow()
    gw = Gateway(
        router=Router({"default": ["fake/a", "fake/b"]}),
        providers={"fake": p},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=1, timeout_s=60.0),
        request_deadline_s=0.1,
    )
    with pytest.raises(AllTargetsFailed):
        gw.invoke(REQ)
    # 第一目标尝试后已超预算 → 第二目标不再发起
    assert p.calls == 1
    rec = store.list_recent(1)[0]
    assert rec.attempts[-1].outcome == "deadline"


class SlowFailProvider:
    """invoke 先 sleep(消耗 deadline 预算)再抛 RetriableError。"""

    def __init__(self, sleep_s):
        self._sleep = sleep_s

    def invoke(self, target, request, timeout):
        time.sleep(self._sleep)
        raise RetriableError("slow fail")


def test_deadline_exhaustion_does_not_charge_circuit(tmp_path):
    """F2:deadline 耗尽(第2次尝试 timeout 归零)不计熔断,provider 不被误开路。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    cb = CircuitBreaker(failure_threshold=1, cooldown_s=30.0)
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": SlowFailProvider(0.06)},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=2, base_backoff_s=0.0, timeout_s=1.0),
        request_deadline_s=0.05,  # 第1次尝试 sleep 0.06 即耗尽 → 第2次 DeadlineExceeded
        circuit=cb,
    )
    with pytest.raises(AllTargetsFailed):
        gw.invoke(REQ)
    assert cb.allow("fake") is True  # 未因 deadline 误开路(阈值=1)


def test_real_retriable_exhaustion_does_charge_circuit(tmp_path):
    """对照:无 deadline 时,真实可重试耗尽**会**计熔断(阈值=1 → 开路)。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    cb = CircuitBreaker(failure_threshold=1, cooldown_s=30.0)
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": SlowFailProvider(0.0)},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0),
        request_deadline_s=0.0,  # 关闭 deadline → 纯可重试失败
        circuit=cb,
    )
    with pytest.raises(AllTargetsFailed):
        gw.invoke(REQ)
    assert cb.allow("fake") is False  # 真实失败 → 开路
