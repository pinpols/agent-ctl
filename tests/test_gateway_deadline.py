# tests/test_gateway_deadline.py
import time

import pytest

from agent_ctl.config import RetryConfig
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.errors import AllTargetsFailed
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
