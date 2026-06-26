# tests/test_concurrency.py
"""本地并发 soak-lite:进程内 + fake provider 持续并发,验线程安全 + 无死锁 + 无捕获丢失。

不联网(真 API 持续压测需 infra,见 ADR/上线评估)。这里只锁住:并发下 circuit/budget/
cache 的锁正确、异步捕获队列在负载下全部落库不丢、整体不死锁。
"""

import concurrent.futures
import time

from agent_ctl.config import RetryConfig
from agent_ctl.core.budget import BudgetGuard
from agent_ctl.core.cache import MemoryCache
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.models import NormalizedRequest, NormalizedResponse
from agent_ctl.store.async_store import AsyncCaptureStore
from agent_ctl.store.sqlite_store import SqliteCaptureStore

RETRY = RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=2.0)


class _SleepOk:
    def invoke(self, target, request, timeout):
        time.sleep(0.003)  # 模拟真实 I/O 时延,制造真并发交错
        return NormalizedResponse(text="ok", input_tokens=2, output_tokens=1)


def test_sustained_concurrency_thread_safe_no_capture_loss(tmp_path):
    inner = SqliteCaptureStore(str(tmp_path / "c.db"))
    store = AsyncCaptureStore(inner)
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": _SleepOk()},
        cost_meter=CostMeter({"a": (1.0, 1.0)}),
        store=store,
        retry=RETRY,
        circuit=CircuitBreaker(failure_threshold=5, cooldown_s=30.0),
        budget=BudgetGuard(global_cap=1e9),  # 不触顶,只压并发
        cache=MemoryCache(),
    )
    n = 120

    def call(i):
        return gw.invoke(
            NormalizedRequest(
                model="default",
                messages=[{"role": "user", "content": f"q{i}"}],  # 各异 → 不命中缓存
                metadata={"consumer": "load"},
            )
        )

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(call, range(n)))
    elapsed = time.monotonic() - started

    # 真实信号:全成功、无异常、无死锁(ThreadPoolExecutor.map 完成即证不死锁)。
    # 不对 elapsed 做紧贴时序的断言(零余量会在 CI 负载下 flaky);只留宽松的死锁上界。
    assert len(results) == n
    assert all(r.text == "ok" for r in results)
    assert elapsed < 10.0  # 仅防真死锁/全串行;不是性能断言
    # 异步捕获在负载下全部落库,一条不丢
    store.flush()
    assert len(inner.list_recent(n + 50)) == n
    store.close()


def test_concurrent_cache_hits_are_consistent(tmp_path):
    """同一请求高并发:缓存并发 get/set 不崩、结果一致(锁正确)。"""
    inner = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["fake/a"]}),
        providers={"fake": _SleepOk()},
        cost_meter=CostMeter({}),
        store=inner,
        retry=RETRY,
        cache=MemoryCache(),
    )
    req = NormalizedRequest(
        model="default",
        messages=[{"role": "user", "content": "same"}],
        metadata={"consumer": "load"},
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda _: gw.invoke(req), range(60)))
    assert all(r.text == "ok" for r in results)  # 并发缓存无崩、结果一致
