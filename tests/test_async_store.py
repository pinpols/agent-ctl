# tests/test_async_store.py
import threading
import time

from agent_ctl.models import CallRecord
from agent_ctl.store.async_store import AsyncCaptureStore, _STOP
from agent_ctl.store.sqlite_store import SqliteCaptureStore


def _rec(i):
    return CallRecord(id=f"r{i}", ts=float(i), consumer="t", status="success")


def test_async_save_then_flush_is_readable(tmp_path):
    inner = SqliteCaptureStore(str(tmp_path / "c.db"))
    store = AsyncCaptureStore(inner)
    for i in range(5):
        store.save(_rec(i))
    # list_recent 先 flush 再读 → 写后即读一致
    got = store.list_recent(10)
    assert len(got) == 5
    store.close()


def test_save_does_not_block_on_slow_inner(tmp_path):
    class SlowStore:
        def __init__(self):
            self.saved = 0

        def save(self, record):
            time.sleep(0.05)
            self.saved += 1

        def close(self):
            pass

    store = AsyncCaptureStore(SlowStore())
    start = time.monotonic()
    for i in range(10):
        store.save(_rec(i))
    elapsed = time.monotonic() - start
    assert elapsed < 0.05  # 入队即返回,不等慢速落库
    store.flush()
    assert store._inner.saved == 10
    store.close()


def test_full_queue_drops_fail_open():
    class BlockingStore:
        def __init__(self):
            self.gate = threading.Event()
            self.saved = 0

        def save(self, record):
            self.gate.wait()  # 卡住后台线程,撑满队列
            self.saved += 1

        def close(self):
            self.gate.set()

    inner = BlockingStore()
    store = AsyncCaptureStore(inner, max_queue=2)
    # 第一条被 worker 取走并卡在 save;队列容量 2 再填满后续 save 应被丢弃而非抛错/阻塞
    for i in range(20):
        store.save(_rec(i))
    assert store.dropped > 0  # 满队列丢弃,主调用不受影响
    inner.gate.set()
    store.close()


def test_close_flushes_pending(tmp_path):
    inner = SqliteCaptureStore(str(tmp_path / "c.db"))
    store = AsyncCaptureStore(inner)
    for i in range(3):
        store.save(_rec(i))
    store.close()  # 关停前应落完
    reopened = SqliteCaptureStore(str(tmp_path / "c.db"))
    assert len(reopened.list_recent(10)) == 3


def test_close_skips_inner_close_while_worker_draining():
    """F4:join 超时(worker 仍在写)时不关内层,避免与活跃写者抢连接。"""
    gate = threading.Event()

    class BlockingInner:
        def __init__(self):
            self.closed = False

        def save(self, record):
            gate.wait()  # 卡住 worker,使其 join 超时

        def close(self):
            self.closed = True

    inner = BlockingInner()
    store = AsyncCaptureStore(inner)
    store.save(_rec(0))  # worker 取走并卡在 save
    store.close(timeout=0.2)  # join 超时 → worker 仍 alive
    assert inner.closed is False  # 未在 worker 仍活时关内层
    gate.set()  # 放行收尾


def test_close_does_not_block_when_queue_full_and_worker_stuck():
    """关闭时队列满 + worker 卡住时,close timeout 仍应生效。"""
    gate = threading.Event()

    class BlockingInner:
        def __init__(self):
            self.started = threading.Event()
            self.saved = 0
            self.closed = False

        def save(self, record):
            self.started.set()
            gate.wait()
            self.saved += 1

        def close(self):
            self.closed = True

    inner = BlockingInner()
    store = AsyncCaptureStore(inner, max_queue=1)
    store.save(_rec(0))
    assert inner.started.wait(timeout=1)
    store.save(_rec(1))  # worker 被第一条卡住,队列容量 1 被第二条填满

    start = time.monotonic()
    store.close(timeout=0.01)
    assert time.monotonic() - start < 0.2
    assert inner.closed is False

    gate.set()
    deadline = time.monotonic() + 1
    while inner.saved < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    store._q.put(_STOP)
    store._thread.join(timeout=1)


def test_close_is_idempotent_and_save_after_close_drops(tmp_path):
    inner = SqliteCaptureStore(str(tmp_path / "c.db"))
    store = AsyncCaptureStore(inner)
    store.save(_rec(0))
    store.close()
    store.close()  # 二次 close 幂等(atexit 兜底也会再调一次)
    store.save(_rec(1))  # 关停后 save 丢弃,不抛、不阻塞
    assert store.dropped >= 1
    reopened = SqliteCaptureStore(str(tmp_path / "c.db"))
    assert len(reopened.list_recent(10)) == 1  # 只落了关停前那条
