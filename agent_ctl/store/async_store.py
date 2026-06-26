# agent_ctl/store/async_store.py
"""把任意 CaptureStore 的写入移出请求主路径:后台单线程消费有界队列。

动机:捕获(脱敏 + SQLite 写,且 SQLite 单写锁串行)原本在 `Gateway.invoke` 返回前
同步执行,给每次调用叠加 I/O 延迟。本装饰器让 `save()` 仅入队即返回(非阻塞),
真正落库在后台线程完成——主调用延迟与存储解耦。

容错(fail-open,承网关一贯原则):队列满 → 丢弃 + 计数告警(绝不阻塞真实调用);
后台落库异常 → 告警吞掉。读路径(list_recent / cost_summary)先 flush 再透传内层,
保证"写后即读"一致。
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading

from agent_ctl.models import CallRecord

log = logging.getLogger("agent_ctl.store.async")

_STOP = object()  # 关停哨兵


class AsyncCaptureStore:
    def __init__(self, inner, max_queue: int = 10_000) -> None:
        self._inner = inner
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="agent-ctl-capture", daemon=True
        )
        self._thread.start()
        # 库形态一次性消费者(如 ops-agent 调一次就退出)不会显式 close →
        # 注册 atexit 兜底落完 pending,避免丢尾记录 / 泄漏后台线程。close 幂等。
        atexit.register(self.close)

    @property
    def dropped(self) -> int:
        return self._dropped

    def save(self, record: CallRecord) -> None:
        if self._closed:  # 已关停 → 后台不再消费,丢弃避免无人取的积压
            self._dropped += 1
            return
        try:
            self._q.put_nowait(record)
        except queue.Full:
            self._dropped += 1
            log.warning(
                "capture queue full (fail-open): dropped record (total dropped=%d)",
                self._dropped,
            )

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP:
                self._q.task_done()
                break
            try:
                self._inner.save(item)
            except Exception as exc:  # fail-open:后台落库失败绝不影响主调用
                log.warning("async capture save failed (fail-open): %s", exc)
            finally:
                self._q.task_done()

    def flush(self) -> None:
        """阻塞至队列排空(测试与"写后即读"用)。"""
        self._q.join()

    def list_recent(self, *args, **kwargs):
        self.flush()
        return self._inner.list_recent(*args, **kwargs)

    def iter_all(self, *args, **kwargs):
        self.flush()
        return self._inner.iter_all(*args, **kwargs)

    def cost_summary(self, *args, **kwargs):
        self.flush()
        return self._inner.cost_summary(*args, **kwargs)

    def close(self, timeout: float = 5.0) -> None:
        """幂等关停:落完队列、停后台线程、关内层。atexit 与显式调用都安全。

        仅在 worker 真正退出后才关内层——否则积压致 join 超时时,关闭内层会与仍在写的
        worker 抢同一连接(SQLite 'closed' 错误)。worker 是 daemon,极端积压下让它
        随进程退出由 OS 回收,不强关。
        """
        if self._closed:
            return
        self._closed = True
        self._q.put(_STOP)
        self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            self._inner.close()
        else:
            log.warning(
                "async capture worker still draining at close; "
                "skip inner.close to avoid racing an active writer"
            )

    def __enter__(self) -> "AsyncCaptureStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
