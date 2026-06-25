# agent_ctl/core/circuit.py
"""按 provider 的熔断器。

连续失败达阈值 → 该 provider "开路" 冷却一段时间,期间网关在回退链里**跳过**它,
直接试下一目标,不再硬打 N 次重试(省时省钱,且让回退更快生效)。冷却结束 → 半开,
放行一次试探;成功则闭合,失败则重新冷却。

`cooldown_s<=0` 或 `failure_threshold<=0` 视为关闭(allow 恒 True)。线程安全。
"""

from __future__ import annotations

import threading
import time


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        now=time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._now = now
        self._lock = threading.Lock()
        self._fails: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._threshold > 0 and self._cooldown > 0

    def allow(self, provider: str) -> bool:
        """该 provider 现在可试吗?开路且冷却未到 → False(跳过)。"""
        if not self.enabled:
            return True
        with self._lock:
            opened = self._opened_at.get(provider)
            if opened is None:
                return True
            if self._now() - opened >= self._cooldown:
                # 冷却到 → 半开:清开路态,放行一次试探
                self._opened_at.pop(provider, None)
                self._fails[provider] = 0
                return True
            return False

    def record_success(self, provider: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._fails[provider] = 0
            self._opened_at.pop(provider, None)

    def record_failure(self, provider: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            n = self._fails.get(provider, 0) + 1
            self._fails[provider] = n
            if n >= self._threshold:
                self._opened_at[provider] = self._now()
