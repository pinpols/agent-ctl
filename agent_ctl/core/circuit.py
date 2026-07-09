# agent_ctl/core/circuit.py
"""按 provider 的熔断器。

连续失败达阈值 → 该 provider "开路" 冷却一段时间,期间网关在回退链里**跳过**它,
直接试下一目标,不再硬打 N 次重试(省时省钱,且让回退更快生效)。冷却结束 → 半开:
**显式建模**——只授一个探测名额(并发调用不会同时涌入),探测成功闭合、探测失败
立即回开路(不需要重新累计满阈值);探测方若从未回报(如被 deadline 跳过),名额
过一个冷却期后自动过期重授,避免永久卡死。

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
        # 半开探测名额:provider → 授出时刻。存在即"探测在途",其余调用仍被拒;
        # 超过一个 cooldown 未回报视为名额过期,可重授。
        self._probe_at: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._threshold > 0 and self._cooldown > 0

    def allow(self, provider: str) -> bool:
        """该 provider 现在可试吗?开路且冷却未到 → False;冷却到 → 半开,
        只放行单个探测(其余并发调用继续被拒,直到探测回报或名额过期)。"""
        if not self.enabled:
            return True
        with self._lock:
            opened = self._opened_at.get(provider)
            if opened is None:
                return True
            now = self._now()
            if now - opened < self._cooldown:
                return False
            probe_at = self._probe_at.get(provider)
            if probe_at is not None and now - probe_at < self._cooldown:
                return False  # 探测在途 → 其余调用仍被拒
            self._probe_at[provider] = now  # 授出(或重授过期的)探测名额
            return True

    def record_success(self, provider: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._fails[provider] = 0
            self._opened_at.pop(provider, None)
            self._probe_at.pop(provider, None)

    def record_failure(self, provider: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if provider in self._probe_at:
                # 半开探测失败 → 立即回开路(重新冷却),不重新累计阈值
                self._probe_at.pop(provider, None)
                self._opened_at[provider] = self._now()
                self._fails[provider] = self._threshold
                return
            n = self._fails.get(provider, 0) + 1
            self._fails[provider] = n
            if n >= self._threshold:
                self._opened_at[provider] = self._now()
