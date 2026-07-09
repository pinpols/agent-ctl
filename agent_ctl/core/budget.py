# agent_ctl/core/budget.py
"""成本预算闸:控制面从"看得见花费"升级到"拦得住超支"的强制治理杠杆。

进程内累计每个 consumer 的已花 USD(及全局合计),调用前 `check()` 若已达上限即
抛 `BudgetExceeded`(在打 provider 之前短路,不产生真实开销);调用后把实际成本
`add()` 进累计。线程安全。

**已知边界(showcase v1)**:累计是进程内、进程生命周期窗口——重启清零、多副本各算各的。
持久化/滚动时间窗/分布式共享预算 = 后续(与捕获库 PG 化、分布式部署一并)。
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from agent_ctl.errors import BudgetExceeded


class BudgetGuard:
    def __init__(
        self,
        per_consumer: dict[str, float] | None = None,
        global_cap: float | None = None,
        max_consumers: int = 10_000,
    ) -> None:
        self._caps = dict(per_consumer or {})
        self._global_cap = global_cap
        # 有界 LRU:consumer 名来自客户端可控的 user 字段,无界 dict 会被
        # 海量伪造名撑爆内存(与 server 限流桶同款问题/同款治法)。淘汰只丢
        # 未配 cap 的 consumer 的明细(其花费仍计入 _global_spent 标量);
        # 配了 cap 的 consumer 淘汰会毁预算强制 → 永不淘汰。
        self._max_consumers = max_consumers
        self._spent: OrderedDict[str, float] = OrderedDict()
        self._global_spent = 0.0
        # 每 consumer 最近一次实际成本,作为 check 的"预留余量":在仍差约一次调用就触顶时
        # 即拒绝,把并发 in-flight 调用导致的越界窗口从"任意并发量"收紧到"约一次调用"。
        # 这是廉价的近似收紧,非精确预留引擎(精确/分布式预算见 ADR-0001 后置项)。
        self._last_cost: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._caps) or self._global_cap is not None

    def check(self, consumer: str) -> None:
        """已达上限(或预留余量后将触顶)→ 抛 BudgetExceeded(调用前短路)。"""
        if not self.enabled:
            return
        with self._lock:
            reserve = self._last_cost.get(consumer, 0.0)  # 预留一次"典型调用"余量
            cap = self._caps.get(consumer)
            spent = self._spent.get(consumer, 0.0)
            if cap is not None and spent + reserve >= cap:
                raise BudgetExceeded(
                    f"consumer {consumer!r} budget exhausted: "
                    f"spent {spent:.6f} (+reserve {reserve:.6f}) >= cap {cap:.6f} USD"
                )
            if (
                self._global_cap is not None
                and self._global_spent + reserve >= self._global_cap
            ):
                raise BudgetExceeded(
                    f"global budget exhausted: spent {self._global_spent:.6f} "
                    f"(+reserve {reserve:.6f}) >= cap {self._global_cap:.6f} USD"
                )

    def add(self, consumer: str, cost: float | None) -> None:
        """把一次调用的实际成本计入累计(cost=None/0 忽略),并更新预留余量。"""
        if not self.enabled or not cost:
            return
        with self._lock:
            self._spent[consumer] = self._spent.get(consumer, 0.0) + cost
            self._spent.move_to_end(consumer)
            self._global_spent += cost
            self._last_cost[consumer] = cost
            self._last_cost.move_to_end(consumer)
            self._evict_locked()

    def _evict_locked(self) -> None:
        while len(self._spent) > self._max_consumers:
            victim = next(
                (k for k in self._spent if k not in self._caps), None
            )  # 最久未见的无 cap consumer
            if victim is None:
                return  # 全是有 cap 的(数量由配置决定,天然有界)→ 不淘汰
            del self._spent[victim]
            self._last_cost.pop(victim, None)

    def tracked_consumers(self) -> int:
        with self._lock:
            return len(self._spent)

    def spent(self, consumer: str) -> float:
        with self._lock:
            return self._spent.get(consumer, 0.0)
