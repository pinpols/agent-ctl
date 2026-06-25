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

from agent_ctl.errors import BudgetExceeded


class BudgetGuard:
    def __init__(
        self,
        per_consumer: dict[str, float] | None = None,
        global_cap: float | None = None,
    ) -> None:
        self._caps = dict(per_consumer or {})
        self._global_cap = global_cap
        self._spent: dict[str, float] = {}
        self._global_spent = 0.0
        # 每 consumer 最近一次实际成本,作为 check 的"预留余量":在仍差约一次调用就触顶时
        # 即拒绝,把并发 in-flight 调用导致的越界窗口从"任意并发量"收紧到"约一次调用"。
        # 这是廉价的近似收紧,非精确预留引擎(精确/分布式预算见 ADR-0001 后置项)。
        self._last_cost: dict[str, float] = {}
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
            self._global_spent += cost
            self._last_cost[consumer] = cost

    def spent(self, consumer: str) -> float:
        with self._lock:
            return self._spent.get(consumer, 0.0)
