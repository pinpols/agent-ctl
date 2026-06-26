# agent_ctl/core/_host.py
"""Runner 协作者依赖的网关内部面(类型契约)+ 单次调用上下文。

StreamRunner / EmbeddingRunner 不再以 mixin 继承 Gateway 私有成员(那是"伪解耦"+ 需全局
关 mypy attr-defined),而是接收一个满足 RunnerHost 的 host(运行时即 Gateway 本身)。
好处:类型可检查(去掉 mypy 禁用)、可注入假 host 独立测试 runner。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agent_ctl.core.budget import BudgetGuard
from agent_ctl.core.capture import Capturer
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.router import Router
from agent_ctl.models import Attempt, NormalizedRequest, NormalizedResponse, Target
from agent_ctl.providers.base import Provider


@dataclass
class CallCtx:
    """一次调用的治理上下文(消除 request/meta/started/deadline/attempts 这组数据团,
    把流式辅助方法从 8–9 个位置参数压到 ctx + 少数变量)。"""

    request: NormalizedRequest
    meta: dict
    started: float
    deadline: float | None
    attempts: list[Attempt] = field(default_factory=list)


class RunnerHost(Protocol):
    """Runner 协作者所需的网关内部面。Gateway 结构化满足之(无需显式继承)。"""

    _router: Router
    _providers: dict[str, Provider]
    _circuit: CircuitBreaker
    _budget: BudgetGuard
    _capturer: Capturer

    def _invoke_target(
        self,
        provider: Provider,
        target: Target,
        request: NormalizedRequest,
        attempts: list[Attempt],
        deadline: float | None = ...,
    ) -> NormalizedResponse: ...

    def _deadline_for(self, started: float) -> float | None: ...

    def _timeout_within(self, deadline: float | None) -> float | None: ...

    def _attempt(
        self, target: Target, outcome: str, t0: float, error: str | None
    ) -> Attempt: ...

    def _deadline_exceeded(
        self, target: Target, deadline: float | None, attempts: list[Attempt]
    ) -> bool: ...

    def _circuit_blocked(self, target: Target, attempts: list[Attempt]) -> bool: ...
