# agentctl/core/gateway.py
from __future__ import annotations

import time

from agentctl.config import RetryConfig
from agentctl.core.cost import CostMeter
from agentctl.core.router import Router
from agentctl.errors import RetriableError, TerminalError
from agentctl.models import Attempt, NormalizedRequest, NormalizedResponse, Target
from agentctl.providers.base import Provider


class Gateway:
    def __init__(
        self,
        router: Router,
        providers: dict[str, Provider],
        cost_meter: CostMeter,
        store=None,
        cache=None,
        retry: RetryConfig | None = None,
        cache_enabled: bool = True,
        cache_ttl_s: int = 600,
    ) -> None:
        self._router = router
        self._providers = providers
        self._cost = cost_meter
        self._store = store
        self._cache = cache
        self._retry = retry or RetryConfig()
        self._cache_enabled = cache_enabled
        self._cache_ttl_s = cache_ttl_s

    def _invoke_target(
        self,
        provider: Provider,
        target: Target,
        request: NormalizedRequest,
        attempts: list[Attempt],
    ) -> NormalizedResponse:
        """对单目标尝试(含重试)。每次尝试都 append 到调用方的 attempts(成功/失败均留痕)。"""
        last_exc: Exception = RetriableError("no attempt made")
        for n in range(self._retry.max_attempts_per_target):
            started = time.monotonic()
            try:
                resp = provider.invoke(target, request, self._retry.timeout_s)
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="success",
                        latency_ms=int((time.monotonic() - started) * 1000),
                    )
                )
                return resp
            except TerminalError as exc:
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="terminal",
                        latency_ms=int((time.monotonic() - started) * 1000),
                        error=str(exc),
                    )
                )
                raise
            except (RetriableError, TimeoutError) as exc:
                outcome = "timeout" if isinstance(exc, TimeoutError) else "retriable"
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome=outcome,
                        latency_ms=int((time.monotonic() - started) * 1000),
                        error=str(exc),
                    )
                )
                last_exc = exc
                if n < self._retry.max_attempts_per_target - 1:
                    time.sleep(self._retry.base_backoff_s * (2**n))
        raise RetriableError(str(last_exc))
