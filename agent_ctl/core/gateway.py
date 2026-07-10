# agent_ctl/core/gateway.py
from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator

from agent_ctl.config import RetryConfig
from agent_ctl.core.budget import BudgetGuard
from agent_ctl.core.cache import make_key
from agent_ctl.core.capture import Capturer
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.embedding_runner import EmbeddingRunner
from agent_ctl.core.router import Router
from agent_ctl.core.stream_runner import StreamRunner
from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    DeadlineExceeded,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import (
    Attempt,
    EmbeddingResponse,
    NormalizedRequest,
    NormalizedResponse,
    StreamChunk,
    Target,
)
from agent_ctl.providers.base import Provider
from agent_ctl.providers.tooltrans import validate_local_content

log = logging.getLogger("agent_ctl.gateway")


class Gateway:
    """治理编排门面:持有 router/providers/circuit/budget/capturer + 缓存,
    把流式/embeddings 委派给协作者(StreamRunner/EmbeddingRunner),自身满足 RunnerHost。
    """

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
        cache_tool_responses: bool = False,
        circuit: CircuitBreaker | None = None,
        request_deadline_s: float = 0.0,
        budget: BudgetGuard | None = None,
    ) -> None:
        self._router = router
        self._providers = providers
        self._cache = cache
        self._retry = retry or RetryConfig()
        self._cache_enabled = cache_enabled
        self._cache_ttl_s = cache_ttl_s
        self._cache_tool_responses = cache_tool_responses
        self._circuit = circuit or CircuitBreaker(failure_threshold=0)  # 默认关闭
        self._deadline_s = request_deadline_s  # 单次调用墙钟总预算;0=不封顶
        self._budget = budget or BudgetGuard()  # 默认空=不限
        # Capturer 与 Gateway 必须共享同一 BudgetGuard 实例(check 在网关、add 在捕获)
        self._capturer = Capturer(cost_meter, store, self._budget)
        # 流式/embeddings 协作者:接收 self(满足 RunnerHost),非 mixin 继承
        self._stream = StreamRunner(self)
        self._embed = EmbeddingRunner(self)

        # 守护 route↔provider 一致性:在构建期快速失败,避免 invoke 时裸 KeyError。
        # 只校验 routes(必经);aliases 是可选项(共享配置里可能含本消费者没 key 的 provider),
        # 留到调用时再校验,故不在此 fail。
        missing = {
            t.provider
            for t in self._router.route_targets()
            if t.provider not in self._providers
        }
        if missing:
            raise GatewayError(
                f"unregistered provider(s) for route targets: {sorted(missing)}"
            )

    def _invoke_target(
        self,
        provider: Provider,
        target: Target,
        request: NormalizedRequest,
        attempts: list[Attempt],
        deadline: float | None = None,
    ) -> NormalizedResponse:
        """对单目标尝试(含重试)。每次尝试都 append 到调用方的 attempts(成功/失败均留痕)。

        deadline(墙钟绝对时刻):每次尝试把单次超时压到 min(配置超时, 剩余预算);
        预算耗尽则不再发起新尝试。
        """
        last_exc: Exception = RetriableError("no attempt made")
        for n in range(self._retry.max_attempts_per_target):
            timeout = self._timeout_within(deadline)
            if timeout is None:
                raise DeadlineExceeded("request deadline exceeded")
            started = time.monotonic()
            try:
                resp = provider.invoke(target, request, timeout)
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
                    time.sleep(self._backoff_s(n))
        raise RetriableError(str(last_exc))

    def _deadline_for(self, started: float) -> float | None:
        return started + self._deadline_s if self._deadline_s > 0 else None

    def _timeout_within(self, deadline: float | None) -> float | None:
        """本次尝试可用超时:无 deadline → 配置超时;有则压到剩余预算;已超 → None。"""
        if deadline is None:
            return self._retry.timeout_s
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        return min(self._retry.timeout_s, remaining)

    def _backoff_s(self, attempt_index: int) -> float:
        base = self._retry.base_backoff_s * (2**attempt_index)
        if base <= 0 or self._retry.jitter_ratio <= 0:
            return base
        jitter = base * self._retry.jitter_ratio
        return max(0.0, random.uniform(base - jitter, base + jitter))

    def _cache_key_for(self, request: NormalizedRequest) -> str | None:
        # 用 is None 而非真值判断:MemoryCache 实现了 __len__,空缓存真值为 False 会误判"无缓存"。
        if self._cache is None or not self._cache_enabled:
            return None
        has_tool_shape = bool(request.tools or request.tool_choice)
        if has_tool_shape and not self._cache_tool_responses:
            return None
        return make_key(request)

    def invoke(self, request: NormalizedRequest) -> NormalizedResponse:
        started = time.monotonic()
        meta = request.metadata or {}
        # 本地可判定的终态校验(多模态/tool_choice):在进任何 provider 路径前拒绝。
        # 若留到 provider 翻译层抛,会被记成 provider 失败计入熔断——5 个坏请求
        # 就能把健康 provider 的熔断打开(HTTP 边界另有一道,库形态调用走这里)。
        try:
            validate_local_content(request.messages, request.tool_choice)
        except TerminalError as exc:
            self._capturer.record(
                request,
                meta,
                started,
                model_resolved=None,
                attempts=[],
                resp=None,
                status="error",
                cache_hit=False,
                cache_key=None,
                error_type="validation",
                error_message=str(exc),
            )
            self._capturer.log(
                request, meta, "error", None, "validation", False, started
            )
            raise
        cache_key = self._cache_key_for(request)

        if cache_key:
            cached = self._safe_cache_get(cache_key)
            if cached is not None:
                self._capturer.record(
                    request,
                    meta,
                    started,
                    model_resolved=None,
                    attempts=[],
                    resp=cached,
                    status="success",
                    cache_hit=True,
                    cache_key=cache_key,
                    error_type=None,
                    error_message=None,
                )
                self._capturer.log(request, meta, "success", None, None, True, started)
                return cached

        # 预算闸:已超上限 → 打 provider 前短路(缓存命中走免费路径不受此限)
        try:
            self._budget.check(meta.get("consumer", "unknown"))
        except BudgetExceeded as exc:
            self._capturer.record(
                request,
                meta,
                started,
                model_resolved=None,
                attempts=[],
                resp=None,
                status="error",
                cache_hit=False,
                cache_key=cache_key,
                error_type="budget",
                error_message=str(exc),
            )
            self._capturer.log(request, meta, "error", None, "budget", False, started)
            raise

        try:
            targets = self._router.resolve(request.model)
        except Exception as exc:
            self._capturer.record(
                request,
                meta,
                started,
                model_resolved=None,
                attempts=[],
                resp=None,
                status="error",
                cache_hit=False,
                cache_key=cache_key,
                error_type="routing",
                error_message=str(exc),
            )
            self._capturer.log(request, meta, "error", None, "routing", False, started)
            raise GatewayError(str(exc)) from exc
        all_attempts: list[Attempt] = []  # 共享:_invoke_target 往里 append,失败也留痕
        deadline = self._deadline_for(started)
        for idx, target in enumerate(targets):
            if self._deadline_exceeded(target, deadline, all_attempts):
                break
            if target.provider not in self._providers:
                # '/'-直连到未注册 provider:抛类型化错误而非裸 KeyError
                self._capturer.record(
                    request,
                    meta,
                    started,
                    model_resolved=target.name,
                    attempts=all_attempts,
                    resp=None,
                    status="error",
                    cache_hit=False,
                    cache_key=cache_key,
                    error_type="provider",
                    error_message=f"unregistered provider: {target.provider!r}",
                )
                self._capturer.log(
                    request, meta, "error", target.name, "provider", False, started
                )
                raise GatewayError(
                    f"unregistered provider: {target.provider!r} (model={request.model!r})"
                )
            try:
                self._capturer.ensure_price(target.name)
            except TerminalError as exc:
                self._capturer.record(
                    request,
                    meta,
                    started,
                    model_resolved=target.name,
                    attempts=all_attempts,
                    resp=None,
                    status="error",
                    cache_hit=False,
                    cache_key=cache_key,
                    error_type="pricing",
                    error_message=str(exc),
                )
                self._capturer.log(
                    request, meta, "error", target.name, "pricing", False, started
                )
                raise
            if self._circuit_blocked(target, all_attempts):
                continue
            provider = self._providers[target.provider]
            try:
                resp = self._invoke_target(
                    provider, target, request, all_attempts, deadline
                )
                self._circuit.record_success(target.provider)
                status = "success" if idx == 0 else "fallback_success"
                if cache_key:
                    self._safe_cache_set(cache_key, resp)
                self._capturer.record(
                    request,
                    meta,
                    started,
                    model_resolved=target.name,
                    attempts=all_attempts,
                    resp=resp,
                    status=status,
                    cache_hit=False,
                    cache_key=cache_key,
                    error_type=None,
                    error_message=None,
                )
                self._capturer.log(
                    request, meta, status, target.name, None, False, started
                )
                return resp
            except TerminalError as exc:
                # 终态(鉴权/参数/欠费):计入熔断(持续 4xx 应短路该 provider),但不回退
                self._circuit.record_failure(target.provider)
                self._capturer.record(
                    request,
                    meta,
                    started,
                    model_resolved=target.name,
                    attempts=all_attempts,
                    resp=None,
                    status="error",
                    cache_hit=False,
                    cache_key=cache_key,
                    error_type="terminal",
                    error_message=str(exc),
                )
                self._capturer.log(
                    request, meta, "error", target.name, "terminal", False, started
                )
                raise
            except DeadlineExceeded:
                continue  # 墙钟预算耗尽:不计熔断,下一轮 deadline 守卫会停止回退
            except RetriableError:
                self._circuit.record_failure(target.provider)
                continue  # 可重试耗尽 → 回退下一目标(attempts 已记入)

        self._capturer.record(
            request,
            meta,
            started,
            model_resolved=None,
            attempts=all_attempts,
            resp=None,
            status="error",
            cache_hit=False,
            cache_key=cache_key,
            error_type="all_failed",
            error_message=all_attempts[-1].error if all_attempts else None,
        )
        self._capturer.log(request, meta, "error", None, "all_failed", False, started)
        raise AllTargetsFailed(f"all targets failed for model {request.model!r}")

    def invoke_stream(self, request: NormalizedRequest) -> Iterator[StreamChunk]:
        return self._stream.invoke_stream(request)

    def embed(
        self, model: str, inputs: list[str], metadata: dict | None = None
    ) -> EmbeddingResponse:
        return self._embed.embed(model, inputs, metadata)

    def _attempt(self, target: Target, outcome: str, t0: float, error: str | None):
        latency = int((time.monotonic() - t0) * 1000) if t0 else 0
        return Attempt(
            provider=target.provider,
            model=target.model,
            outcome=outcome,
            latency_ms=latency,
            error=error,
        )

    # ── 共享治理守卫(invoke / invoke_stream / embed 三个 runner 共用)──────────
    # 抽出避免三处各写一遍:此前 deadline 守卫、circuit-skip 散布多处,F2/F3 类修复
    # 不得不在多个 runner 重复改。集中到一处后,守卫语义只有一个真相源。

    def _deadline_exceeded(
        self, target: Target, deadline: float | None, attempts: list[Attempt]
    ) -> bool:
        """到达目标前墙钟总预算已耗尽 → 留 deadline 痕,返回 True(调用方应 break)。"""
        if deadline is not None and time.monotonic() >= deadline:
            attempts.append(
                self._attempt(target, "deadline", 0, "request deadline exceeded")
            )
            return True
        return False

    def _circuit_blocked(self, target: Target, attempts: list[Attempt]) -> bool:
        """该 provider 熔断开路 → 留 circuit_open 痕,返回 True(调用方应 continue 跳过)。"""
        if not self._circuit.allow(target.provider):
            attempts.append(self._attempt(target, "circuit_open", 0, "circuit open"))
            return True
        return False

    def _safe_cache_get(self, key):
        try:
            return self._cache.get(key)
        except Exception as exc:  # fail-open
            log.warning("cache get failed (fail-open): %s", exc)
            return None

    def _safe_cache_set(self, key, resp) -> None:
        try:
            self._cache.set(key, resp, self._cache_ttl_s)
        except Exception as exc:
            log.warning("cache set failed (fail-open): %s", exc)
