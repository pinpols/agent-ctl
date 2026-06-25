# agent_ctl/core/gateway.py
from __future__ import annotations

import itertools
import logging
import random
import time
import uuid
from collections.abc import Iterator

from agent_ctl.config import RetryConfig
from agent_ctl.core.budget import BudgetGuard
from agent_ctl.core.cache import make_key
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.router import Router
from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import (
    Attempt,
    CallRecord,
    EmbeddingResponse,
    NormalizedRequest,
    NormalizedResponse,
    StreamChunk,
    Target,
)
from agent_ctl.obs import metrics
from agent_ctl.providers.base import Provider
from agent_ctl.store.redaction import redact, redact_messages

log = logging.getLogger("agent_ctl.gateway")


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
        cache_tool_responses: bool = False,
        circuit: CircuitBreaker | None = None,
        request_deadline_s: float = 0.0,
        budget: BudgetGuard | None = None,
    ) -> None:
        self._router = router
        self._providers = providers
        self._cost = cost_meter
        self._store = store
        self._cache = cache
        self._retry = retry or RetryConfig()
        self._cache_enabled = cache_enabled
        self._cache_ttl_s = cache_ttl_s
        self._cache_tool_responses = cache_tool_responses
        self._circuit = circuit or CircuitBreaker(failure_threshold=0)  # 默认关闭
        self._deadline_s = request_deadline_s  # 单次调用墙钟总预算;0=不封顶
        self._budget = budget or BudgetGuard()  # 默认空=不限

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
                raise RetriableError("request deadline exceeded")
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
        if not (self._cache and self._cache_enabled):
            return None
        has_tool_shape = bool(request.tools or request.tool_choice)
        if has_tool_shape and not self._cache_tool_responses:
            return None
        return make_key(request)

    def invoke(self, request: NormalizedRequest) -> NormalizedResponse:
        started = time.monotonic()
        meta = request.metadata or {}
        cache_key = self._cache_key_for(request)

        if cache_key:
            cached = self._safe_cache_get(cache_key)
            if cached is not None:
                self._capture(
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
                self._log_capture(request, meta, "success", None, None, True, started)
                return cached

        # 预算闸:已超上限 → 打 provider 前短路(缓存命中走免费路径不受此限)
        try:
            self._budget.check(meta.get("consumer", "unknown"))
        except BudgetExceeded as exc:
            self._capture(
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
            self._log_capture(request, meta, "error", None, "budget", False, started)
            raise

        try:
            targets = self._router.resolve(request.model)
        except Exception as exc:
            self._capture(
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
            self._log_capture(request, meta, "error", None, "routing", False, started)
            raise GatewayError(str(exc)) from exc
        all_attempts: list[Attempt] = []  # 共享:_invoke_target 往里 append,失败也留痕
        deadline = self._deadline_for(started)
        for idx, target in enumerate(targets):
            if deadline is not None and time.monotonic() >= deadline:
                # 墙钟总预算耗尽 → 停止回退,余下目标不再尝试(留痕)
                all_attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="deadline",
                        latency_ms=0,
                        error="request deadline exceeded",
                    )
                )
                break
            if target.provider not in self._providers:
                # '/'-直连到未注册 provider:抛类型化错误而非裸 KeyError
                self._capture(
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
                self._log_capture(
                    request, meta, "error", target.name, "provider", False, started
                )
                raise GatewayError(
                    f"unregistered provider: {target.provider!r} (model={request.model!r})"
                )
            if not self._circuit.allow(target.provider):
                # 该 provider 熔断开路 → 跳过,试下一目标(留痕)
                all_attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="circuit_open",
                        latency_ms=0,
                        error="circuit open",
                    )
                )
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
                self._capture(
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
                self._log_capture(
                    request, meta, status, target.name, None, False, started
                )
                return resp
            except TerminalError as exc:
                # 终态(鉴权/参数/欠费):计入熔断(持续 4xx 应短路该 provider),但不回退
                self._circuit.record_failure(target.provider)
                self._capture(
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
                self._log_capture(
                    request, meta, "error", target.name, "terminal", False, started
                )
                raise
            except RetriableError:
                self._circuit.record_failure(target.provider)
                continue  # 可重试耗尽 → 回退下一目标(attempts 已记入)

        self._capture(
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
        self._log_capture(request, meta, "error", None, "all_failed", False, started)
        raise AllTargetsFailed(f"all targets failed for model {request.model!r}")

    def invoke_stream(self, request: NormalizedRequest) -> Iterator[StreamChunk]:
        """真·流式:逐块产出文本增量,末块 done=True 带最终计量。

        治理一致:预算闸/路由/熔断/deadline 照旧;**开流前**失败可回退下一目标,
        一旦首块已出即提交该目标不再回退(已发字节无法回退,业界一致)。无原生
        stream 能力的 provider 退化为缓冲式(跑非流式再切块,保留全部治理)。
        捕获在流结束后按累计文本+计量落一条记录(与非流式同一 _capture)。
        """
        started = time.monotonic()
        meta = request.metadata or {}
        consumer = meta.get("consumer", "unknown")
        try:
            self._budget.check(consumer)
        except BudgetExceeded as exc:
            self._capture_stream_error(request, meta, started, [], "budget", str(exc))
            raise
        try:
            targets = self._router.resolve(request.model)
        except Exception as exc:
            self._capture_stream_error(request, meta, started, [], "routing", str(exc))
            raise GatewayError(str(exc)) from exc

        deadline = self._deadline_for(started)
        attempts: list[Attempt] = []
        for idx, target in enumerate(targets):
            if deadline is not None and time.monotonic() >= deadline:
                attempts.append(
                    self._attempt(target, "deadline", 0, "deadline exceeded")
                )
                break
            if target.provider not in self._providers:
                self._capture_stream_error(
                    request,
                    meta,
                    started,
                    attempts,
                    "provider",
                    f"unregistered provider: {target.provider!r}",
                    target.name,
                )
                raise GatewayError(f"unregistered provider: {target.provider!r}")
            if not self._circuit.allow(target.provider):
                attempts.append(
                    self._attempt(target, "circuit_open", 0, "circuit open")
                )
                continue
            provider = self._providers[target.provider]
            t0 = time.monotonic()
            stream_fn = getattr(provider, "stream", None)

            if stream_fn is None:
                # 退化:缓冲式(含重试),成功切块产出;失败按类型回退/抛
                try:
                    resp = self._invoke_target(
                        provider, target, request, attempts, deadline
                    )
                except TerminalError as exc:
                    self._circuit.record_failure(target.provider)
                    self._capture_stream_error(
                        request,
                        meta,
                        started,
                        attempts,
                        "terminal",
                        str(exc),
                        target.name,
                    )
                    raise
                except (RetriableError, TimeoutError):
                    self._circuit.record_failure(target.provider)
                    continue
                self._circuit.record_success(target.provider)
                self._capture_stream_ok(
                    request, meta, started, target, attempts, resp, idx
                )
                yield from self._chunks_of(resp)
                return

            # 真流式:先拉首块,允许开流前回退
            timeout = self._timeout_within(deadline)
            if timeout is None:
                attempts.append(
                    self._attempt(target, "deadline", 0, "deadline exceeded")
                )
                break
            gen = stream_fn(target, request, timeout)
            try:
                first: StreamChunk | None = next(gen)
            except StopIteration:
                first = None
            except TerminalError as exc:
                self._circuit.record_failure(target.provider)
                attempts.append(self._attempt(target, "terminal", t0, str(exc)))
                self._capture_stream_error(
                    request, meta, started, attempts, "terminal", str(exc), target.name
                )
                raise
            except (RetriableError, TimeoutError) as exc:
                self._circuit.record_failure(target.provider)
                attempts.append(self._attempt(target, "retriable", t0, str(exc)))
                continue  # 开流前失败 → 回退下一目标

            # 已开流 → 提交此 target,后续不回退
            self._circuit.record_success(target.provider)
            parts: list[str] = []
            fr: str | None = None
            it = ot = 0
            try:
                for chunk in itertools.chain([] if first is None else [first], gen):
                    if chunk is None:
                        continue
                    if chunk.done:
                        fr, it, ot = (
                            chunk.finish_reason,
                            chunk.input_tokens,
                            chunk.output_tokens,
                        )
                    elif chunk.text:
                        parts.append(chunk.text)
                        yield chunk
            except Exception as exc:  # 流中途失败:已发字节无法回退 → 记错并抛
                attempts.append(self._attempt(target, "stream_error", t0, str(exc)))
                self._capture_stream_error(
                    request, meta, started, attempts, "stream", str(exc), target.name
                )
                raise GatewayError(str(exc)) from exc
            attempts.append(self._attempt(target, "success", t0, None))
            resp = NormalizedResponse(
                text="".join(parts),
                finish_reason=fr,
                input_tokens=it,
                output_tokens=ot,
            )
            self._capture_stream_ok(request, meta, started, target, attempts, resp, idx)
            yield StreamChunk(
                done=True, finish_reason=fr, input_tokens=it, output_tokens=ot
            )
            return

        self._capture_stream_error(
            request,
            meta,
            started,
            attempts,
            "all_failed",
            attempts[-1].error if attempts else None,
        )
        raise AllTargetsFailed(f"all stream targets failed for model {request.model!r}")

    def _attempt(self, target: Target, outcome: str, t0: float, error: str | None):
        latency = int((time.monotonic() - t0) * 1000) if t0 else 0
        return Attempt(
            provider=target.provider,
            model=target.model,
            outcome=outcome,
            latency_ms=latency,
            error=error,
        )

    @staticmethod
    def _chunks_of(resp: NormalizedResponse) -> Iterator[StreamChunk]:
        if resp.text:
            yield StreamChunk(text=resp.text)
        yield StreamChunk(
            done=True,
            finish_reason=resp.finish_reason,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )

    def _capture_stream_ok(self, request, meta, started, target, attempts, resp, idx):
        status = "success" if idx == 0 else "fallback_success"
        self._capture(
            request,
            meta,
            started,
            model_resolved=target.name,
            attempts=attempts,
            resp=resp,
            status=status,
            cache_hit=False,
            cache_key=None,
            error_type=None,
            error_message=None,
        )
        self._log_capture(request, meta, status, target.name, None, False, started)

    def _capture_stream_error(
        self,
        request,
        meta,
        started,
        attempts,
        error_type,
        error_message,
        model_resolved=None,
    ):
        self._capture(
            request,
            meta,
            started,
            model_resolved=model_resolved,
            attempts=attempts,
            resp=None,
            status="error",
            cache_hit=False,
            cache_key=None,
            error_type=error_type,
            error_message=error_message,
        )
        self._log_capture(
            request, meta, "error", model_resolved, error_type, False, started
        )

    def embed(
        self, model: str, inputs: list[str], metadata: dict | None = None
    ) -> EmbeddingResponse:
        """Embeddings 走与 invoke 同一治理:路由→熔断跳过→回退→留痕→捕获/指标。

        不支持 embed 的目标(如 Anthropic)在回退链里被跳过(留痕 no_embed)。
        embeddings 单次调用不做 per-target 重试(provider SDK 自带重试)。
        """
        started = time.monotonic()
        meta = metadata or {}
        try:
            self._budget.check(meta.get("consumer", "unknown"))
        except BudgetExceeded as exc:
            self._capture_embed(
                model, meta, started, None, [], None, "error", "budget", str(exc)
            )
            raise
        try:
            targets = self._router.resolve(model)
        except Exception as exc:
            self._capture_embed(
                model, meta, started, None, [], None, "error", "routing", str(exc)
            )
            raise GatewayError(str(exc)) from exc
        attempts: list[Attempt] = []
        for idx, target in enumerate(targets):
            if target.provider not in self._providers:
                self._capture_embed(
                    model,
                    meta,
                    started,
                    target.name,
                    attempts,
                    None,
                    "error",
                    "provider",
                    f"unregistered provider: {target.provider!r}",
                )
                raise GatewayError(
                    f"unregistered provider: {target.provider!r} (model={model!r})"
                )
            provider = self._providers[target.provider]
            embed_fn = getattr(provider, "embed", None)
            if embed_fn is None:
                # 该 provider 无 embeddings 能力 → 跳过试下一目标(留痕)
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="no_embed",
                        latency_ms=0,
                        error="provider has no embeddings capability",
                    )
                )
                continue
            if not self._circuit.allow(target.provider):
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="circuit_open",
                        latency_ms=0,
                        error="circuit open",
                    )
                )
                continue
            started_t = time.monotonic()
            try:
                resp = embed_fn(target, inputs, self._retry.timeout_s)
                self._circuit.record_success(target.provider)
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="success",
                        latency_ms=int((time.monotonic() - started_t) * 1000),
                    )
                )
                status = "success" if idx == 0 else "fallback_success"
                self._capture_embed(
                    model,
                    meta,
                    started,
                    target.name,
                    attempts,
                    resp,
                    status,
                    None,
                    None,
                )
                return resp
            except TerminalError as exc:
                self._circuit.record_failure(target.provider)
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="terminal",
                        latency_ms=int((time.monotonic() - started_t) * 1000),
                        error=str(exc),
                    )
                )
                self._capture_embed(
                    model,
                    meta,
                    started,
                    target.name,
                    attempts,
                    None,
                    "error",
                    "terminal",
                    str(exc),
                )
                raise
            except (RetriableError, TimeoutError) as exc:
                self._circuit.record_failure(target.provider)
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="retriable",
                        latency_ms=int((time.monotonic() - started_t) * 1000),
                        error=str(exc),
                    )
                )
                continue
        self._capture_embed(
            model,
            meta,
            started,
            None,
            attempts,
            None,
            "error",
            "all_failed",
            attempts[-1].error if attempts else None,
        )
        raise AllTargetsFailed(f"all embedding targets failed for model {model!r}")

    def _capture_embed(
        self,
        model,
        meta,
        started,
        model_resolved,
        attempts,
        resp,
        status,
        error_type,
        error_message,
    ) -> None:
        latency_ms = int((time.monotonic() - started) * 1000)
        input_tokens = resp.input_tokens if resp else 0
        cost = None
        if resp is not None and model_resolved:
            try:
                cost = self._cost.cost(model_resolved, input_tokens, 0)
            except Exception as exc:
                log.warning("embed cost calculation failed (cost=None): %s", exc)
        self._budget.add(meta.get("consumer", "unknown"), cost)
        metrics.record_call(
            model_resolved=model_resolved,
            status=status,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=0,
            cost_usd=cost,
            cache_hit=False,
            error_type=error_type,
        )
        if self._store is None:
            return
        try:
            rec = CallRecord(
                id=str(uuid.uuid4()),
                ts=time.time(),
                latency_ms=latency_ms,
                consumer=meta.get("consumer", "unknown"),
                call_site=meta.get("call_site"),
                trace_id=meta.get("trace_id"),
                model_requested=model,
                params={"embed": True, "n_inputs": len(resp.vectors) if resp else 0},
                model_resolved=model_resolved,
                attempts=attempts,
                input_tokens=input_tokens,
                cost_usd=cost,
                status=status,
                error_type=error_type,
                error_message_redacted=redact(error_message) if error_message else None,
                last_error=redact(attempts[-1].error) if attempts else None,
            )
            self._store.save(rec)
        except Exception as exc:  # fail-open
            log.warning("embed capture failed (fail-open): %s", exc)

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

    def _capture(
        self,
        request,
        meta,
        started,
        *,
        model_resolved,
        attempts,
        resp,
        status,
        cache_hit,
        cache_key,
        error_type,
        error_message,
    ) -> None:
        latency_ms = int((time.monotonic() - started) * 1000)
        cost = None
        if cache_hit:
            cost = 0.0  # 命中缓存=省下真实开销
        elif resp is not None and model_resolved:
            try:
                cost = self._cost.cost(
                    model_resolved, resp.input_tokens, resp.output_tokens
                )
            except Exception as exc:
                log.warning("cost calculation failed (cost=None): %s", exc)
        # 实际成本计入预算累计(None/0 自动忽略)
        self._budget.add(meta.get("consumer", "unknown"), cost)
        # 指标:无论是否落库都上报(Prometheus,未装则 no-op)
        metrics.record_call(
            model_resolved=model_resolved,
            status=status,
            latency_ms=latency_ms,
            input_tokens=resp.input_tokens if resp else 0,
            output_tokens=resp.output_tokens if resp else 0,
            cost_usd=cost,
            cache_hit=cache_hit,
            error_type=error_type,
        )
        if self._store is None:
            return
        try:
            rec = CallRecord(
                id=str(uuid.uuid4()),
                ts=time.time(),
                latency_ms=latency_ms,
                consumer=meta.get("consumer", "unknown"),
                call_site=meta.get("call_site"),
                trace_id=meta.get("trace_id"),
                model_requested=request.model,
                params={
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "has_tools": bool(request.tools),
                },
                messages_redacted=redact_messages(request.messages),
                prompt_version=meta.get("prompt_version"),
                model_resolved=model_resolved,
                attempts=attempts,
                output_redacted=redact(resp.text) if resp else None,
                finish_reason=resp.finish_reason if resp else None,
                tool_calls=resp.tool_calls if resp else 0,
                input_tokens=resp.input_tokens if resp else 0,
                output_tokens=resp.output_tokens if resp else 0,
                cost_usd=cost,
                cache_hit=cache_hit,
                cache_key=cache_key,
                status=status,
                error_type=error_type,
                error_message_redacted=redact(error_message) if error_message else None,
                last_error=redact(attempts[-1].error) if attempts else None,
            )
            self._store.save(rec)
        except Exception as exc:  # fail-open:捕获绝不打断主调用
            log.warning("capture failed (fail-open): %s", exc)

    def _log_capture(
        self,
        request: NormalizedRequest,
        meta: dict,
        status: str,
        model_resolved: str | None,
        error_type: str | None,
        cache_hit: bool,
        started: float,
    ) -> None:
        log.info(
            "llm_call",
            extra={
                "trace_id": meta.get("trace_id"),
                "consumer": meta.get("consumer", "unknown"),
                "model_requested": request.model,
                "model_resolved": model_resolved,
                "status": status,
                "error_type": error_type,
                "cache_hit": cache_hit,
                "latency_ms": int((time.monotonic() - started) * 1000),
            },
        )
