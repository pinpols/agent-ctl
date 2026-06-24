# agent_ctl/core/gateway.py
from __future__ import annotations

import logging
import random
import time
import uuid

from agent_ctl.config import RetryConfig
from agent_ctl.core.cache import make_key
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.router import Router
from agent_ctl.errors import (
    AllTargetsFailed,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import (
    Attempt,
    CallRecord,
    NormalizedRequest,
    NormalizedResponse,
    Target,
)
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

        # 守护 route↔provider 一致性:在构建期快速失败,避免 invoke 时裸 KeyError
        missing = {
            t.provider
            for t in self._router.all_targets()
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
                    time.sleep(self._backoff_s(n))
        raise RetriableError(str(last_exc))

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
        for idx, target in enumerate(targets):
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
            provider = self._providers[target.provider]
            try:
                resp = self._invoke_target(provider, target, request, all_attempts)
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
                # 终态(鉴权/参数):不回退,attempts 已由 _invoke_target 记入 all_attempts
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
        if self._store is None:
            return
        try:
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
            rec = CallRecord(
                id=str(uuid.uuid4()),
                ts=time.time(),
                latency_ms=int((time.monotonic() - started) * 1000),
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
