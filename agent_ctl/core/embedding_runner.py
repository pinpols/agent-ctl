# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import time

from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import Attempt, EmbeddingResponse


class EmbeddingRunnerMixin:
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
            self._capturer.record_embed(
                model, meta, started, None, [], None, "error", "budget", str(exc)
            )
            raise
        try:
            targets = self._router.resolve(model)
        except Exception as exc:
            self._capturer.record_embed(
                model, meta, started, None, [], None, "error", "routing", str(exc)
            )
            raise GatewayError(str(exc)) from exc
        attempts: list[Attempt] = []
        deadline = self._deadline_for(started)
        for idx, target in enumerate(targets):
            if self._deadline_exceeded(target, deadline, attempts):
                break
            if target.provider not in self._providers:
                self._capturer.record_embed(
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
                attempts.append(
                    self._attempt(
                        target, "no_embed", 0, "provider has no embeddings capability"
                    )
                )
                continue
            if self._circuit_blocked(target, attempts):
                continue
            embed_timeout = self._timeout_within(deadline)
            if embed_timeout is None:
                attempts.append(
                    self._attempt(target, "deadline", 0, "request deadline exceeded")
                )
                break
            started_t = time.monotonic()
            try:
                resp = embed_fn(target, inputs, embed_timeout)
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
                self._capturer.record_embed(
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
                self._capturer.record_embed(
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
        self._capturer.record_embed(
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
