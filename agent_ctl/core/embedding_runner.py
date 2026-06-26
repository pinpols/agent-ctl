# agent_ctl/core/embedding_runner.py
from __future__ import annotations

import time

from agent_ctl.core._host import RunnerHost
from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import Attempt, EmbeddingResponse
from agent_ctl.providers.base import EmbeddingProvider


class EmbeddingRunner:
    """Embeddings 执行(协作者):用 host 的治理面,不再继承 Gateway 私有成员。"""

    def __init__(self, host: RunnerHost) -> None:
        self._host = host

    def embed(
        self, model: str, inputs: list[str], metadata: dict | None = None
    ) -> EmbeddingResponse:
        """Embeddings 走与 invoke 同一治理:路由→熔断跳过→回退→留痕→捕获/指标。

        不支持 embed 的目标(如 Anthropic)在回退链里被跳过(留痕 no_embed)。
        embeddings 单次调用不做 per-target 重试(provider SDK 自带重试)。
        """
        h = self._host
        started = time.monotonic()
        meta = metadata or {}
        try:
            h._budget.check(meta.get("consumer", "unknown"))
        except BudgetExceeded as exc:
            h._capturer.record_embed(
                model, meta, started, None, [], None, "error", "budget", str(exc)
            )
            raise
        try:
            targets = h._router.resolve(model)
        except Exception as exc:
            h._capturer.record_embed(
                model, meta, started, None, [], None, "error", "routing", str(exc)
            )
            raise GatewayError(str(exc)) from exc
        attempts: list[Attempt] = []
        deadline = h._deadline_for(started)
        for idx, target in enumerate(targets):
            if h._deadline_exceeded(target, deadline, attempts):
                break
            if target.provider not in h._providers:
                h._capturer.record_embed(
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
            provider = h._providers[target.provider]
            if not isinstance(provider, EmbeddingProvider):
                attempts.append(
                    h._attempt(
                        target, "no_embed", 0, "provider has no embeddings capability"
                    )
                )
                continue
            embed_fn = provider.embed
            try:
                h._capturer.ensure_price(target.name)
            except TerminalError as exc:
                h._capturer.record_embed(
                    model,
                    meta,
                    started,
                    target.name,
                    attempts,
                    None,
                    "error",
                    "pricing",
                    str(exc),
                )
                raise
            if h._circuit_blocked(target, attempts):
                continue
            embed_timeout = h._timeout_within(deadline)
            if embed_timeout is None:
                attempts.append(
                    h._attempt(target, "deadline", 0, "request deadline exceeded")
                )
                break
            started_t = time.monotonic()
            try:
                resp = embed_fn(target, inputs, embed_timeout)
                h._circuit.record_success(target.provider)
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="success",
                        latency_ms=int((time.monotonic() - started_t) * 1000),
                    )
                )
                status = "success" if idx == 0 else "fallback_success"
                h._capturer.record_embed(
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
                h._circuit.record_failure(target.provider)
                attempts.append(
                    Attempt(
                        provider=target.provider,
                        model=target.model,
                        outcome="terminal",
                        latency_ms=int((time.monotonic() - started_t) * 1000),
                        error=str(exc),
                    )
                )
                h._capturer.record_embed(
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
                h._circuit.record_failure(target.provider)
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
        h._capturer.record_embed(
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
