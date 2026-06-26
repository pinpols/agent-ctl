# agent_ctl/core/stream_runner.py
from __future__ import annotations

import itertools
import time
from collections.abc import Generator, Iterator

from agent_ctl.core._host import CallCtx, RunnerHost
from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    DeadlineExceeded,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import NormalizedRequest, NormalizedResponse, StreamChunk
from agent_ctl.providers.base import StreamingProvider


class StreamRunner:
    """真·流式执行(协作者):用 host 的治理面,不再继承 Gateway 私有成员。

    CallCtx 承载 (request, meta, started, deadline, attempts),把各辅助方法的参数从
    8–9 个压到 ctx + 少数变量。
    """

    def __init__(self, host: RunnerHost) -> None:
        self._host = host

    def invoke_stream(self, request: NormalizedRequest) -> Iterator[StreamChunk]:
        """逐块产出文本增量,末块 done=True 带最终计量。"""
        h = self._host
        started = time.monotonic()
        meta = request.metadata or {}
        ctx = CallCtx(request=request, meta=meta, started=started, deadline=None)
        try:
            h._budget.check(meta.get("consumer", "unknown"))
        except BudgetExceeded as exc:
            self._error(ctx, "budget", str(exc))
            raise
        try:
            targets = h._router.resolve(request.model)
        except Exception as exc:
            self._error(ctx, "routing", str(exc))
            raise GatewayError(str(exc)) from exc

        ctx.deadline = h._deadline_for(started)
        for idx, target in enumerate(targets):
            if h._deadline_exceeded(target, ctx.deadline, ctx.attempts):
                break
            if target.provider not in h._providers:
                self._error(
                    ctx,
                    "provider",
                    f"unregistered provider: {target.provider!r}",
                    target.name,
                )
                raise GatewayError(f"unregistered provider: {target.provider!r}")
            try:
                h._capturer.ensure_price(target.name)
            except TerminalError as exc:
                self._error(ctx, "pricing", str(exc), target.name)
                raise
            if h._circuit_blocked(target, ctx.attempts):
                continue
            provider = h._providers[target.provider]

            if not isinstance(provider, StreamingProvider):
                # 无原生流式能力 → 缓冲式降级(跑非流式再切块)
                yielded = yield from self._buffered_target(provider, target, ctx, idx)
            else:
                yielded = yield from self._native_target(
                    provider.stream, target, ctx, idx
                )
            if yielded:
                return

        self._error(ctx, "all_failed", ctx.attempts[-1].error if ctx.attempts else None)
        raise AllTargetsFailed(f"all stream targets failed for model {request.model!r}")

    def _buffered_target(
        self, provider, target, ctx: CallCtx, idx: int
    ) -> Generator[StreamChunk, None, bool]:
        h = self._host
        try:
            resp = h._invoke_target(
                provider, target, ctx.request, ctx.attempts, ctx.deadline
            )
        except TerminalError as exc:
            h._circuit.record_failure(target.provider)
            self._error(ctx, "terminal", str(exc), target.name)
            raise
        except DeadlineExceeded:
            return False
        except (RetriableError, TimeoutError):
            h._circuit.record_failure(target.provider)
            return False
        h._circuit.record_success(target.provider)
        self._ok(ctx, target, resp, idx)
        yield from self._chunks_of(resp)
        return True

    def _native_target(
        self, stream_fn, target, ctx: CallCtx, idx: int
    ) -> Generator[StreamChunk, None, bool]:
        h = self._host
        timeout = h._timeout_within(ctx.deadline)
        if timeout is None:
            ctx.attempts.append(h._attempt(target, "deadline", 0, "deadline exceeded"))
            return False
        t0 = time.monotonic()
        gen = stream_fn(target, ctx.request, timeout)
        try:
            first: StreamChunk | None = next(gen)
        except StopIteration:
            first = None
        except TerminalError as exc:
            h._circuit.record_failure(target.provider)
            ctx.attempts.append(h._attempt(target, "terminal", t0, str(exc)))
            self._error(ctx, "terminal", str(exc), target.name)
            raise
        except (RetriableError, TimeoutError) as exc:
            h._circuit.record_failure(target.provider)
            ctx.attempts.append(h._attempt(target, "retriable", t0, str(exc)))
            return False

        parts: list[str] = []
        fr: str | None = None
        it = ot = 0
        tcs: list | None = None
        try:
            for chunk in itertools.chain([] if first is None else [first], gen):
                if ctx.deadline is not None and time.monotonic() >= ctx.deadline:
                    # 开流后墙钟预算耗尽:逐块截断(已发部分保留),按 deadline 落库 + 收尾 done。
                    # 注:无法中断单个已阻塞的 next() 读取(那由 provider SDK 的 read timeout 兜),
                    # 本检查约束的是「长流/多块」的总墙钟,防止流式完全无视 deadline。
                    ctx.attempts.append(
                        h._attempt(
                            target, "deadline", t0, "deadline exceeded mid-stream"
                        )
                    )
                    self._capture(
                        ctx,
                        target.name,
                        NormalizedResponse(
                            text="".join(parts),
                            finish_reason="length",
                            input_tokens=it,
                            output_tokens=ot,
                        ),
                        "deadline",
                        "deadline",
                    )
                    yield StreamChunk(
                        done=True,
                        finish_reason="length",
                        input_tokens=it,
                        output_tokens=ot,
                    )
                    return True
                if chunk is None:
                    continue
                if chunk.done:
                    fr, it, ot = (
                        chunk.finish_reason,
                        chunk.input_tokens,
                        chunk.output_tokens,
                    )
                    tcs = chunk.tool_calls
                elif chunk.text:
                    parts.append(chunk.text)
                    yield chunk
        except Exception as exc:
            h._circuit.record_failure(target.provider)
            ctx.attempts.append(h._attempt(target, "stream_error", t0, str(exc)))
            self._error(ctx, "stream", str(exc), target.name)
            raise GatewayError(str(exc)) from exc
        except BaseException:
            ctx.attempts.append(
                h._attempt(target, "aborted", t0, "client disconnected")
            )
            self._capture(
                ctx,
                target.name,
                NormalizedResponse(
                    text="".join(parts),
                    finish_reason=fr,
                    input_tokens=it,
                    output_tokens=ot,
                ),
                "aborted",
                "client_abort",
            )
            raise
        h._circuit.record_success(target.provider)
        ctx.attempts.append(h._attempt(target, "success", t0, None))
        resp = NormalizedResponse(
            text="".join(parts),
            finish_reason=fr,
            input_tokens=it,
            output_tokens=ot,
            tool_calls=len(tcs or []),
        )
        self._ok(ctx, target, resp, idx)
        yield StreamChunk(
            done=True,
            finish_reason=fr,
            input_tokens=it,
            output_tokens=ot,
            tool_calls=tcs,
        )
        return True

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

    # ── 捕获便捷(ctx 折叠 request/meta/started)──────────────────────────────

    def _capture(self, ctx: CallCtx, model_resolved, resp, status, error_type) -> None:
        self._host._capturer.record(
            ctx.request,
            ctx.meta,
            ctx.started,
            model_resolved=model_resolved,
            attempts=ctx.attempts,
            resp=resp,
            status=status,
            cache_hit=False,
            cache_key=None,
            error_type=error_type,
            error_message=None,
        )
        self._host._capturer.log(
            ctx.request,
            ctx.meta,
            status,
            model_resolved,
            error_type,
            False,
            ctx.started,
        )

    def _ok(self, ctx: CallCtx, target, resp, idx: int) -> None:
        status = "success" if idx == 0 else "fallback_success"
        self._capture(ctx, target.name, resp, status, None)

    def _error(
        self, ctx: CallCtx, error_type, error_message, model_resolved=None
    ) -> None:
        self._host._capturer.record(
            ctx.request,
            ctx.meta,
            ctx.started,
            model_resolved=model_resolved,
            attempts=ctx.attempts,
            resp=None,
            status="error",
            cache_hit=False,
            cache_key=None,
            error_type=error_type,
            error_message=error_message,
        )
        self._host._capturer.log(
            ctx.request,
            ctx.meta,
            "error",
            model_resolved,
            error_type,
            False,
            ctx.started,
        )
