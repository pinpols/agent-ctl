# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import itertools
import time
from collections.abc import Generator, Iterator

from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    DeadlineExceeded,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import Attempt, NormalizedRequest, NormalizedResponse, StreamChunk


class StreamRunnerMixin:
    def invoke_stream(self, request: NormalizedRequest) -> Iterator[StreamChunk]:
        """真·流式:逐块产出文本增量,末块 done=True 带最终计量。"""
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
            if self._deadline_exceeded(target, deadline, attempts):
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
            if self._circuit_blocked(target, attempts):
                continue
            provider = self._providers[target.provider]
            stream_fn = getattr(provider, "stream", None)

            if stream_fn is None:
                yielded = yield from self._invoke_buffered_stream_target(
                    provider, target, request, attempts, deadline, started, meta, idx
                )
                if yielded:
                    return
                continue

            yielded = yield from self._invoke_native_stream_target(
                stream_fn, target, request, attempts, deadline, started, meta, idx
            )
            if yielded:
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

    def _invoke_buffered_stream_target(
        self, provider, target, request, attempts, deadline, started, meta, idx
    ) -> Generator[StreamChunk, None, bool]:
        try:
            resp = self._invoke_target(provider, target, request, attempts, deadline)
        except TerminalError as exc:
            self._circuit.record_failure(target.provider)
            self._capture_stream_error(
                request, meta, started, attempts, "terminal", str(exc), target.name
            )
            raise
        except DeadlineExceeded:
            return False
        except (RetriableError, TimeoutError):
            self._circuit.record_failure(target.provider)
            return False
        self._circuit.record_success(target.provider)
        self._capture_stream_ok(request, meta, started, target, attempts, resp, idx)
        yield from self._chunks_of(resp)
        return True

    def _invoke_native_stream_target(
        self, stream_fn, target, request, attempts, deadline, started, meta, idx
    ) -> Generator[StreamChunk, None, bool]:
        timeout = self._timeout_within(deadline)
        if timeout is None:
            attempts.append(self._attempt(target, "deadline", 0, "deadline exceeded"))
            return False
        t0 = time.monotonic()
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
            return False

        parts: list[str] = []
        fr: str | None = None
        it = ot = 0
        tcs: list | None = None
        try:
            for chunk in itertools.chain([] if first is None else [first], gen):
                if deadline is not None and time.monotonic() >= deadline:
                    # 开流后墙钟预算耗尽:逐块截断(已发部分保留),按 deadline 落库 + 收尾 done。
                    # 注:无法中断单个已阻塞的 next() 读取(那由 provider SDK 的 read timeout 兜),
                    # 本检查约束的是「长流/多块」的总墙钟,防止流式完全无视 deadline。
                    attempts.append(
                        self._attempt(
                            target, "deadline", t0, "deadline exceeded mid-stream"
                        )
                    )
                    self._capturer.record(
                        request,
                        meta,
                        started,
                        model_resolved=target.name,
                        attempts=attempts,
                        resp=NormalizedResponse(
                            text="".join(parts),
                            finish_reason="length",
                            input_tokens=it,
                            output_tokens=ot,
                        ),
                        status="deadline",
                        cache_hit=False,
                        cache_key=None,
                        error_type="deadline",
                        error_message=None,
                    )
                    self._capturer.log(
                        request,
                        meta,
                        "deadline",
                        target.name,
                        "deadline",
                        False,
                        started,
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
            self._circuit.record_failure(target.provider)
            attempts.append(self._attempt(target, "stream_error", t0, str(exc)))
            self._capture_stream_error(
                request, meta, started, attempts, "stream", str(exc), target.name
            )
            raise GatewayError(str(exc)) from exc
        except BaseException:
            attempts.append(self._attempt(target, "aborted", t0, "client disconnected"))
            self._capturer.record(
                request,
                meta,
                started,
                model_resolved=target.name,
                attempts=attempts,
                resp=NormalizedResponse(
                    text="".join(parts),
                    finish_reason=fr,
                    input_tokens=it,
                    output_tokens=ot,
                ),
                status="aborted",
                cache_hit=False,
                cache_key=None,
                error_type="client_abort",
                error_message=None,
            )
            self._capturer.log(
                request, meta, "aborted", target.name, "client_abort", False, started
            )
            raise
        self._circuit.record_success(target.provider)
        attempts.append(self._attempt(target, "success", t0, None))
        resp = NormalizedResponse(
            text="".join(parts),
            finish_reason=fr,
            input_tokens=it,
            output_tokens=ot,
            tool_calls=len(tcs or []),
        )
        self._capture_stream_ok(request, meta, started, target, attempts, resp, idx)
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

    def _capture_stream_ok(self, request, meta, started, target, attempts, resp, idx):
        status = "success" if idx == 0 else "fallback_success"
        self._capturer.record(
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
        self._capturer.log(request, meta, status, target.name, None, False, started)

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
        self._capturer.record(
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
        self._capturer.log(
            request, meta, "error", model_resolved, error_type, False, started
        )
