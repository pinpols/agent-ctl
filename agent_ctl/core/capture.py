# agent_ctl/core/capture.py
"""捕获 / 计量 / 可观测协作者。

把"算成本 → 计预算 → 上报指标 → 脱敏落库 → 结构化日志"这一组横切关注从 Gateway 抽出:
Gateway 只管控制流(路由/回退/熔断/deadline),把"每跳如何留痕"交给 Capturer。全程
fail-open——任何捕获侧异常都只告警,绝不打断真实调用。
"""

from __future__ import annotations

import logging
import time
import uuid

from agent_ctl.core.budget import BudgetGuard
from agent_ctl.core.cost import CostMeter
from agent_ctl.models import CallRecord
from agent_ctl.obs import metrics
from agent_ctl.store.redaction import redact, redact_messages

log = logging.getLogger("agent_ctl.gateway")


class Capturer:
    def __init__(self, cost_meter: CostMeter, store, budget: BudgetGuard) -> None:
        self._cost = cost_meter
        self._store = store
        self._budget = budget

    def ensure_price(self, model_resolved: str | None) -> None:
        if not model_resolved:
            return
        ensure = getattr(self._cost, "ensure_price", None)
        if ensure is not None:
            ensure(model_resolved)

    # ── 共享:成本/计量/落库(chat 与 embed 复用,各自只组装 CallRecord 形状)──────

    def _safe_cost(self, model_resolved, input_tokens, output_tokens) -> float | None:
        if not model_resolved:
            return None
        try:
            return self._cost.cost(model_resolved, input_tokens, output_tokens)
        except Exception as exc:
            log.warning("cost calculation failed (cost=None): %s", exc)
            return None

    def _meter(
        self,
        meta,
        *,
        model_resolved,
        status,
        latency_ms,
        input_tokens,
        output_tokens,
        cost,
        cache_hit,
        error_type,
    ) -> None:
        # 实际成本计入预算累计(None/0 自动忽略)+ 指标上报(Prometheus,未装则 no-op)
        self._budget.add(meta.get("consumer", "unknown"), cost)
        metrics.record_call(
            model_resolved=model_resolved,
            status=status,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cache_hit=cache_hit,
            error_type=error_type,
        )

    def _base_record(
        self,
        meta,
        *,
        latency_ms,
        model_requested,
        model_resolved,
        attempts,
        cost,
        status,
        error_type,
        error_message,
    ) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "latency_ms": latency_ms,
            "consumer": meta.get("consumer", "unknown"),
            "call_site": meta.get("call_site"),
            "trace_id": meta.get("trace_id"),
            "model_requested": model_requested,
            "model_resolved": model_resolved,
            "attempts": attempts,
            "cost_usd": cost,
            "status": status,
            "error_type": error_type,
            "error_message_redacted": redact(error_message) if error_message else None,
            "last_error": redact(attempts[-1].error) if attempts else None,
        }

    def _persist(self, **rec_kwargs) -> None:
        if self._store is None:
            return
        try:
            self._store.save(CallRecord(**rec_kwargs))
        except Exception as exc:  # fail-open:捕获绝不打断主调用
            log.warning("capture failed (fail-open): %s", exc)

    def record(
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
        """chat(invoke / 流式)一跳的捕获:成本/预算/指标 + 脱敏落库。"""
        latency_ms = int((time.monotonic() - started) * 1000)
        in_tok = resp.input_tokens if resp else 0
        out_tok = resp.output_tokens if resp else 0
        cost: float | None
        if cache_hit:
            cost = 0.0  # 命中缓存=省下真实开销
        else:
            cost = self._safe_cost(model_resolved, in_tok, out_tok) if resp else None
        self._meter(
            meta,
            model_resolved=model_resolved,
            status=status,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost=cost,
            cache_hit=cache_hit,
            error_type=error_type,
        )
        self._persist(
            **self._base_record(
                meta,
                latency_ms=latency_ms,
                model_requested=request.model,
                model_resolved=model_resolved,
                attempts=attempts,
                cost=cost,
                status=status,
                error_type=error_type,
                error_message=error_message,
            ),
            params={
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "has_tools": bool(request.tools),
            },
            messages_redacted=redact_messages(request.messages),
            prompt_version=meta.get("prompt_version"),
            output_redacted=redact(resp.text) if resp else None,
            finish_reason=resp.finish_reason if resp else None,
            tool_calls=resp.tool_calls if resp else 0,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_hit=cache_hit,
            cache_key=cache_key,
        )

    def record_embed(
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
        """embeddings 一跳的捕获(无 messages/output 文本,params 标 embed)。"""
        latency_ms = int((time.monotonic() - started) * 1000)
        in_tok = resp.input_tokens if resp else 0
        cost = self._safe_cost(model_resolved, in_tok, 0) if resp else None
        self._meter(
            meta,
            model_resolved=model_resolved,
            status=status,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=0,
            cost=cost,
            cache_hit=False,
            error_type=error_type,
        )
        self._persist(
            **self._base_record(
                meta,
                latency_ms=latency_ms,
                model_requested=model,
                model_resolved=model_resolved,
                attempts=attempts,
                cost=cost,
                status=status,
                error_type=error_type,
                error_message=error_message,
            ),
            params={"embed": True, "n_inputs": len(resp.vectors) if resp else 0},
            input_tokens=in_tok,
        )

    def log(
        self,
        request,
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
