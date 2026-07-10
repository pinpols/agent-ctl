# agent_ctl/obs/metrics.py
"""Prometheus 指标(可观测)。

设计:`Metrics` 持有一组指标(各自独立 CollectorRegistry,可多实例不冲突);
`MetricsRegistry.get()` 是进程级单例(网关/server 用);prometheus_client 未装时
`MetricsRegistry.get()` 返回 None,模块级 `record_call`/`render` 全 no-op
(不硬依赖,符合"治理侧不影响主调用"原则)。
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("agent_ctl.metrics")


def _prometheus_available() -> bool:
    try:
        import prometheus_client  # noqa: F401

        return True
    except ImportError:
        return False


class Metrics:
    """一组网关指标。各实例用独立 registry,故可在测试里多次实例化不撞名。"""

    def __init__(self, registry=None) -> None:
        from prometheus_client import CollectorRegistry, Counter, Histogram

        self.registry = registry or CollectorRegistry()
        self.requests_total = Counter(
            "agentctl_requests_total",
            "调用总数",
            ["provider", "model", "status"],
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "agentctl_request_duration_seconds",
            "端到端延迟(秒)",
            ["provider"],
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "agentctl_tokens_total",
            "token 数",
            ["provider", "model", "direction"],
            registry=self.registry,
        )
        self.cost_usd_total = Counter(
            "agentctl_cost_usd_total",
            "成本 USD",
            ["provider", "model"],
            registry=self.registry,
        )
        self.errors_total = Counter(
            "agentctl_errors_total",
            "错误数",
            ["provider", "error_type"],
            registry=self.registry,
        )
        self.cache_hits_total = Counter(
            "agentctl_cache_hits_total", "缓存命中数", registry=self.registry
        )

    def record_request(
        self,
        *,
        model: str,
        provider: str,
        status: str,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
        error_type: str | None = None,
    ) -> None:
        self.requests_total.labels(provider, model, status).inc()
        self.request_duration_seconds.labels(provider).observe(
            (latency_ms or 0) / 1000.0
        )
        self.tokens_total.labels(provider, model, "input").inc(input_tokens or 0)
        self.tokens_total.labels(provider, model, "output").inc(output_tokens or 0)
        if cost_usd:
            self.cost_usd_total.labels(provider, model).inc(cost_usd)
        if error_type:
            self.errors_total.labels(provider, error_type).inc()

    def record_cache_hit(self) -> None:
        self.cache_hits_total.inc()

    def export_text(self) -> str:
        from prometheus_client import generate_latest

        return generate_latest(self.registry).decode()


class MetricsRegistry:
    """进程级单例(网关/server 共用)。prometheus 未装时为 None。"""

    _instance: Metrics | None = None
    _resolved = False
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> Metrics | None:
        # 双检加锁 + 赋值顺序(先 _instance 后 _resolved):原实现先置 _resolved=True
        # 再构造 Metrics,并发下另一线程会在窗口内拿到 None → 该请求的指标静默丢失。
        if not cls._resolved:
            with cls._lock:
                if not cls._resolved:
                    cls._instance = Metrics() if _prometheus_available() else None
                    cls._resolved = True
        return cls._instance


# ── 网关/server 接线用的模块级便捷函数(无 prometheus 时 no-op)──────────────


def record_call(
    *,
    model_resolved: str | None,
    status: str,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None,
    cache_hit: bool,
    error_type: str | None = None,
) -> None:
    """网关每次调用上报。任何异常吞掉(指标不影响主流程)。"""
    m = MetricsRegistry.get()
    if m is None:
        return
    try:
        if cache_hit:
            m.record_cache_hit()
            provider, model = "cache", "cache"
        elif model_resolved:
            provider, _, model = model_resolved.partition("/")
            model = model or model_resolved
        else:
            provider, model = "none", "none"
        m.record_request(
            model=model,
            provider=provider,
            status=status,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            error_type=error_type,
        )
    except Exception as exc:  # 指标失败绝不影响主流程
        log.warning("metrics record failed (ignored): %s", exc)


def render() -> tuple[str, bytes]:
    """(content_type, body) 供 /metrics。prometheus 未装时返回提示文本。"""
    m = MetricsRegistry.get()
    if m is None:
        return "text/plain; charset=utf-8", b"# prometheus_client not installed\n"
    from prometheus_client import CONTENT_TYPE_LATEST

    return CONTENT_TYPE_LATEST, m.export_text().encode()
