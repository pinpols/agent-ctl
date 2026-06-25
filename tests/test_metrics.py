"""Prometheus metrics collection tests."""

from agent_ctl.obs.metrics import Metrics, MetricsRegistry


def test_metrics_initialization():
    """Metrics registry initializes counters and histograms."""
    metrics = Metrics()
    assert metrics.requests_total is not None
    assert metrics.request_duration_seconds is not None
    assert metrics.tokens_total is not None


def test_record_request_success():
    """Record a successful request."""
    metrics = Metrics()
    metrics.record_request(
        model="deepseek-chat",
        provider="deepseek",
        status="success",
        latency_ms=150,
        input_tokens=20,
        output_tokens=50,
        cost_usd=0.0001,
    )
    assert metrics.requests_total is not None


def test_record_request_failure():
    """Record a failed request."""
    metrics = Metrics()
    metrics.record_request(
        model="gpt-4o",
        provider="openai",
        status="error",
        latency_ms=3000,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0,
        error_type="insufficient_quota",
    )
    assert metrics.requests_total is not None


def test_record_cache_hit():
    """Record a cache hit."""
    metrics = Metrics()
    metrics.record_cache_hit()
    assert metrics.cache_hits_total is not None


def test_registry_singleton():
    """Registry is a process-wide singleton."""
    r1 = MetricsRegistry.get()
    r2 = MetricsRegistry.get()
    assert r1 is r2


def test_metrics_export_format():
    """Metrics can be exported as Prometheus text format."""
    metrics = Metrics()
    metrics.record_request(
        model="claude-opus-4-8",
        provider="anthropic",
        status="success",
        latency_ms=200,
        input_tokens=100,
        output_tokens=200,
        cost_usd=0.01,
    )
    text = metrics.export_text()
    assert isinstance(text, str)
    assert "agentctl_requests_total" in text
