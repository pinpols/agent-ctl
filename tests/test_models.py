# tests/test_models.py
from agentctl.models import Target, CallRecord, Attempt
from agentctl.errors import RetriableError, TerminalError, GatewayError


def test_target_parse_and_name():
    t = Target.parse("anthropic/claude-opus-4-8")
    assert t.provider == "anthropic"
    assert t.model == "claude-opus-4-8"
    assert t.name == "anthropic/claude-opus-4-8"


def test_call_record_minimal():
    rec = CallRecord(
        id="abc",
        consumer="ops-agent",
        model_requested="default",
        model_resolved="anthropic/claude-opus-4-8",
        status="success",
        latency_ms=120,
        input_tokens=10,
        output_tokens=5,
        attempts=[
            Attempt(
                provider="anthropic",
                model="claude-opus-4-8",
                outcome="success",
                latency_ms=120,
                error=None,
            )
        ],
    )
    assert rec.status == "success"
    assert rec.attempts[0].outcome == "success"


def test_error_hierarchy():
    assert issubclass(RetriableError, GatewayError)
    assert issubclass(TerminalError, GatewayError)
