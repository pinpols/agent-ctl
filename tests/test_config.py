import pytest
from pydantic import ValidationError

from agent_ctl.config import Config, RetryConfig, load_config


def test_load_config_from_yaml(tmp_path):
    cfg_file = tmp_path / "agent_ctl.yaml"
    cfg_file.write_text(
        "routes:\n"
        "  default: [anthropic/claude-opus-4-8, anthropic/claude-sonnet-4-6]\n"
        "prices:\n"
        "  claude-opus-4-8: [5.0, 25.0]\n"
        "cache_enabled: true\n"
        "cache_ttl_s: 600\n"
        "cache_tool_responses: true\n"
        "profile: dev\n"
        "db_path: ':memory:'\n"
        "retry:\n"
        "  max_attempts_per_target: 2\n"
        "  base_backoff_s: 0.01\n"
        "  jitter_ratio: 0.3\n"
        "  timeout_s: 30\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.routes["default"] == [
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-6",
    ]
    assert cfg.prices["claude-opus-4-8"] == (5.0, 25.0)
    assert cfg.cache_tool_responses is True
    assert cfg.retry.max_attempts_per_target == 2
    assert cfg.retry.jitter_ratio == 0.3


def test_load_config_defaults_when_missing():
    cfg = load_config(None)
    assert cfg.profile == "dev"
    assert cfg.cache_enabled is True


def test_retry_config_rejects_invalid_values():
    with pytest.raises(ValidationError):
        RetryConfig(max_attempts_per_target=0)
    with pytest.raises(ValidationError):
        RetryConfig(base_backoff_s=-0.1)
    with pytest.raises(ValidationError):
        RetryConfig(timeout_s=0)
    with pytest.raises(ValidationError):
        RetryConfig(jitter_ratio=1.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prices": {"bad": [-1.0, 1.0]}},
        {"cache_ttl_s": -1},
        {"cache_max_entries": -1},
        {"request_deadline_s": -1.0},
        {"budgets": {"consumer": -1.0}},
        {"budget_global": -1.0},
        {"circuit_failure_threshold": -1},
        {"circuit_cooldown_s": -1.0},
    ],
)
def test_config_rejects_negative_production_knobs(kwargs):
    with pytest.raises(ValidationError):
        Config(**kwargs)
