from agentctl.config import load_config

def test_load_config_from_yaml(tmp_path):
    cfg_file = tmp_path / "agentctl.yaml"
    cfg_file.write_text(
        "routes:\n"
        "  default: [anthropic/claude-opus-4-8, anthropic/claude-sonnet-4-6]\n"
        "prices:\n"
        "  claude-opus-4-8: [5.0, 25.0]\n"
        "cache_enabled: true\n"
        "cache_ttl_s: 600\n"
        "profile: dev\n"
        "db_path: ':memory:'\n"
        "retry:\n"
        "  max_attempts_per_target: 2\n"
        "  base_backoff_s: 0.01\n"
        "  timeout_s: 30\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.routes["default"] == ["anthropic/claude-opus-4-8", "anthropic/claude-sonnet-4-6"]
    assert cfg.prices["claude-opus-4-8"] == (5.0, 25.0)
    assert cfg.retry.max_attempts_per_target == 2

def test_load_config_defaults_when_missing():
    cfg = load_config(None)
    assert cfg.profile == "dev"
    assert cfg.cache_enabled is True
