# tests/test_client.py
from agent_ctl.client.gateway_client import GatewayClient
from agent_ctl.config import Config, RetryConfig
from agent_ctl.providers.fake import FakeProvider


def test_client_messages_routes_and_returns(tmp_path):
    cfg = Config(
        routes={"default": ["fake/m"]},
        prices={"m": (5.0, 25.0)},
        cache_enabled=False,
        db_path=str(tmp_path / "c.db"),
        retry=RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0),
    )
    client = GatewayClient.from_config(cfg, providers={"fake": FakeProvider(["ok"])})
    resp = client.messages(
        "default", [{"role": "user", "content": "hi"}], consumer="ops-agent"
    )
    assert resp.text == "fake-ok"


def test_from_config_rejects_unregistered_provider(tmp_path):
    import pytest

    cfg = Config(routes={"default": ["openai/gpt"]}, db_path=str(tmp_path / "c.db"))
    with pytest.raises(ValueError, match="openai"):
        GatewayClient.from_config(cfg, providers={"fake": FakeProvider(["ok"])})
