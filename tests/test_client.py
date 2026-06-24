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


def test_alias_to_unregistered_provider_is_lenient(tmp_path):
    """共享配置常含本消费者没 key 的别名 → 装配不应崩;请求到该别名时才在调用层报错。"""
    import pytest

    from agent_ctl.errors import GatewayError

    cfg = Config(
        routes={"default": ["fake/m"]},  # 必经路由有效
        model_aliases={
            "gpt": "openai/gpt"
        },  # 别名指向未注册 provider —— 可选,不该 fail
        cache_enabled=False,
        db_path=str(tmp_path / "c.db"),
        retry=RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0),
    )
    # 装配通过(别名宽松)
    client = GatewayClient.from_config(cfg, providers={"fake": FakeProvider(["ok"])})
    # 用有效路由仍正常
    assert (
        client.messages("default", [{"role": "user", "content": "hi"}]).text
        == "fake-ok"
    )
    # 但真去请求那个未注册别名 → 调用层报 GatewayError
    with pytest.raises(GatewayError, match="openai"):
        client.messages("gpt", [{"role": "user", "content": "hi"}])
