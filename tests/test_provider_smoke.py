import os

import pytest

from agent_ctl.client.gateway_client import GatewayClient
from agent_ctl.config import Config, RetryConfig
from agent_ctl.providers.catalog import build_providers


pytestmark = pytest.mark.skipif(
    os.getenv("AGENT_CTL_RUN_PROVIDER_SMOKE") != "1",
    reason="set AGENT_CTL_RUN_PROVIDER_SMOKE=1 and provider API keys to run smoke tests",
)


def test_real_provider_chat_smoke():
    providers = build_providers()
    if not providers:
        pytest.skip("no provider API key configured")
    provider = next(iter(providers))
    model = "claude-3-5-haiku-latest" if provider == "anthropic" else "gpt-4o-mini"
    cfg = Config(
        routes={"smoke": [f"{provider}/{model}"]},
        cache_enabled=False,
        capture_async=False,
        db_path=":memory:",
        retry=RetryConfig(max_attempts_per_target=1, base_backoff_s=0, timeout_s=20),
    )
    resp = GatewayClient.from_config(cfg, providers).messages(
        "smoke",
        [{"role": "user", "content": "Reply with exactly: ok"}],
        max_tokens=8,
        consumer="smoke-test",
    )
    assert resp.text.strip()
