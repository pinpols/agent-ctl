# tests/test_smoke_real.py
"""真 provider 冒烟测试——默认跳过,需显式开启 + 环境里有 provider key。

  AGENT_CTL_SMOKE=1 DEEPSEEK_API_KEY=... pytest tests/test_smoke_real.py -q

可选覆盖:AGENT_CTL_SMOKE_MODEL(chat 模型名)、AGENT_CTL_SMOKE_EMBED_MODEL(embeddings 模型名)。
低 max_tokens,只验"真链路打得通",不进默认 CI(单测从不联网)。
"""

import os

import pytest

from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.models import NormalizedRequest
from agent_ctl.providers.catalog import (
    available_providers,
    build_providers,
    provider_capabilities,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("AGENT_CTL_SMOKE"),
    reason="real-provider smoke disabled; set AGENT_CTL_SMOKE=1 + a provider api key",
)

_DEFAULT_CHAT = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
    "qwen": "qwen-max",
    "glm": "glm-4",
}


def _first_provider() -> str:
    avail = available_providers()
    if not avail:
        pytest.skip("no provider api key in env")
    return avail[0]


def _chat_gateway():
    name = _first_provider()
    model = os.getenv("AGENT_CTL_SMOKE_MODEL") or _DEFAULT_CHAT.get(name, "")
    gw = Gateway(
        router=Router({"default": [f"{name}/{model}"]}),
        providers=build_providers(),
        cost_meter=CostMeter({}),
        request_deadline_s=30,
    )
    return gw, name


def _req(content: str, max_tokens: int) -> NormalizedRequest:
    return NormalizedRequest(
        model="default",
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        metadata={"consumer": "smoke"},
    )


def test_smoke_chat():
    gw, _ = _chat_gateway()
    resp = gw.invoke(_req("回复一个字:好", 5))
    assert resp.text.strip()
    assert resp.output_tokens > 0


def test_smoke_stream():
    gw, _ = _chat_gateway()
    chunks = list(gw.invoke_stream(_req("从1数到3", 20)))
    assert any(c.text for c in chunks if not c.done)  # 有真实增量
    assert chunks[-1].done  # 末块 done


def test_smoke_embed():
    name = _first_provider()
    if "embed" not in provider_capabilities(name):
        pytest.skip(f"{name} 无 embeddings 能力")
    emb_model = os.getenv("AGENT_CTL_SMOKE_EMBED_MODEL")
    if not emb_model:
        pytest.skip("设 AGENT_CTL_SMOKE_EMBED_MODEL 才测 embeddings")
    gw = Gateway(
        router=Router({"default": [f"{name}/{emb_model}"]}),
        providers=build_providers(),
        cost_meter=CostMeter({}),
    )
    resp = gw.embed("default", ["hello", "world"], {"consumer": "smoke"})
    assert len(resp.vectors) == 2
    assert len(resp.vectors[0]) > 0
