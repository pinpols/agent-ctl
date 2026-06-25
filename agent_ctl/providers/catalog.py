# agent_ctl/providers/catalog.py
"""5 家 provider 的静态目录 + 从环境构造 providers 注册表。

只有 Claude 走原生 Anthropic SDK;openai/deepseek/qwen/glm 都是 OpenAI 兼容,
共用 OpenAIProvider,仅 base_url + api key 不同。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# provider 名 → {kind, key_env, base_url}
PROVIDER_CATALOG: dict[str, dict] = {
    "anthropic": {
        "kind": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
    "openai": {"kind": "openai", "key_env": "OPENAI_API_KEY", "base_url": None},
    "deepseek": {
        "kind": "openai",
        "key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    },
    "qwen": {
        "kind": "openai",
        "key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "glm": {
        "kind": "openai",
        "key_env": "GLM_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
}


def provider_capabilities(name: str) -> set[str]:
    """某 provider 的能力集(静态,按适配器类型推导,无需 api key / 不构造实例)。

    chat/tools 所有适配器都支持;stream 由原生流式适配器支持;embed 仅 OpenAI 兼容家族
    (Anthropic 无 embeddings API)。doctor 据此提示"路由目标能不能干这件事"。
    """
    kind = PROVIDER_CATALOG.get(name, {}).get("kind")
    if kind not in ("anthropic", "openai"):
        return set()  # 无内建适配器
    caps = {"chat", "tools", "stream"}
    if kind == "openai":
        caps.add("embed")
    return caps


def available_providers(env: Mapping[str, str] | None = None) -> list[str]:
    """目录中 api key 已在环境里设置的 provider 名(按目录顺序)。纯函数,无 SDK 依赖。"""
    env_map = env if env is not None else os.environ
    return [
        name
        for name, c in PROVIDER_CATALOG.items()
        if env_map.get(c["key_env"], "").strip()
    ]


# SDK 构造抽出为可 monkeypatch 的小函数(单测不依赖真 SDK 安装)。
def _make_anthropic(api_key: str):
    import anthropic  # type: ignore[import-not-found]

    from agent_ctl.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(anthropic.Anthropic(api_key=api_key))


def _make_openai(api_key: str, base_url: str | None):
    from openai import OpenAI  # type: ignore[import-not-found]

    from agent_ctl.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(OpenAI(api_key=api_key, base_url=base_url))


def build_providers(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """按目录构造 providers 注册表;仅纳入 api key 存在的 provider。"""
    env_map = env if env is not None else os.environ
    providers: dict[str, Any] = {}
    for name in available_providers(env_map):
        c = PROVIDER_CATALOG[name]
        key = env_map[c["key_env"]]
        if c["kind"] == "anthropic":
            providers[name] = _make_anthropic(key)
        else:
            providers[name] = _make_openai(key, c["base_url"])
    return providers
