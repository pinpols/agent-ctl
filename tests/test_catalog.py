from agent_ctl.providers import catalog


def test_available_providers_only_those_with_keys():
    env = {"ANTHROPIC_API_KEY": "x", "DEEPSEEK_API_KEY": "y", "OPENAI_API_KEY": "  "}
    avail = catalog.available_providers(env)
    assert "anthropic" in avail
    assert "deepseek" in avail
    assert "openai" not in avail  # 空白 key 不算
    assert "qwen" not in avail and "glm" not in avail


def test_catalog_covers_five_providers():
    assert set(catalog.PROVIDER_CATALOG) == {
        "anthropic",
        "openai",
        "deepseek",
        "qwen",
        "glm",
    }
    # 4 家 OpenAI 兼容,1 家原生
    kinds = {n: c["kind"] for n, c in catalog.PROVIDER_CATALOG.items()}
    assert kinds["anthropic"] == "anthropic"
    assert all(kinds[p] == "openai" for p in ("openai", "deepseek", "qwen", "glm"))


def test_provider_capabilities():
    assert catalog.provider_capabilities("deepseek") == {
        "chat",
        "stream",
        "tools",
        "embed",
    }
    assert catalog.provider_capabilities("openai") == {
        "chat",
        "stream",
        "tools",
        "embed",
    }
    anthropic_caps = catalog.provider_capabilities("anthropic")
    assert "embed" not in anthropic_caps  # 无 embeddings API
    assert {"chat", "stream", "tools"} <= anthropic_caps
    assert catalog.provider_capabilities("nope") == set()  # 未知 → 空集


def test_build_providers_constructs_only_keyed(monkeypatch):
    # monkeypatch SDK 构造器,避免依赖真 anthropic/openai 安装
    monkeypatch.setattr(
        catalog, "_make_anthropic", lambda key: f"anthropic-client:{key}"
    )
    monkeypatch.setattr(
        catalog, "_make_openai", lambda key, base_url: f"openai-client:{key}@{base_url}"
    )
    env = {"ANTHROPIC_API_KEY": "ak", "GLM_API_KEY": "gk"}
    providers = catalog.build_providers(env)
    assert set(providers) == {"anthropic", "glm"}
    assert providers["anthropic"] == "anthropic-client:ak"
    # glm 走 OpenAIProvider + 智谱 base_url
    assert "open.bigmodel.cn" in providers["glm"]
