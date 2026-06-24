import pytest
from agent_ctl.core.router import Router


def test_resolve_returns_ordered_targets():
    r = Router({"default": ["anthropic/opus", "anthropic/sonnet"]})
    targets = r.resolve("default")
    assert [t.name for t in targets] == ["anthropic/opus", "anthropic/sonnet"]


def test_resolve_unknown_logical_raises():
    r = Router({"default": ["anthropic/opus"]})
    with pytest.raises(KeyError):
        r.resolve("nope")


def test_resolve_alias_bare_name():
    r = Router({}, aliases={"deepseek-chat": "deepseek/deepseek-chat"})
    targets = r.resolve("deepseek-chat")
    assert [t.name for t in targets] == ["deepseek/deepseek-chat"]


def test_resolve_provider_slash_model_direct():
    r = Router({})
    targets = r.resolve("glm/glm-4")
    assert targets[0].provider == "glm"
    assert targets[0].model == "glm-4"


def test_resolve_order_routes_then_alias_then_direct():
    r = Router({"x": ["anthropic/a"]}, aliases={"y": "openai/b"})
    assert r.resolve("x")[0].name == "anthropic/a"  # routes 优先
    assert r.resolve("y")[0].name == "openai/b"  # 再 alias
    assert r.resolve("qwen/z")[0].name == "qwen/z"  # 再 '/'-直连
    with pytest.raises(KeyError):
        r.resolve("bare-unknown")


def test_all_targets_includes_aliases():
    r = Router({"x": ["anthropic/a"]}, aliases={"y": "openai/b"})
    names = sorted(t.name for t in r.all_targets())
    assert names == ["anthropic/a", "openai/b"]
