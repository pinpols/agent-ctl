# tests/test_cache.py
from agent_ctl.models import NormalizedRequest, NormalizedResponse
from agent_ctl.core.cache import make_key, MemoryCache

REQ = NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}])


def test_make_key_stable_and_distinct():
    k1 = make_key(REQ)
    k2 = make_key(
        NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}])
    )
    k3 = make_key(
        NormalizedRequest(
            model="default", messages=[{"role": "user", "content": "bye"}]
        )
    )
    assert k1 == k2
    assert k1 != k3


def test_make_key_differs_on_system_tools_tool_choice():
    """system / tools / tool_choice 改变响应形状,必须进 key,否则不同形状请求错误命中同一缓存。"""
    base = NormalizedRequest(
        model="default", messages=[{"role": "user", "content": "hi"}]
    )
    assert make_key(base) != make_key(base.model_copy(update={"system": "你是 SRE"}))
    assert make_key(base) != make_key(
        base.model_copy(update={"tools": [{"name": "x"}]})
    )
    assert make_key(base) != make_key(
        base.model_copy(update={"tool_choice": {"type": "tool", "name": "x"}})
    )


def test_make_key_differs_on_max_tokens():
    """max_tokens 不同应生成不同 key。"""
    k1 = make_key(
        NormalizedRequest(
            model="default", messages=[{"role": "user", "content": "hi"}], max_tokens=64
        )
    )
    k2 = make_key(
        NormalizedRequest(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=128,
        )
    )
    assert k1 != k2


def test_make_key_differs_on_temperature():
    """temperature 不同应生成不同 key。"""
    k1 = make_key(
        NormalizedRequest(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
        )
    )
    k2 = make_key(
        NormalizedRequest(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.9,
        )
    )
    assert k1 != k2


def test_cache_get_set_and_miss():
    c = MemoryCache()
    assert c.get("k") is None
    c.set("k", NormalizedResponse(text="cached"), ttl_s=60)
    assert c.get("k").text == "cached"


def test_cache_expiry(monkeypatch):
    import agent_ctl.core.cache as mod

    now = {"t": 1000.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: now["t"])
    c = mod.MemoryCache()
    c.set("k", NormalizedResponse(text="x"), ttl_s=10)
    now["t"] = 1005.0
    assert c.get("k") is not None
    now["t"] = 1011.0
    assert c.get("k") is None
