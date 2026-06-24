# tests/test_cache.py
from agentctl.models import NormalizedRequest, NormalizedResponse
from agentctl.core.cache import make_key, MemoryCache

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


def test_cache_get_set_and_miss():
    c = MemoryCache()
    assert c.get("k") is None
    c.set("k", NormalizedResponse(text="cached"), ttl_s=60)
    assert c.get("k").text == "cached"


def test_cache_expiry(monkeypatch):
    import agentctl.core.cache as mod

    now = {"t": 1000.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: now["t"])
    c = mod.MemoryCache()
    c.set("k", NormalizedResponse(text="x"), ttl_s=10)
    now["t"] = 1005.0
    assert c.get("k") is not None
    now["t"] = 1011.0
    assert c.get("k") is None
