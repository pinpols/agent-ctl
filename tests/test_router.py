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
