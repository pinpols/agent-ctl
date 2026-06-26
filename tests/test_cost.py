from agent_ctl.core.cost import CostMeter
from agent_ctl.errors import UnknownPriceError


def test_cost_known_model():
    m = CostMeter({"opus": (5.0, 25.0)})  # $/1M
    # 1000 in, 500 out → 1000/1e6*5 + 500/1e6*25 = 0.005 + 0.0125
    assert m.cost("opus", 1000, 500) == 0.0175


def test_cost_unknown_model_returns_none():
    m = CostMeter({})
    assert m.cost("mystery", 1000, 500) is None


def test_cost_prefers_provider_qualified_price():
    m = CostMeter({"openai/gpt": (1.0, 2.0), "gpt": (9.0, 9.0)})
    assert m.cost("openai/gpt", 1000, 500) == 0.002


def test_cost_falls_back_to_bare_model_price():
    m = CostMeter({"gpt": (1.0, 2.0)})
    assert m.cost("openai/gpt", 1000, 500) == 0.002


def test_cost_strict_unknown_model_raises():
    m = CostMeter({}, fail_unknown=True)
    try:
        m.cost("mystery", 1000, 500)
    except UnknownPriceError as exc:
        assert "mystery" in str(exc)
    else:
        raise AssertionError("expected UnknownPriceError")
