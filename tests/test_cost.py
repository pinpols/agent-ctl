from agent_ctl.core.cost import CostMeter


def test_cost_known_model():
    m = CostMeter({"opus": (5.0, 25.0)})  # $/1M
    # 1000 in, 500 out → 1000/1e6*5 + 500/1e6*25 = 0.005 + 0.0125
    assert m.cost("opus", 1000, 500) == 0.0175


def test_cost_unknown_model_returns_none():
    m = CostMeter({})
    assert m.cost("mystery", 1000, 500) is None
