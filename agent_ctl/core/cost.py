from __future__ import annotations

import logging

log = logging.getLogger("agent_ctl.cost")


class CostMeter:
    """按价表(每 1M token 美元)算调用成本;未知模型返回 None。"""

    def __init__(self, prices: dict[str, tuple[float, float]]) -> None:
        self._prices = prices

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self._prices.get(model)
        if price is None and "/" in model:
            price = self._prices.get(model.split("/", 1)[1])
        if price is None:
            log.warning("unknown model for pricing: %s (cost=None)", model)
            return None
        in_price, out_price = price
        return round(
            input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price,
            6,
        )
