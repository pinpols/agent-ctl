from __future__ import annotations

import logging

from agent_ctl.errors import UnknownPriceError

log = logging.getLogger("agent_ctl.cost")


class CostMeter:
    """按价表(每 1M token 美元)算调用成本;未知模型返回 None。"""

    def __init__(
        self, prices: dict[str, tuple[float, float]], *, fail_unknown: bool = False
    ) -> None:
        self._prices = prices
        self._fail_unknown = fail_unknown

    def _price_for(self, model: str) -> tuple[float, float] | None:
        price = self._prices.get(model)
        if price is None and "/" in model:
            price = self._prices.get(model.split("/", 1)[1])
        return price

    def ensure_price(self, model: str) -> None:
        if self._fail_unknown and self._price_for(model) is None:
            raise UnknownPriceError(f"unknown model for pricing: {model}")

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self._price_for(model)
        if price is None:
            if self._fail_unknown:
                raise UnknownPriceError(f"unknown model for pricing: {model}")
            log.warning("unknown model for pricing: %s (cost=None)", model)
            return None
        in_price, out_price = price
        return round(
            input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price,
            6,
        )
