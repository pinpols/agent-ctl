# agentctl/client/gateway_client.py
from __future__ import annotations

from agentctl.config import Config
from agentctl.core.cache import MemoryCache
from agentctl.core.cost import CostMeter
from agentctl.core.gateway import Gateway
from agentctl.core.router import Router
from agentctl.models import NormalizedRequest, NormalizedResponse
from agentctl.models import Target
from agentctl.providers.base import Provider
from agentctl.store.sqlite_store import SqliteCaptureStore


def validate_routes(
    routes: dict[str, list[str]], providers: dict[str, Provider]
) -> list[str]:
    """每条路由目标的 provider 必须已注册。返回问题列表(空=通过)。"""
    problems: list[str] = []
    for logical, targets in routes.items():
        for spec in targets:
            try:
                target = Target.parse(spec)
            except ValueError as exc:
                problems.append(f"route {logical!r}: {exc}")
                continue
            if target.provider not in providers:
                problems.append(
                    f"route {logical!r} → {spec!r}: provider {target.provider!r} 未注册"
                )
    return problems


class GatewayClient:
    """库形态门面:消费者(如 ops-agent)直接调这个。"""

    def __init__(self, gateway: Gateway) -> None:
        self._gateway = gateway

    @classmethod
    def from_config(
        cls, config: Config, providers: dict[str, Provider]
    ) -> "GatewayClient":
        problems = validate_routes(config.routes, providers)
        if problems:
            raise ValueError("路由配置校验失败:\n  - " + "\n  - ".join(problems))
        gateway = Gateway(
            router=Router(config.routes),
            providers=providers,
            cost_meter=CostMeter(config.prices),
            store=SqliteCaptureStore(config.db_path),
            cache=MemoryCache() if config.cache_enabled else None,
            retry=config.retry,
            cache_enabled=config.cache_enabled,
            cache_ttl_s=config.cache_ttl_s,
        )
        return cls(gateway)

    def messages(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float | None = None,
        tools: list | None = None,
        **metadata,
    ) -> NormalizedResponse:
        request = NormalizedRequest(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            metadata=metadata,
        )
        return self._gateway.invoke(request)
