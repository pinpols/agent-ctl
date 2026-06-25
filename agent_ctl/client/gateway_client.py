# agent_ctl/client/gateway_client.py
from __future__ import annotations

from agent_ctl.config import Config
from agent_ctl.core.cache import MemoryCache
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.models import NormalizedRequest, NormalizedResponse
from agent_ctl.models import Target
from agent_ctl.providers.base import Provider
from agent_ctl.store.sqlite_store import SqliteCaptureStore


def validate_routes(
    routes: dict[str, list[str]],
    providers: dict[str, Provider],
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """**routes** 的每个目标 provider 必须已注册(返回问题列表,空=通过)。

    **aliases 是可选项,不在此 fail**:共享配置常列出多家别名,而某消费者只持部分 provider 的
    key——别名引用未注册 provider 只代表"此处不可用",请求到它时才在调用层报错,而非启动崩溃。
    `aliases` 参数保留仅为签名兼容(忽略)。
    """
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
        problems = validate_routes(config.routes, providers, config.model_aliases)
        if problems:
            raise ValueError("路由配置校验失败:\n  - " + "\n  - ".join(problems))
        gateway = Gateway(
            router=Router(config.routes, config.model_aliases),
            providers=providers,
            cost_meter=CostMeter(config.prices),
            store=SqliteCaptureStore(config.db_path),
            cache=MemoryCache() if config.cache_enabled else None,
            retry=config.retry,
            cache_enabled=config.cache_enabled,
            cache_ttl_s=config.cache_ttl_s,
            cache_tool_responses=config.cache_tool_responses,
            circuit=CircuitBreaker(
                config.circuit_failure_threshold, config.circuit_cooldown_s
            ),
        )
        return cls(gateway)

    def messages(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float | None = None,
        tools: list | None = None,
        system: str | None = None,
        tool_choice: dict | None = None,
        **metadata,
    ) -> NormalizedResponse:
        request = NormalizedRequest(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            system=system,
            tool_choice=tool_choice,
            metadata=metadata,
        )
        return self._gateway.invoke(request)
