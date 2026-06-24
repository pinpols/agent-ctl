from __future__ import annotations

from agent_ctl.models import Target


class Router:
    """逻辑模型名 → 有序目标链。纯查表,无副作用。"""

    def __init__(self, routes: dict[str, list[str]]) -> None:
        self._routes = {k: [Target.parse(s) for s in v] for k, v in routes.items()}

    def resolve(self, logical: str) -> list[Target]:
        if logical not in self._routes:
            raise KeyError(f"unknown logical model: {logical!r}")
        return list(self._routes[logical])

    def all_targets(self) -> list[Target]:
        """返回所有路由中的每个 Target(不去重)。"""
        result: list[Target] = []
        for targets in self._routes.values():
            result.extend(targets)
        return result
