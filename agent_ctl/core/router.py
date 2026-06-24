from __future__ import annotations

from agent_ctl.models import Target


class Router:
    """模型名 → 有序目标链。解析顺序:routes(逻辑名,可回退链)→ aliases(裸名→单 target)
    → 含 ``/`` 当 ``provider/model`` 直连 → 否则 KeyError。纯查表,无副作用。
    """

    def __init__(
        self, routes: dict[str, list[str]], aliases: dict[str, str] | None = None
    ) -> None:
        self._routes = {k: [Target.parse(s) for s in v] for k, v in routes.items()}
        self._aliases = {k: Target.parse(v) for k, v in (aliases or {}).items()}

    def resolve(self, model: str) -> list[Target]:
        if model in self._routes:
            return list(self._routes[model])
        if model in self._aliases:
            return [self._aliases[model]]
        if "/" in model:
            return [Target.parse(model)]
        raise KeyError(f"unknown model: {model!r} (not in routes/aliases, no '/')")

    def all_targets(self) -> list[Target]:
        """所有 routes + aliases 里的 Target(不含 '/'-直连,那种运行时才知)。"""
        result: list[Target] = []
        for targets in self._routes.values():
            result.extend(targets)
        result.extend(self._aliases.values())
        return result
