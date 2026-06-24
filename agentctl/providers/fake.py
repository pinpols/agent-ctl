from __future__ import annotations

from agentctl.errors import RetriableError, TerminalError
from agentctl.models import NormalizedRequest, NormalizedResponse, Target


class FakeProvider:
    """离线测试用:按脚本逐次产出 ok/retriable/terminal/timeout。"""

    def __init__(self, script: list[str] | None = None) -> None:
        self._script = list(script or ["ok"])
        self._i = 0
        self.calls: list[Target] = []

    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse:
        self.calls.append(target)
        action = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        if action == "ok":
            return NormalizedResponse(
                text="fake-ok",
                finish_reason="end_turn",
                input_tokens=10,
                output_tokens=5,
            )
        if action == "retriable":
            raise RetriableError("fake retriable")
        if action == "terminal":
            raise TerminalError("fake terminal")
        if action == "timeout":
            raise TimeoutError("fake timeout")
        raise ValueError(f"bad script action: {action}")
