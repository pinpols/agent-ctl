from __future__ import annotations

from typing import Protocol

from agent_ctl.models import NormalizedRequest, NormalizedResponse, Target


class Provider(Protocol):
    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse: ...
