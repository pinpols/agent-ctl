# agentctl/store/base.py
from __future__ import annotations

from typing import Protocol

from agentctl.models import CallRecord


class CaptureStore(Protocol):
    def save(self, record: CallRecord) -> None: ...
    def list_recent(self, limit: int) -> list[CallRecord]: ...
    def cost_summary(self) -> dict: ...
