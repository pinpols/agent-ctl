# agent_ctl/store/base.py
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from agent_ctl.models import CallRecord


class CaptureStore(Protocol):
    """捕获存储契约。签名与 SqliteCaptureStore / AsyncCaptureStore 实现对齐——
    据此可对替代实现(如 PgStore)做类型校验。读侧过滤参数为关键字。
    """

    def save(self, record: CallRecord) -> None: ...

    def list_recent(
        self,
        limit: int,
        *,
        consumer: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since: float | None = None,
    ) -> list[CallRecord]: ...

    def iter_all(
        self,
        *,
        consumer: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since: float | None = None,
        ascending: bool = True,
    ) -> Iterator[CallRecord]: ...

    def cost_summary(
        self,
        *,
        consumer: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since: float | None = None,
        group_by: str | None = None,
    ) -> dict: ...

    def close(self) -> None: ...
