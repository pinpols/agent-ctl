# agent_ctl/providers/_http.py
"""HTTP 状态 → 类型化网关错误的共享映射(provider 无关)。

此前 openai/anthropic provider 各持一份相同实现(为避免互相 import)。集中到这个不依赖
任何具体 provider 的小模块,两边 import,消除必须手工同步的复制粘贴。
"""

from __future__ import annotations

from agent_ctl.errors import RetriableError, TerminalError


def classify_status(status: int) -> str:
    """HTTP 状态 → retriable/terminal。429 与 5xx 可重试,其余 4xx 终态。"""
    if status == 429 or status >= 500:
        return "retriable"
    return "terminal"


def typed_error(exc: Exception) -> Exception:
    """SDK 异常 → 类型化网关错误。有状态码按 4xx/5xx 分类,无状态码(网络)按可重试。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        return RetriableError(str(exc))  # 网络/未知 → 可重试
    if classify_status(status) == "retriable":
        return RetriableError(str(exc))
    return TerminalError(str(exc))
