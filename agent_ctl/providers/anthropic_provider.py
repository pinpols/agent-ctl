# agent_ctl/providers/anthropic_provider.py
from __future__ import annotations

from agent_ctl.errors import RetriableError, TerminalError
from agent_ctl.models import NormalizedRequest, NormalizedResponse, Target


def classify_status(status: int) -> str:
    """HTTP 状态 → retriable/terminal。429 与 5xx 可重试,其余 4xx 终态。"""
    if status == 429 or status >= 500:
        return "retriable"
    return "terminal"


class AnthropicProvider:
    """把 anthropic SDK 适配为 Provider 协议。client 可注入便于测试。"""

    def __init__(self, client) -> None:
        self._client = client

    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse:
        try:
            kwargs = {
                "model": target.model,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "timeout": timeout,  # anthropic SDK 支持 per-request 墙钟超时(httpx)
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.tools:
                kwargs["tools"] = request.tools
            # system 提示 + 强制工具选择:工具调用型消费者依赖,缺省不传(向后兼容纯文本路由)。
            if request.system is not None:
                kwargs["system"] = request.system
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice
            msg = self._client.messages.create(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is not None and classify_status(status) == "retriable":
                raise RetriableError(str(exc)) from exc
            if status is not None:
                raise TerminalError(str(exc)) from exc
            raise RetriableError(str(exc)) from exc  # 网络/未知 → 可重试

        text = "".join(
            getattr(b, "text", "")
            for b in msg.content
            if getattr(b, "type", None) == "text"
        )
        tool_calls = sum(
            1 for b in msg.content if getattr(b, "type", None) == "tool_use"
        )
        # raw=完整结构化响应(含 tool_use 块的 input):工具调用型消费者据此还原原生响应。
        # 纯文本消费者忽略它即可,零成本。SDK Message 是 pydantic → model_dump;无该方法则 None。
        raw = msg.model_dump(mode="json") if hasattr(msg, "model_dump") else None
        return NormalizedResponse(
            text=text,
            finish_reason=getattr(msg, "stop_reason", None),
            tool_calls=tool_calls,
            input_tokens=getattr(msg.usage, "input_tokens", 0),
            output_tokens=getattr(msg.usage, "output_tokens", 0),
            raw=raw,
        )
