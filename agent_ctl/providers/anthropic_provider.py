# agent_ctl/providers/anthropic_provider.py
from __future__ import annotations

from collections.abc import Iterator

from agent_ctl.models import (
    NormalizedRequest,
    NormalizedResponse,
    StreamChunk,
    Target,
)
from agent_ctl.providers._http import classify_status, typed_error

# 兼容旧 import 点(测试/外部)：classify_status 经本模块再导出。
__all__ = ["AnthropicProvider", "classify_status"]
_typed_error = typed_error


class AnthropicProvider:
    """把 anthropic SDK 适配为 Provider 协议。client 可注入便于测试。"""

    def __init__(self, client) -> None:
        self._client = client

    def _message_kwargs(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> dict:
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
        return kwargs

    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse:
        try:
            msg = self._client.messages.create(
                **self._message_kwargs(target, request, timeout)
            )
        except Exception as exc:
            raise _typed_error(exc) from exc

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

    def stream(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> Iterator[StreamChunk]:
        """原生流式:解析 Anthropic SSE 事件(message_start/content_block_delta/
        message_delta)→ StreamChunk。input_tokens 来自 message_start,output_tokens
        与 stop_reason 来自 message_delta。连接前异常可被网关在开流前回退。"""
        try:
            events = self._client.messages.create(
                stream=True, **self._message_kwargs(target, request, timeout)
            )
        except Exception as exc:
            raise _typed_error(exc) from exc

        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0
        # 工具调用:content_block_start 给 id/name,input_json_delta 分片拼 arguments(按 index)。
        tool_frags: dict[int, dict] = {}
        for ev in events:
            etype = getattr(ev, "type", None)
            if etype == "message_start":
                usage = getattr(getattr(ev, "message", None), "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) or 0
            elif etype == "content_block_start":
                block = getattr(ev, "content_block", None)
                if getattr(block, "type", None) == "tool_use":
                    tool_frags[getattr(ev, "index", 0) or 0] = {
                        "id": getattr(block, "id", "") or "",
                        "name": getattr(block, "name", "") or "",
                        "args": "",
                    }
            elif etype == "content_block_delta":
                delta = getattr(ev, "delta", None)
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    text = getattr(delta, "text", "") or ""
                    if text:
                        yield StreamChunk(text=text)
                elif dtype == "input_json_delta":
                    frag = tool_frags.get(getattr(ev, "index", 0) or 0)
                    if frag is not None:
                        frag["args"] += getattr(delta, "partial_json", "") or ""
            elif etype == "message_delta":
                delta = getattr(ev, "delta", None)
                fr = getattr(delta, "stop_reason", None)
                if fr:
                    finish_reason = fr
                usage = getattr(ev, "usage", None)
                if usage:
                    output_tokens = getattr(usage, "output_tokens", 0) or output_tokens
        tool_calls = [
            {"id": f["id"], "name": f["name"], "arguments": f["args"]}
            for _, f in sorted(tool_frags.items())
        ] or None
        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
        )
