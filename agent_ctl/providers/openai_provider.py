# agent_ctl/providers/openai_provider.py
from __future__ import annotations

from agent_ctl.errors import RetriableError, TerminalError
from agent_ctl.models import NormalizedRequest, NormalizedResponse, Target
from agent_ctl.providers.tooltrans import (
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_response_to_anthropic_raw,
)


def classify_status(status: int) -> str:
    """HTTP 状态 → retriable/terminal。429 与 5xx 可重试,其余 4xx 终态。

    与 AnthropicProvider 同语义(HTTP 层映射,provider 无关);三行纯函数,
    各 provider 各持一份以免跨 provider 模块互相 import。
    """
    if status == 429 or status >= 500:
        return "retriable"
    return "terminal"


class OpenAIProvider:
    """把 OpenAI Chat Completions SDK 适配为 Provider 协议。client 可注入便于测试。

    兼容任意 OpenAI 兼容端点(OpenAI / 本地 Ollama·vLLM / 各家兼容网关)——
    只要调用方传入的 client 的 base_url 指向对应服务即可。
    """

    def __init__(self, client) -> None:
        self._client = client

    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse:
        # OpenAI 把 system 当作 messages 里的一条 role=system(Anthropic 是独立 system 参数),
        # 故在此把 NormalizedRequest.system 规整为首条 system 消息。
        messages = list(request.messages)
        if request.system is not None:
            messages = [{"role": "system", "content": request.system}, *messages]
        kwargs = {
            "model": target.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "timeout": timeout,  # openai SDK 支持 per-request 超时
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        # 工具:NormalizedRequest.tools/tool_choice 是规范的 Anthropic 形,翻成 OpenAI function 形再发。
        if request.tools:
            kwargs["tools"] = anthropic_tools_to_openai(request.tools)
        if request.tool_choice is not None:
            kwargs["tool_choice"] = anthropic_tool_choice_to_openai(request.tool_choice)
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is not None and classify_status(status) == "retriable":
                raise RetriableError(str(exc)) from exc
            if status is not None:
                raise TerminalError(str(exc)) from exc
            raise RetriableError(str(exc)) from exc  # 网络/未知 → 可重试

        choice = resp.choices[0]
        text = choice.message.content or ""
        tool_calls = len(getattr(choice.message, "tool_calls", None) or [])
        usage = getattr(resp, "usage", None)
        # raw 统一成 Anthropic 风格 content(text + tool_use 块),让消费者(ops-agent shim)
        # 对任何 provider 都能还原 tool_use —— 这是"工具调用跨 provider 通用"的关键。
        raw = openai_response_to_anthropic_raw(choice, usage)
        return NormalizedResponse(
            text=text,
            finish_reason=getattr(choice, "finish_reason", None),
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            raw=raw,
        )
