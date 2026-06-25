# agent_ctl/providers/openai_provider.py
from __future__ import annotations

from collections.abc import Iterator

from agent_ctl.errors import RetriableError, TerminalError
from agent_ctl.models import (
    EmbeddingResponse,
    NormalizedRequest,
    NormalizedResponse,
    StreamChunk,
    Target,
)
from agent_ctl.providers.tooltrans import (
    anthropic_messages_to_openai,
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


def _typed_error(exc: Exception) -> Exception:
    """SDK 异常 → 类型化网关错误。有状态码按 4xx/5xx 分类,无状态码(网络)按可重试。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        return RetriableError(str(exc))  # 网络/未知 → 可重试
    if classify_status(status) == "retriable":
        return RetriableError(str(exc))
    return TerminalError(str(exc))


class OpenAIProvider:
    """把 OpenAI Chat Completions SDK 适配为 Provider 协议。client 可注入便于测试。

    兼容任意 OpenAI 兼容端点(OpenAI / 本地 Ollama·vLLM / 各家兼容网关)——
    只要调用方传入的 client 的 base_url 指向对应服务即可。
    """

    def __init__(self, client) -> None:
        self._client = client

    def _chat_kwargs(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> dict:
        # 消息:Anthropic 形(多轮工具循环含 tool_use/tool_result 块)→ OpenAI 形。
        # system 是独立字段,规整为首条 role=system 消息。
        messages = anthropic_messages_to_openai(request.messages)
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
        return kwargs

    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse:
        try:
            resp = self._client.chat.completions.create(
                **self._chat_kwargs(target, request, timeout)
            )
        except Exception as exc:
            raise _typed_error(exc) from exc

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

    def stream(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> Iterator[StreamChunk]:
        """原生流式:stream=True + include_usage(末块仅含 usage,choices 为空)。

        连接前异常(create 抛)→ 类型化错误,可被网关在开流前回退;迭代中(已开流)
        异常按原样向上抛,由网关捕获记错(此时已发字节,无法回退)。
        """
        kwargs = self._chat_kwargs(target, request, timeout)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _typed_error(exc) from exc

        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0
        # 工具调用以分片到达(按 index,arguments 跨 chunk 拼接)→ 重组,勿丢。
        tool_frags: dict[int, dict] = {}
        for chunk in resp:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                fr = getattr(choices[0], "finish_reason", None)
                if fr:
                    finish_reason = fr
                if content:
                    yield StreamChunk(text=content)
                for tc in (getattr(delta, "tool_calls", None) or []) if delta else []:
                    idx = getattr(tc, "index", 0) or 0
                    frag = tool_frags.setdefault(
                        idx, {"id": "", "name": "", "args": ""}
                    )
                    if getattr(tc, "id", None):
                        frag["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            frag["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            frag["args"] += fn.arguments
            usage = getattr(chunk, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
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

    def embed(
        self, target: Target, inputs: list[str], timeout: float
    ) -> EmbeddingResponse:
        """OpenAI 兼容 embeddings。input 顺序与返回向量顺序一致(按 data.index 排序保证)。"""
        try:
            resp = self._client.embeddings.create(
                model=target.model, input=inputs, timeout=timeout
            )
        except Exception as exc:
            raise _typed_error(exc) from exc
        data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
        vectors = [list(d.embedding) for d in data]
        usage = getattr(resp, "usage", None)
        return EmbeddingResponse(
            vectors=vectors,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        )
