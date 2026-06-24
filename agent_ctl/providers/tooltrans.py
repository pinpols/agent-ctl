# agent_ctl/providers/tooltrans.py
"""跨 provider 工具格式归一。

网关内部以 **Anthropic 风格** 作为工具的规范表示(NormalizedRequest.tools / tool_choice,
以及 NormalizedResponse.raw 的 content 块)——因为库形态消费者(ops-agent)本就发 Anthropic 形,
AnthropicProvider 原生即用。OpenAI 家族 provider 在边界处来回翻译:

  请求:Anthropic tools/tool_choice → OpenAI function 形(发给 DeepSeek/OpenAI/通义/GLM)。
  响应:OpenAI message(content + tool_calls)→ Anthropic 风格 content 块([{type:text},{type:tool_use}]),
       塞进 NormalizedResponse.raw,使 ops-agent 的 reconstruct_response 对任何 provider 都成立。
"""

from __future__ import annotations

import json
from typing import Any


def anthropic_tools_to_openai(tools: list | None) -> list | None:
    """Anthropic [{name, description, input_schema}] → OpenAI [{type:function, function:{...}}]。"""
    if not tools:
        return None
    out = []
    for t in tools:
        # 已是 OpenAI 形(含 'function')则原样透传,避免重复翻译。
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            out.append(t)
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def anthropic_tool_choice_to_openai(tc: dict | str | None) -> Any:
    """Anthropic tool_choice → OpenAI。{type:tool,name} → 指定 function;any→required;auto→auto。"""
    if tc is None or isinstance(tc, str):
        return tc
    kind = tc.get("type")
    if kind == "tool":
        return {"type": "function", "function": {"name": tc["name"]}}
    if kind == "any":
        return "required"
    return "auto"


def _finish_to_stop_reason(finish_reason: str | None) -> str | None:
    return {
        "tool_calls": "tool_use",
        "stop": "end_turn",
        "length": "max_tokens",
    }.get(finish_reason or "", finish_reason)


def openai_message_to_anthropic_content(message: Any) -> list[dict]:
    """OpenAI choice.message → Anthropic 风格 content 块。含 tool_calls 时产 tool_use 块(带 input)。"""
    blocks: list[dict] = []
    text = getattr(message, "content", None)
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        raw_args = getattr(fn, "arguments", "") if fn else ""
        try:
            args = (
                json.loads(raw_args)
                if isinstance(raw_args, str) and raw_args
                else (raw_args or {})
            )
        except (json.JSONDecodeError, TypeError):
            args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": getattr(tc, "id", ""),
                "name": name,
                "input": args,
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def anthropic_messages_to_openai(messages: list[dict]) -> list[dict]:
    """多轮工具循环的消息翻译:Anthropic 形 messages → OpenAI 形。

    - content 为 str:原样透传。
    - assistant 的 content 块:text → content 字符串;tool_use → OpenAI tool_calls。
    - user 的 content 块:tool_result → 独立的 {role:tool, tool_call_id, content} 消息;text → user 文本。

    使工具调用型 agent 的**多步循环**(把工具结果发回模型)也能在 OpenAI 家族 provider 上跑。
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, list):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
            tool_calls = [
                {
                    "id": b.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                    },
                }
                for b in content
                if b.get("type") == "tool_use"
            ]
            msg: dict = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:
            # user/tool:tool_result 拆成 role=tool 消息;其余文本并成 user 消息。
            text_parts = []
            for b in content:
                if b.get("type") == "tool_result":
                    tc = b.get("content")
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": b.get("tool_use_id", ""),
                            "content": tc
                            if isinstance(tc, str)
                            else json.dumps(tc, ensure_ascii=False),
                        }
                    )
                elif b.get("type") == "text":
                    text_parts.append(b.get("text", ""))
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
    return out


def openai_response_to_anthropic_raw(choice: Any, usage: Any) -> dict:
    """组装 Anthropic 风格 raw(content + stop_reason + usage),供 ops-agent shim 直接还原。"""
    return {
        "content": openai_message_to_anthropic_content(choice.message),
        "stop_reason": _finish_to_stop_reason(getattr(choice, "finish_reason", None)),
        "usage": {
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        },
    }
