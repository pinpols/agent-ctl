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

from agent_ctl.errors import TerminalError

# 多模态语义:目标 = 不静默丢弃且尽量直通。
#   图像:两个方向都支持——同形直通,跨形转换(image_url ↔ Anthropic image,
#         base64 data URI ↔ base64 source,http(s) URL ↔ url source)。
#   音频(input_audio):Anthropic 不支持,且本网关不做音频转码 → 仍显式终态错。
# 本地可判定的拒绝(不打网络)统一走 validate_local_content,在网关入口/HTTP 边界
# 执行,不进 provider 调用路径 → 不污染熔断记账。
_UNSUPPORTED_BLOCK_TYPES = {"input_audio"}

_TOOL_CHOICE_STRINGS = {"auto", "required", "none"}


def _reject_unsupported(block_type: str | None) -> None:
    if block_type in _UNSUPPORTED_BLOCK_TYPES:
        raise TerminalError(
            f"multimodal content block {block_type!r} is not supported by this gateway"
        )


def _image_url_of(block: dict) -> str:
    iu = block.get("image_url")
    if isinstance(iu, dict):
        return iu.get("url") or ""
    return iu if isinstance(iu, str) else ""


def _check_image_url(url: str) -> None:
    """data: URI 只支持 base64 编码(两方向的转换都要 base64 体)。"""
    if url.startswith("data:"):
        head, sep, _ = url.partition(",")
        if not sep or "base64" not in head:
            raise TerminalError(
                "image_url data URI must be base64-encoded (data:<mime>;base64,...)"
            )


def _openai_image_url_to_anthropic(block: dict) -> dict:
    """OpenAI image_url 块 → Anthropic image 块(data: → base64 source;http → url source)。"""
    url = _image_url_of(block)
    _check_image_url(url)
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        media_type = head[5:].split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _anthropic_image_to_openai(block: dict) -> dict:
    """Anthropic image 块 → OpenAI image_url 块(base64 source → data: URI;url source 直通)。"""
    source = block.get("source") or {}
    if source.get("type") == "base64":
        media_type = source.get("media_type") or "image/png"
        url = f"data:{media_type};base64,{source.get('data', '')}"
    else:
        url = source.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def validate_local_content(
    messages: list[dict], tool_choice: dict | str | None = None
) -> None:
    """本地可判定的终态校验(不打任何网络):不支持的多模态块 / 非法 tool_choice 字符串 /
    非 base64 的 image_url data URI。

    在网关入口(库形态)与 HTTP server 的 to_normalized 边界各执行一次,使这类拒绝
    发生在 provider 调用路径之外——否则 5 个坏请求就能把任意 provider 的熔断打开
    (拒绝发生在任何网络调用前,却被记成 provider 失败)。
    """
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            _reject_unsupported(b.get("type"))
            if b.get("type") == "image_url":
                _check_image_url(_image_url_of(b))
    if isinstance(tool_choice, str) and tool_choice not in _TOOL_CHOICE_STRINGS:
        raise TerminalError(f"unsupported tool_choice string: {tool_choice!r}")


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
            # user/tool:tool_result 拆成 role=tool 消息;文本/图像并成 user 消息
            # (纯文本保持字符串形;含图像则用 OpenAI content 数组形)。
            parts: list[dict] = []
            for b in content:
                btype = b.get("type")
                _reject_unsupported(btype)
                if btype == "tool_result":
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
                elif btype == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif btype == "image":
                    # Anthropic 原生 image 块 → OpenAI image_url(OpenAI 家族也支持视觉)
                    parts.append(_anthropic_image_to_openai(b))
                elif btype == "image_url":
                    # 已是 OpenAI 形 → 直通
                    parts.append(b)
            if parts:
                if all(p.get("type") == "text" for p in parts):
                    out.append(
                        {"role": "user", "content": "".join(p["text"] for p in parts)}
                    )
                else:
                    out.append({"role": "user", "content": parts})
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


# ── OpenAI 形 → Anthropic 形(HTTP server 收 OpenAI 请求、路由到 Anthropic 后端时用)──


def openai_tools_to_anthropic(tools: list | None) -> list | None:
    """OpenAI [{type:function, function:{name,description,parameters}}] →
    Anthropic [{name, description, input_schema}]。已是 Anthropic 形则原样透传。"""
    if not tools:
        return None
    out = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            fn = t["function"] or {}
            out.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        else:
            out.append(t)
    return out


def openai_tool_choice_to_anthropic(tc: dict | str | None) -> dict | None:
    """OpenAI tool_choice → Anthropic。"auto"→{type:auto};"required"→{type:any};
    "none"→{type:none};{type:function,function:{name}}→{type:tool,name}。
    已是 Anthropic 形 dict 则原样透传。"""
    if tc is None:
        return None
    if isinstance(tc, str):
        mapped = {"auto": "auto", "required": "any", "none": "none"}.get(tc)
        if mapped is None:
            raise TerminalError(f"unsupported tool_choice string: {tc!r}")
        return {"type": mapped}
    if tc.get("type") == "function":
        return {"type": "tool", "name": (tc.get("function") or {}).get("name", "")}
    return tc


def _openai_content_to_anthropic_blocks(content: list) -> list:
    """OpenAI 形 content 块数组 → Anthropic 形:image_url 转 image,原生块直通,音频报错。"""
    blocks = []
    for b in content:
        if isinstance(b, dict):
            _reject_unsupported(b.get("type"))
            if b.get("type") == "image_url":
                blocks.append(_openai_image_url_to_anthropic(b))
                continue
        blocks.append(b)
    return blocks


def openai_messages_to_anthropic(messages: list[dict]) -> list[dict]:
    """OpenAI 形 messages → Anthropic 形(工具循环的请求方向翻译)。

    - role=tool → user 消息里的 tool_result 块(连续多条 tool 合并为一条 user 消息,
      满足 Anthropic 角色交替约束)。
    - assistant.tool_calls → assistant content 的 tool_use 块(arguments JSON 解析为 input)。
    - content 块:image_url → Anthropic image(data:base64 → base64 source,http → url source);
      原生 image 块直通;音频显式报错(见 _reject_unsupported)。其余消息原样透传。
    """
    out: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": content
                    if isinstance(content, str)
                    else json.dumps(content, ensure_ascii=False),
                }
            )
            continue
        if isinstance(content, list):
            content = _openai_content_to_anthropic_blocks(content)
            m = {**m, "content": content}
        if role == "user" and pending_tool_results:
            # 紧随 tool_result 的 user 内容并入同一条 user 消息(tool_result 块在前):
            # 否则产出连续两条 user,违反 Anthropic 角色交替约束被 400。
            blocks = list(pending_tool_results)
            pending_tool_results.clear()
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend(content)
            out.append({"role": "user", "content": blocks})
            continue
        flush_tool_results()
        if role == "assistant" and m.get("tool_calls"):
            blocks: list[dict] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "")
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append(m)
    flush_tool_results()
    return out


def stop_reason_to_finish(stop_reason: str | None) -> str | None:
    """Anthropic stop_reason → OpenAI finish_reason(响应方向反向映射)。

    end_turn/stop_sequence→stop、max_tokens→length、tool_use→tool_calls;
    已是 OpenAI 值(或未知)则原样透传。"""
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason or "", stop_reason)
