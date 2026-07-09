from types import SimpleNamespace

import pytest

from agent_ctl.errors import TerminalError
from agent_ctl.providers.tooltrans import (
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_message_to_anthropic_content,
    openai_messages_to_anthropic,
    openai_response_to_anthropic_raw,
    openai_tool_choice_to_anthropic,
    openai_tools_to_anthropic,
    stop_reason_to_finish,
)


def test_multiturn_messages_anthropic_to_openai():
    """多轮工具循环:assistant tool_use → tool_calls;user tool_result → role=tool 消息。"""
    msgs = [
        {"role": "user", "content": "查一下 PG 状态"},  # 纯字符串透传
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我来查"},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "query_pg",
                    "input": {"sql": "SELECT 1"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok 1 row"}
            ],
        },
    ]
    out = anthropic_messages_to_openai(msgs)
    assert out[0] == {"role": "user", "content": "查一下 PG 状态"}
    # assistant 带 tool_calls
    assert out[1]["role"] == "assistant"
    assert out[1]["tool_calls"][0]["function"]["name"] == "query_pg"
    assert '"sql"' in out[1]["tool_calls"][0]["function"]["arguments"]
    # tool_result → role=tool,带 tool_call_id
    assert out[2] == {"role": "tool", "tool_call_id": "tu_1", "content": "ok 1 row"}


def test_anthropic_tools_to_openai():
    tools = [
        {
            "name": "diagnose",
            "description": "结构化诊断",
            "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
    ]
    out = anthropic_tools_to_openai(tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "diagnose"
    assert out[0]["function"]["parameters"]["properties"]["x"]["type"] == "string"


def test_anthropic_tools_passthrough_if_already_openai():
    already = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    assert anthropic_tools_to_openai(already) == already


def test_tool_choice_translation():
    assert anthropic_tool_choice_to_openai({"type": "tool", "name": "d"}) == {
        "type": "function",
        "function": {"name": "d"},
    }
    assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"
    assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"
    assert anthropic_tool_choice_to_openai(None) is None


def test_openai_message_tool_calls_to_anthropic_tool_use():
    msg = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="diagnose", arguments='{"root_cause":"OOM"}'
                ),
            )
        ],
    )
    blocks = openai_message_to_anthropic_content(msg)
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["id"] == "call_1"
    assert blocks[0]["name"] == "diagnose"
    assert blocks[0]["input"] == {"root_cause": "OOM"}  # arguments 字符串被解析成 dict


def test_openai_response_to_anthropic_raw_shape():
    choice = SimpleNamespace(
        message=SimpleNamespace(content="hi", tool_calls=None),
        finish_reason="stop",
    )
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    raw = openai_response_to_anthropic_raw(choice, usage)
    assert raw["content"][0] == {"type": "text", "text": "hi"}
    assert raw["stop_reason"] == "end_turn"  # OpenAI 'stop' → Anthropic 'end_turn'
    assert raw["usage"] == {"input_tokens": 7, "output_tokens": 3}


# ── 深审 round4:openai→anthropic 请求方向 + stop_reason 反向映射 + 多模态显式拒绝 ──


def test_openai_tools_to_anthropic():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "diagnose",
                "description": "d",
                "parameters": {"type": "object", "properties": {"x": {}}},
            },
        }
    ]
    assert openai_tools_to_anthropic(tools) == [
        {
            "name": "diagnose",
            "description": "d",
            "input_schema": {"type": "object", "properties": {"x": {}}},
        }
    ]


def test_openai_tools_to_anthropic_passthrough_if_already_anthropic():
    tools = [{"name": "t", "input_schema": {"type": "object"}}]
    assert openai_tools_to_anthropic(tools) == tools


def test_openai_tool_choice_to_anthropic():
    assert openai_tool_choice_to_anthropic(None) is None
    assert openai_tool_choice_to_anthropic("auto") == {"type": "auto"}
    assert openai_tool_choice_to_anthropic("required") == {"type": "any"}
    assert openai_tool_choice_to_anthropic("none") == {"type": "none"}
    assert openai_tool_choice_to_anthropic(
        {"type": "function", "function": {"name": "f"}}
    ) == {"type": "tool", "name": "f"}
    # 已是 Anthropic 形 → 透传
    assert openai_tool_choice_to_anthropic({"type": "tool", "name": "f"}) == {
        "type": "tool",
        "name": "f",
    }


def test_openai_messages_to_anthropic_tool_loop():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"x": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result-1"},
        {"role": "tool", "tool_call_id": "c2", "content": "result-2"},
        {"role": "user", "content": "continue"},
    ]
    out = openai_messages_to_anthropic(msgs)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == [
        {"type": "tool_use", "id": "c1", "name": "f", "input": {"x": 1}}
    ]
    # 连续两条 role=tool 合并成一条 user 消息(Anthropic 角色交替约束)
    assert out[2]["role"] == "user"
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["c1", "c2"]
    assert all(b["type"] == "tool_result" for b in out[2]["content"])
    assert out[3] == {"role": "user", "content": "continue"}


def test_stop_reason_to_finish_mapping():
    assert stop_reason_to_finish("end_turn") == "stop"
    assert stop_reason_to_finish("stop_sequence") == "stop"
    assert stop_reason_to_finish("max_tokens") == "length"
    assert stop_reason_to_finish("tool_use") == "tool_calls"
    # 已是 OpenAI 值 / 未知 / None → 原样透传
    assert stop_reason_to_finish("stop") == "stop"
    assert stop_reason_to_finish("length") == "length"
    assert stop_reason_to_finish(None) is None


def test_image_url_block_rejected_openai_direction():
    """P2-9:多模态 image_url 块不再静默丢弃,显式终态错(→ HTTP 400)。"""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看这张图"},
                {"type": "image_url", "image_url": {"url": "https://x/1.png"}},
            ],
        }
    ]
    with pytest.raises(TerminalError, match="image_url"):
        openai_messages_to_anthropic(msgs)


def test_image_block_rejected_anthropic_direction():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
        }
    ]
    with pytest.raises(TerminalError, match="image"):
        anthropic_messages_to_openai(msgs)
