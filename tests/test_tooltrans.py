from types import SimpleNamespace

from agent_ctl.providers.tooltrans import (
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_message_to_anthropic_content,
    openai_response_to_anthropic_raw,
)


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
