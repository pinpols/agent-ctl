from agent_ctl.providers.openai_provider import OpenAIProvider, classify_status
from agent_ctl.models import Target, NormalizedRequest
from agent_ctl.errors import RetriableError, TerminalError
import pytest

REQ = NormalizedRequest(
    model="default",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=64,
    system="你是助手",
)
T = Target(provider="openai", model="gpt-4o")


class _FakeCompletions:
    def __init__(self, behavior):
        self._b = behavior
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if isinstance(self._b, Exception):
            raise self._b
        msg = type("M", (), {"content": "hello", "tool_calls": None})()
        choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
        usage = type("U", (), {"prompt_tokens": 7, "completion_tokens": 3})()
        return type("R", (), {"choices": [choice], "usage": usage})()


class _FakeClient:
    def __init__(self, behavior):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(behavior)})()


def test_invoke_maps_response():
    p = OpenAIProvider(_FakeClient("ok"))
    resp = p.invoke(T, REQ, timeout=5.0)
    assert resp.text == "hello"
    assert resp.input_tokens == 7
    assert resp.output_tokens == 3
    assert resp.finish_reason == "stop"


def test_system_prepended_as_message_and_timeout_passed():
    client = _FakeClient("ok")
    OpenAIProvider(client).invoke(T, REQ, timeout=12.5)
    sent = client.chat.completions.last_kwargs
    assert sent["messages"][0] == {"role": "system", "content": "你是助手"}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    assert sent["timeout"] == 12.5


def test_classify_status():
    assert classify_status(429) == "retriable"
    assert classify_status(503) == "retriable"
    assert classify_status(401) == "terminal"
    assert classify_status(400) == "terminal"


def test_status_based_exception_routing():
    err401 = type("E", (Exception,), {"status_code": 401})("auth")
    with pytest.raises(TerminalError):
        OpenAIProvider(_FakeClient(err401)).invoke(T, REQ, timeout=5.0)
    err503 = type("E", (Exception,), {"status_code": 503})("overloaded")
    with pytest.raises(RetriableError):
        OpenAIProvider(_FakeClient(err503)).invoke(T, REQ, timeout=5.0)


class _ToolCallCompletions:
    """返回一个 tool_calls 的假 OpenAI 响应,并记录收到的 kwargs。"""

    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        tc = type(
            "TC",
            (),
            {
                "id": "call_1",
                "function": type(
                    "F", (), {"name": "diagnose", "arguments": '{"root_cause":"OOM"}'}
                )(),
            },
        )()
        msg = type("M", (), {"content": None, "tool_calls": [tc]})()
        choice = type("C", (), {"message": msg, "finish_reason": "tool_calls"})()
        usage = type("U", (), {"prompt_tokens": 12, "completion_tokens": 8})()
        return type("R", (), {"choices": [choice], "usage": usage})()


class _ToolClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _ToolCallCompletions()})()


class _FakeEmbeddings:
    def __init__(self, behavior="ok"):
        self._b = behavior
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if isinstance(self._b, Exception):
            raise self._b
        # 故意乱序返回,验证 provider 按 index 排序还原输入顺序
        d1 = type("D", (), {"index": 1, "embedding": [0.4, 0.5]})()
        d0 = type("D", (), {"index": 0, "embedding": [0.1, 0.2]})()
        usage = type("U", (), {"prompt_tokens": 9})()
        return type("R", (), {"data": [d1, d0], "usage": usage})()


class _EmbedClient:
    def __init__(self, behavior="ok"):
        self.embeddings = _FakeEmbeddings(behavior)


def test_embed_maps_vectors_in_index_order():
    client = _EmbedClient("ok")
    resp = OpenAIProvider(client).embed(
        Target(provider="openai", model="text-embedding-3-small"),
        ["a", "b"],
        timeout=5.0,
    )
    assert resp.vectors == [[0.1, 0.2], [0.4, 0.5]]  # 按 index 排序
    assert resp.input_tokens == 9
    assert client.embeddings.last_kwargs["model"] == "text-embedding-3-small"
    assert client.embeddings.last_kwargs["input"] == ["a", "b"]


def test_embed_status_based_exception_routing():
    err401 = type("E", (Exception,), {"status_code": 401})("auth")
    with pytest.raises(TerminalError):
        OpenAIProvider(_EmbedClient(err401)).embed(T, ["x"], timeout=5.0)
    err503 = type("E", (Exception,), {"status_code": 503})("overloaded")
    with pytest.raises(RetriableError):
        OpenAIProvider(_EmbedClient(err503)).embed(T, ["x"], timeout=5.0)


class _StreamCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        def _delta(content, fr=None):
            ch = type(
                "Ch",
                (),
                {"delta": type("D", (), {"content": content})(), "finish_reason": fr},
            )()
            return type("K", (), {"choices": [ch], "usage": None})()

        def _final():
            usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 4})()
            return type("K", (), {"choices": [], "usage": usage})()

        return iter([_delta("Hel"), _delta("lo", fr="stop"), _final()])


class _StreamClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _StreamCompletions()})()


def test_stream_parses_deltas_and_usage():
    client = _StreamClient()
    chunks = list(OpenAIProvider(client).stream(T, REQ, timeout=5.0))
    assert [c.text for c in chunks if not c.done] == ["Hel", "lo"]
    done = chunks[-1]
    assert done.done and done.finish_reason == "stop"
    assert done.input_tokens == 11 and done.output_tokens == 4
    # 流式请求带 stream + include_usage
    assert client.chat.completions.last_kwargs["stream"] is True
    assert client.chat.completions.last_kwargs["stream_options"] == {
        "include_usage": True
    }


def test_stream_connect_error_is_typed():
    err503 = type("E", (Exception,), {"status_code": 503})("overloaded")

    class _BoomClient:
        def __init__(self):
            comp = type("C", (), {"create": self._boom})()
            self.chat = type("Chat", (), {"completions": comp})()

        def _boom(self, **kwargs):
            raise err503

    with pytest.raises(RetriableError):
        list(OpenAIProvider(_BoomClient()).stream(T, REQ, timeout=5.0))


class _ToolStreamCompletions:
    """流式工具调用:tool_calls 分片到达(id+name 在首片,arguments 跨片拼接)。"""

    def create(self, **kwargs):
        def _tc(index, id=None, name=None, args=None):
            fn = type("F", (), {"name": name, "arguments": args})()
            return type("TC", (), {"index": index, "id": id, "function": fn})()

        def _chunk(tool_calls, fr=None, usage=None):
            delta = type("D", (), {"content": None, "tool_calls": tool_calls})()
            ch = type("Ch", (), {"delta": delta, "finish_reason": fr})()
            return type("K", (), {"choices": [ch], "usage": usage})()

        usage = type("U", (), {"prompt_tokens": 12, "completion_tokens": 8})()
        return iter(
            [
                _chunk([_tc(0, id="call_1", name="diagnose", args='{"root')]),
                _chunk([_tc(0, args='_cause":"OOM"}')], fr="tool_calls"),
                type("K", (), {"choices": [], "usage": usage})(),
            ]
        )


def test_stream_reassembles_fragmented_tool_calls():
    client = type(
        "C", (), {"chat": type("Chat", (), {"completions": _ToolStreamCompletions()})()}
    )()
    chunks = list(OpenAIProvider(client).stream(T, REQ, timeout=5.0))
    done = chunks[-1]
    assert done.done
    assert done.tool_calls == [
        {"id": "call_1", "name": "diagnose", "arguments": '{"root_cause":"OOM"}'}
    ]
    assert done.finish_reason == "tool_calls"
    assert done.output_tokens == 8


def test_tool_calling_anthropic_in_openai_out_anthropic_raw():
    """ops-agent 发 Anthropic 形 tools → OpenAIProvider 翻成 OpenAI 形发出;
    DeepSeek/OpenAI 回 tool_calls → raw 还原成 Anthropic 风格 tool_use,使消费者通用。"""
    req = NormalizedRequest(
        model="default",
        messages=[{"role": "user", "content": "诊断这段日志"}],
        max_tokens=64,
        system="你是诊断助手",
        tools=[
            {
                "name": "diagnose",
                "description": "结构化诊断",
                "input_schema": {"type": "object"},
            }
        ],
        tool_choice={"type": "tool", "name": "diagnose"},
    )
    client = _ToolClient()
    resp = OpenAIProvider(client).invoke(T, req, timeout=5.0)

    # 1) 发给上游的 tools 已翻成 OpenAI function 形,tool_choice 同理
    sent = client.chat.completions.last_kwargs
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "diagnose"
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "diagnose"}}

    # 2) 响应 raw 是 Anthropic 风格,含 tool_use 块 + 解析后的 input(ops-agent shim 据此还原)
    block = next(b for b in resp.raw["content"] if b["type"] == "tool_use")
    assert block["name"] == "diagnose"
    assert block["input"] == {"root_cause": "OOM"}
    assert resp.raw["stop_reason"] == "tool_use"
