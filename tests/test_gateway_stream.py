# tests/test_gateway_stream.py
import pytest

from agent_ctl.config import RetryConfig
from agent_ctl.core.budget import BudgetGuard
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.errors import (
    AllTargetsFailed,
    BudgetExceeded,
    GatewayError,
    RetriableError,
    TerminalError,
)
from agent_ctl.models import NormalizedRequest, StreamChunk
from agent_ctl.providers.fake import FakeProvider
from agent_ctl.store.sqlite_store import SqliteCaptureStore

RETRY = RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0)
REQ = NormalizedRequest(
    model="default",
    messages=[{"role": "user", "content": "hi"}],
    metadata={"consumer": "t"},
)


class FakeStreamProvider:
    """带 stream 能力:start_error 在开流前抛(可回退);mid_error 在产出首块后抛。"""

    def __init__(
        self, parts, start_error=None, mid_error=None, it=10, ot=5, tool_calls=None
    ):
        self._parts = parts
        self._start_error = start_error
        self._mid_error = mid_error
        self._it, self._ot = it, ot
        self._tool_calls = tool_calls
        self.calls = 0

    def invoke(self, target, request, timeout):
        raise NotImplementedError

    def stream(self, target, request, timeout):
        self.calls += 1
        if self._start_error:
            raise self._start_error
        for i, p in enumerate(self._parts):
            if self._mid_error and i == 1:
                raise self._mid_error
            yield StreamChunk(text=p)
        yield StreamChunk(
            done=True,
            finish_reason="tool_calls" if self._tool_calls else "stop",
            input_tokens=self._it,
            output_tokens=self._ot,
            tool_calls=self._tool_calls,
        )


def _gw(providers, routes, store, circuit=None):
    return Gateway(
        router=Router(routes),
        providers=providers,
        cost_meter=CostMeter({"m": (1000.0, 1000.0)}),
        store=store,
        retry=RETRY,
        circuit=circuit,
    )


def test_stream_yields_deltas_and_captures(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    p = FakeStreamProvider(["he", "llo"])
    gw = _gw({"s": p}, {"default": ["s/m"]}, store)
    chunks = list(gw.invoke_stream(REQ))
    texts = [c.text for c in chunks if not c.done]
    assert texts == ["he", "llo"]
    done = [c for c in chunks if c.done][-1]
    assert done.finish_reason == "stop"
    rec = store.list_recent(1)[0]
    assert rec.status == "success"
    assert rec.output_redacted == "hello"  # 累计文本落库
    assert rec.input_tokens == 10 and rec.output_tokens == 5
    assert rec.cost_usd is not None


def test_stream_pre_open_failure_falls_back(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    bad = FakeStreamProvider([], start_error=RetriableError("503"))
    good = FakeStreamProvider(["ok"])
    gw = _gw(
        {"bad": bad, "good": good},
        {"default": ["bad/m", "good/m"]},
        store,
    )
    chunks = list(gw.invoke_stream(REQ))
    assert "".join(c.text for c in chunks if not c.done) == "ok"
    assert bad.calls == 1 and good.calls == 1  # 开流前失败 → 回退到 good
    rec = store.list_recent(1)[0]
    assert rec.status == "fallback_success"


def test_stream_terminal_before_open_propagates(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    bad = FakeStreamProvider([], start_error=TerminalError("401"))
    gw = _gw({"bad": bad}, {"default": ["bad/m"]}, store)
    with pytest.raises(TerminalError):
        list(gw.invoke_stream(REQ))
    rec = store.list_recent(1)[0]
    assert rec.status == "error"
    assert rec.error_type == "terminal"


def test_stream_mid_failure_no_fallback(tmp_path):
    """首块已出后中途失败:不回退(已发字节),记错并抛。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    mid = FakeStreamProvider(["a", "b", "c"], mid_error=RetriableError("drop"))
    other = FakeStreamProvider(["x"])
    gw = _gw(
        {"mid": mid, "other": other},
        {"default": ["mid/m", "other/m"]},
        store,
    )
    collected = []
    with pytest.raises(Exception):
        for c in gw.invoke_stream(REQ):
            if not c.done:
                collected.append(c.text)
    assert collected == ["a"]  # 首块已出
    assert other.calls == 0  # 已开流 → 不回退
    rec = store.list_recent(1)[0]
    assert rec.error_type == "stream"


def test_stream_degrades_to_buffered_when_no_stream_capability(tmp_path):
    """无 stream 能力的 provider(FakeProvider)退化为缓冲式,仍产出 chunk + 捕获。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    p = FakeProvider(["ok"])  # 只有 invoke,无 stream
    gw = _gw({"f": p}, {"default": ["f/m"]}, store)
    chunks = list(gw.invoke_stream(REQ))
    assert "".join(c.text for c in chunks if not c.done) == "fake-ok"
    rec = store.list_recent(1)[0]
    assert rec.status == "success"


def test_stream_mid_failure_charges_circuit_and_can_open(tmp_path):
    """F3:首块已出后中途断流计熔断;反复中途失败累计到阈值即开路(不自我赦免)。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30.0)
    p = FakeStreamProvider(["a", "b"], mid_error=RetriableError("drop"))
    gw = _gw({"s": p}, {"default": ["s/m"]}, store, circuit=cb)
    for _ in range(2):
        with pytest.raises(Exception):
            list(gw.invoke_stream(REQ))  # 每次:出首块 "a" 后中途失败 → record_failure
    assert cb.allow("s") is False  # 两次中途失败累计 → 开路(此前会"开流即成功"永不开路)


def test_stream_client_abort_captures_aborted_record(tmp_path):
    """G3:客户端中途断流(gen.close → GeneratorExit)落 aborted 记录(按已累计)。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    p = FakeStreamProvider(["a", "b", "c"])
    gw = _gw({"s": p}, {"default": ["s/m"]}, store)
    gen = gw.invoke_stream(REQ)
    first = next(gen)  # 拿首块 "a"
    assert first.text == "a"
    gen.close()  # 模拟客户端断开 → GeneratorExit 注入
    rec = store.list_recent(1)[0]
    assert rec.status == "aborted"
    assert rec.error_type == "client_abort"
    assert rec.output_redacted == "a"  # 已累计的部分文本仍落库


def test_stream_tool_calls_surfaced_and_captured(tmp_path):
    """G5:流式工具调用经末块 tool_calls 透出,捕获记录 tool_calls 计数。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    p = FakeStreamProvider(
        [],
        tool_calls=[{"id": "c1", "name": "diagnose", "arguments": '{"x":1}'}],
    )
    gw = _gw({"s": p}, {"default": ["s/m"]}, store)
    chunks = list(gw.invoke_stream(REQ))
    done = [c for c in chunks if c.done][-1]
    assert done.tool_calls == [{"id": "c1", "name": "diagnose", "arguments": '{"x":1}'}]
    rec = store.list_recent(1)[0]
    assert rec.tool_calls == 1  # 捕获记录工具调用数(此前流式恒为 0)


def test_stream_open_circuit_skips_provider(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    bad = FakeStreamProvider([], start_error=RetriableError("503"))
    good = FakeStreamProvider(["ok"])
    cb = CircuitBreaker(failure_threshold=1, cooldown_s=30.0)
    gw = _gw(
        {"bad": bad, "good": good},
        {"default": ["bad/m", "good/m"]},
        store,
        circuit=cb,
    )
    list(gw.invoke_stream(REQ))  # bad 开流失败 1 次 → 开路
    list(gw.invoke_stream(REQ))  # 第二次 bad 被熔断跳过
    assert bad.calls == 1  # 第二次未再打 bad


# ── 覆盖补强:stream 内预算/路由/未注册/全失败/缓冲回退/deadline ──────────────


def test_stream_budget_exceeded_before_open(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["s/m"]}),
        providers={"s": FakeStreamProvider(["x"])},
        cost_meter=CostMeter({}),
        store=store,
        retry=RETRY,
        budget=BudgetGuard(per_consumer={"t": 0.0}),  # cap 0 → 首次即超
    )
    with pytest.raises(BudgetExceeded):
        list(gw.invoke_stream(REQ))
    rec = store.list_recent(1)[0]
    assert rec.status == "error" and rec.error_type == "budget"


def test_stream_unknown_model_routing_error(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = _gw({"s": FakeStreamProvider(["x"])}, {"default": ["s/m"]}, store)
    with pytest.raises(GatewayError):
        list(
            gw.invoke_stream(
                NormalizedRequest(
                    model="missing",
                    messages=[{"role": "user", "content": "hi"}],
                    metadata={"consumer": "t"},
                )
            )
        )
    rec = store.list_recent(1)[0]
    assert rec.error_type == "routing"


def test_stream_direct_unregistered_provider(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = _gw({"s": FakeStreamProvider(["x"])}, {"default": ["s/m"]}, store)
    with pytest.raises(GatewayError):
        list(
            gw.invoke_stream(
                NormalizedRequest(
                    model="nope/x",
                    messages=[{"role": "user", "content": "hi"}],
                    metadata={"consumer": "t"},
                )
            )
        )
    rec = store.list_recent(1)[0]
    assert rec.error_type == "provider"


def test_stream_all_targets_fail(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    only = FakeStreamProvider([], start_error=RetriableError("503"))
    gw = _gw({"s": only}, {"default": ["s/m"]}, store)
    with pytest.raises(AllTargetsFailed):
        list(gw.invoke_stream(REQ))
    rec = store.list_recent(1)[0]
    assert rec.error_type == "all_failed"


def test_stream_buffered_terminal_propagates(tmp_path):
    """无 stream 能力的目标缓冲降级,invoke 抛 terminal → 透传不回退。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = _gw({"f": FakeProvider(["terminal"])}, {"default": ["f/m"]}, store)
    with pytest.raises(TerminalError):
        list(gw.invoke_stream(REQ))
    rec = store.list_recent(1)[0]
    assert rec.error_type == "terminal"


def test_stream_buffered_retriable_falls_back_to_native(tmp_path):
    """无 stream 能力目标缓冲降级遇 retriable → 回退到下一个原生流式目标。"""
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    buf = FakeProvider(["retriable"])  # 无 stream
    nat = FakeStreamProvider(["ok"])
    gw = _gw({"buf": buf, "nat": nat}, {"default": ["buf/m", "nat/m"]}, store)
    chunks = list(gw.invoke_stream(REQ))
    assert "".join(c.text for c in chunks if not c.done) == "ok"
    rec = store.list_recent(1)[0]
    assert rec.status == "fallback_success"


def test_stream_deadline_break_stops_fallback(tmp_path):
    """第一目标耗时超 deadline → 第二目标在循环顶被 deadline 守卫跳过。"""
    import time as _t

    class SlowBufferedFail:
        def invoke(self, target, request, timeout):
            _t.sleep(0.06)
            raise RetriableError("slow")

    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["a/m", "b/m"]}),
        providers={"a": SlowBufferedFail(), "b": FakeStreamProvider(["x"])},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=1, timeout_s=60.0),
        request_deadline_s=0.05,
    )
    with pytest.raises(AllTargetsFailed):
        list(gw.invoke_stream(REQ))
    rec = store.list_recent(1)[0]
    assert rec.attempts[-1].outcome == "deadline"


def test_stream_empty_generator_captures_and_emits_done(tmp_path):
    """provider 的 stream 一个块都不产(连 done 都没)→ first=None,仍捕获 + 收尾 done。"""

    class EmptyStream:
        def invoke(self, target, request, timeout):
            raise NotImplementedError

        def stream(self, target, request, timeout):
            return iter(())  # 空生成器 → next() 直接 StopIteration

    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = _gw({"e": EmptyStream()}, {"default": ["e/m"]}, store)
    chunks = list(gw.invoke_stream(REQ))
    assert chunks and chunks[-1].done  # 收尾 done
    assert "".join(c.text for c in chunks if not c.done) == ""
    rec = store.list_recent(1)[0]
    assert rec.status == "success"


def test_stream_deadline_truncates_long_stream(tmp_path):
    """#2:开流后 deadline 也要管——长流超预算应被逐块截断,不是无视 deadline 跑完。"""
    import time as _t

    class ManyChunks:
        def invoke(self, *a):
            raise NotImplementedError

        def stream(self, target, request, timeout):
            for i in range(20):
                _t.sleep(0.02)  # 20 块 × 20ms = 0.4s 总时长
                yield StreamChunk(text=str(i))
            yield StreamChunk(
                done=True, finish_reason="stop", input_tokens=1, output_tokens=1
            )

    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    gw = Gateway(
        router=Router({"default": ["s/m"]}),
        providers={"s": ManyChunks()},
        cost_meter=CostMeter({}),
        store=store,
        retry=RetryConfig(max_attempts_per_target=1, timeout_s=60.0),
        request_deadline_s=0.1,  # 100ms 总预算
    )
    t0 = _t.monotonic()
    texts = [c.text for c in gw.invoke_stream(REQ) if not c.done]
    elapsed = _t.monotonic() - t0
    assert elapsed < 0.3  # 被 deadline 截断,远小于 0.4s
    assert len(texts) < 20  # 没把整条流跑完
    rec = store.list_recent(1)[0]
    assert rec.status == "deadline"
