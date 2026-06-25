# tests/test_gateway_stream.py
import pytest

from agent_ctl.config import RetryConfig
from agent_ctl.core.circuit import CircuitBreaker
from agent_ctl.core.cost import CostMeter
from agent_ctl.core.gateway import Gateway
from agent_ctl.core.router import Router
from agent_ctl.errors import RetriableError, TerminalError
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

    def __init__(self, parts, start_error=None, mid_error=None, it=10, ot=5):
        self._parts = parts
        self._start_error = start_error
        self._mid_error = mid_error
        self._it, self._ot = it, ot
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
            finish_reason="stop",
            input_tokens=self._it,
            output_tokens=self._ot,
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
