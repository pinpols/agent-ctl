# agentctl 调用网关 + 全量捕获 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个传输无关的 LLM 调用治理引擎(路由/回退/超时重试/成本/缓存)+ 每次调用全量捕获,并以库形态接入 ops-agent 作为首个消费者。

**Architecture:** 纯逻辑核心 `Gateway` 编排 Router/Provider/CostMeter/Cache/CaptureStore(全经接口注入);本期出"库形态"`GatewayClient`,代理形态(server)仅留骨架。存储用 SQLite,离线 `FakeProvider` 支撑全部 TDD。

**Tech Stack:** Python 3.12、pydantic v2、anthropic SDK、sqlite3(stdlib)、pytest、ruff、mypy。

## Global Constraints

- Python ≥ 3.11(开发用 3.12,与 ops-agent 同栈)。
- 依赖最小:核心仅 `pydantic`;provider 层 `anthropic`;测试 `pytest`。不引入 web 框架(server 本期只留骨架)。
- **治理侧 fail-open**:cache/store/cost 任何异常只记 warn 不阻断主调用;LLM 调用自身终态错误照常透传。
- **不写 if-chain ≥3 分支**,用 dict 路由表。
- 所有持久化经接口(Protocol),本期实现 SQLite,预留 PG。
- TDD:每个任务先写失败测试;离线 `FakeProvider`,测试不需真 API key。
- 落库前必脱敏(token/DSN 口令/JWT/邮箱/手机号)。
- **线程安全**:消费者可能多线程并发调用 → `SqliteCaptureStore`(WAL + Lock + `check_same_thread=False`)、`MemoryCache`(Lock)。
- **失败也留痕**:任一目标的每次尝试(成功/失败)都必须进入最终 `CallRecord.attempts`(共享 attempts 列表,不困在被抛弃的局部变量)。
- **启动期 fail-fast**:装配/自检时校验每条路由目标的 provider 已注册/已知。
- **仅非流式**:v1 不做 streaming(治理/捕获/成本/缓存键在非流式上先做扎实)。
- 项目名 `agentctl`;包根 `agentctl/`。

---

### Task 1: 项目脚手架 + 配置

**Files:**
- Create: `pyproject.toml`
- Create: `agentctl/__init__.py`
- Create: `agentctl/config.py`
- Create: `tests/test_config.py`
- Create: `agentctl.example.yaml`

**Interfaces:**
- Produces: `load_config(path: str | None = None) -> Config`;`Config` 含 `routes: dict[str, list[str]]`、`prices: dict[str, tuple[float, float]]`、`cache_enabled: bool`、`cache_ttl_s: int`、`profile: str`、`db_path: str`、`retry: RetryConfig`。`RetryConfig(max_attempts_per_target: int, base_backoff_s: float, timeout_s: float)`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from agentctl.config import load_config

def test_load_config_from_yaml(tmp_path):
    cfg_file = tmp_path / "agentctl.yaml"
    cfg_file.write_text(
        "routes:\n"
        "  default: [anthropic/claude-opus-4-8, anthropic/claude-sonnet-4-6]\n"
        "prices:\n"
        "  claude-opus-4-8: [5.0, 25.0]\n"
        "cache_enabled: true\n"
        "cache_ttl_s: 600\n"
        "profile: dev\n"
        "db_path: ':memory:'\n"
        "retry:\n"
        "  max_attempts_per_target: 2\n"
        "  base_backoff_s: 0.01\n"
        "  timeout_s: 30\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.routes["default"] == ["anthropic/claude-opus-4-8", "anthropic/claude-sonnet-4-6"]
    assert cfg.prices["claude-opus-4-8"] == (5.0, 25.0)
    assert cfg.retry.max_attempts_per_target == 2

def test_load_config_defaults_when_missing():
    cfg = load_config(None)
    assert cfg.profile == "dev"
    assert cfg.cache_enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.config'`

- [ ] **Step 3: Write pyproject.toml + package init**

```toml
# pyproject.toml
[project]
name = "agentctl"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pydantic>=2", "pyyaml>=6"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.5", "mypy>=1.10"]
anthropic = ["anthropic>=0.40"]

[project.scripts]
agentctl = "agentctl.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# agentctl/__init__.py
"""agentctl — AgentOps 控制面:调用网关 + 全量捕获。"""
__version__ = "0.1.0"
```

- [ ] **Step 4: Write config.py**

```python
# agentctl/config.py
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class RetryConfig(BaseModel):
    max_attempts_per_target: int = 2
    base_backoff_s: float = 0.2
    timeout_s: float = 60.0


class Config(BaseModel):
    routes: dict[str, list[str]] = {"default": ["anthropic/claude-sonnet-4-6"]}
    prices: dict[str, tuple[float, float]] = {}
    cache_enabled: bool = True
    cache_ttl_s: int = 600
    profile: str = "dev"
    db_path: str = ".agentctl/capture.db"
    retry: RetryConfig = RetryConfig()


def load_config(path: str | None = None) -> Config:
    """从 yaml 读配置;path 为 None 时尝试 ./agentctl.yaml,无则用默认。env 不覆盖结构,仅 profile。"""
    data: dict = {}
    candidate = path or ("agentctl.yaml" if Path("agentctl.yaml").exists() else None)
    if candidate:
        with open(candidate, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    cfg = Config(**data)
    if env_profile := os.getenv("AGENTCTL_PROFILE"):
        cfg = cfg.model_copy(update={"profile": env_profile})
    return cfg
```

- [ ] **Step 5: Write example config**

```yaml
# agentctl.example.yaml
routes:
  default: [anthropic/claude-opus-4-8, anthropic/claude-sonnet-4-6]
prices:                 # [input_per_1M_usd, output_per_1M_usd]
  claude-opus-4-8: [5.0, 25.0]
  claude-sonnet-4-6: [3.0, 15.0]
cache_enabled: true
cache_ttl_s: 600
profile: dev
db_path: .agentctl/capture.db
retry:
  max_attempts_per_target: 2
  base_backoff_s: 0.2
  timeout_s: 60
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml agentctl/ tests/test_config.py agentctl.example.yaml
git commit -m "feat(config): 项目脚手架 + yaml 配置加载"
```

---

### Task 2: 领域模型 + 错误类型

**Files:**
- Create: `agentctl/models.py`
- Create: `agentctl/errors.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `Target(provider: str, model: str)`,`.name -> "{provider}/{model}"`,`Target.parse("anthropic/x") -> Target`。
  - `NormalizedRequest(model: str, messages: list[dict], max_tokens: int = 1024, temperature: float | None = None, tools: list | None = None, metadata: dict = {})`。
  - `NormalizedResponse(text: str, finish_reason: str | None, tool_calls: int, input_tokens: int, output_tokens: int, raw: dict | None = None)`。
  - `Attempt(provider: str, model: str, outcome: str, latency_ms: int, error: str | None)`。
  - `CallRecord(...)` 全字段见 spec §4。
  - 错误:`GatewayError`、`RetriableError(GatewayError)`、`TerminalError(GatewayError)`、`AllTargetsFailed(GatewayError)`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from agentctl.models import Target, NormalizedRequest, CallRecord, Attempt
from agentctl.errors import RetriableError, TerminalError, GatewayError

def test_target_parse_and_name():
    t = Target.parse("anthropic/claude-opus-4-8")
    assert t.provider == "anthropic"
    assert t.model == "claude-opus-4-8"
    assert t.name == "anthropic/claude-opus-4-8"

def test_call_record_minimal():
    rec = CallRecord(
        id="abc", consumer="ops-agent", model_requested="default",
        model_resolved="anthropic/claude-opus-4-8", status="success",
        latency_ms=120, input_tokens=10, output_tokens=5,
        attempts=[Attempt(provider="anthropic", model="claude-opus-4-8",
                          outcome="success", latency_ms=120, error=None)],
    )
    assert rec.status == "success"
    assert rec.attempts[0].outcome == "success"

def test_error_hierarchy():
    assert issubclass(RetriableError, GatewayError)
    assert issubclass(TerminalError, GatewayError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.models'`

- [ ] **Step 3: Write errors.py**

```python
# agentctl/errors.py
class GatewayError(Exception):
    """网关层基异常。"""

class RetriableError(GatewayError):
    """可重试(429/5xx/overloaded/timeout)。"""

class TerminalError(GatewayError):
    """终态(4xx 鉴权/参数),不重试,直接透传。"""

class AllTargetsFailed(GatewayError):
    """路由链全部目标耗尽。"""
```

- [ ] **Step 4: Write models.py**

```python
# agentctl/models.py
from __future__ import annotations

from pydantic import BaseModel


class Target(BaseModel):
    provider: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    @classmethod
    def parse(cls, spec: str) -> "Target":
        provider, _, model = spec.partition("/")
        if not provider or not model:
            raise ValueError(f"bad target spec: {spec!r} (want 'provider/model')")
        return cls(provider=provider, model=model)


class NormalizedRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float | None = None
    tools: list | None = None
    metadata: dict = {}


class NormalizedResponse(BaseModel):
    text: str
    finish_reason: str | None = None
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict | None = None


class Attempt(BaseModel):
    provider: str
    model: str
    outcome: str  # success | retriable | terminal | timeout
    latency_ms: int
    error: str | None = None


class CallRecord(BaseModel):
    id: str
    ts: float = 0.0
    latency_ms: int = 0
    consumer: str = "unknown"
    call_site: str | None = None
    trace_id: str | None = None
    model_requested: str = ""
    params: dict = {}
    messages_redacted: list[dict] | None = None
    prompt_version: str | None = None
    model_resolved: str | None = None
    attempts: list[Attempt] = []
    output_redacted: str | None = None
    finish_reason: str | None = None
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    cache_hit: bool = False
    cache_key: str | None = None
    status: str = "success"  # success | fallback_success | error
    error_type: str | None = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add agentctl/models.py agentctl/errors.py tests/test_models.py
git commit -m "feat(models): 领域模型 + 错误类型"
```

---

### Task 3: Router

**Files:**
- Create: `agentctl/core/__init__.py`
- Create: `agentctl/core/router.py`
- Create: `tests/test_router.py`

**Interfaces:**
- Consumes: `Target`(Task 2)。
- Produces: `Router(routes: dict[str, list[str]])`;`resolve(logical: str) -> list[Target]`(逻辑名命中返回有序目标链;未命中抛 `KeyError`)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
import pytest
from agentctl.core.router import Router

def test_resolve_returns_ordered_targets():
    r = Router({"default": ["anthropic/opus", "anthropic/sonnet"]})
    targets = r.resolve("default")
    assert [t.name for t in targets] == ["anthropic/opus", "anthropic/sonnet"]

def test_resolve_unknown_logical_raises():
    r = Router({"default": ["anthropic/opus"]})
    with pytest.raises(KeyError):
        r.resolve("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.core.router'`

- [ ] **Step 3: Write implementation**

```python
# agentctl/core/__init__.py
```

```python
# agentctl/core/router.py
from __future__ import annotations

from agentctl.models import Target


class Router:
    """逻辑模型名 → 有序目标链。纯查表,无副作用。"""

    def __init__(self, routes: dict[str, list[str]]) -> None:
        self._routes = {k: [Target.parse(s) for s in v] for k, v in routes.items()}

    def resolve(self, logical: str) -> list[Target]:
        if logical not in self._routes:
            raise KeyError(f"unknown logical model: {logical!r}")
        return list(self._routes[logical])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/core/ tests/test_router.py
git commit -m "feat(router): 逻辑名→有序目标链路由表"
```

---

### Task 4: Provider 协议 + FakeProvider

**Files:**
- Create: `agentctl/providers/__init__.py`
- Create: `agentctl/providers/base.py`
- Create: `agentctl/providers/fake.py`
- Create: `tests/test_fake_provider.py`

**Interfaces:**
- Consumes: `Target`、`NormalizedRequest`、`NormalizedResponse`、`RetriableError`、`TerminalError`(Tasks 2)。
- Produces:
  - `Provider`(Protocol):`invoke(target: Target, request: NormalizedRequest, timeout: float) -> NormalizedResponse`。
  - `FakeProvider(script: list[str] | None)`:按脚本逐次产出行为。脚本项:`"ok"` / `"retriable"` / `"terminal"` / `"timeout"`;默认全 `"ok"`。`ok` 返回固定 `NormalizedResponse(text="fake-ok", input_tokens=10, output_tokens=5)`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fake_provider.py
import pytest
from agentctl.models import Target, NormalizedRequest
from agentctl.providers.fake import FakeProvider
from agentctl.errors import RetriableError, TerminalError

REQ = NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}])
T = Target(provider="fake", model="m")

def test_fake_ok():
    p = FakeProvider(["ok"])
    resp = p.invoke(T, REQ, timeout=1.0)
    assert resp.text == "fake-ok"
    assert resp.input_tokens == 10

def test_fake_retriable_then_ok():
    p = FakeProvider(["retriable", "ok"])
    with pytest.raises(RetriableError):
        p.invoke(T, REQ, timeout=1.0)
    assert p.invoke(T, REQ, timeout=1.0).text == "fake-ok"

def test_fake_terminal():
    p = FakeProvider(["terminal"])
    with pytest.raises(TerminalError):
        p.invoke(T, REQ, timeout=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fake_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.providers.fake'`

- [ ] **Step 3: Write base.py + fake.py**

```python
# agentctl/providers/__init__.py
```

```python
# agentctl/providers/base.py
from __future__ import annotations

from typing import Protocol

from agentctl.models import NormalizedRequest, NormalizedResponse, Target


class Provider(Protocol):
    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse: ...
```

```python
# agentctl/providers/fake.py
from __future__ import annotations

from agentctl.errors import RetriableError, TerminalError
from agentctl.models import NormalizedRequest, NormalizedResponse, Target


class FakeProvider:
    """离线测试用:按脚本逐次产出 ok/retriable/terminal/timeout。"""

    def __init__(self, script: list[str] | None = None) -> None:
        self._script = list(script or ["ok"])
        self._i = 0
        self.calls: list[Target] = []

    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse:
        self.calls.append(target)
        action = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        if action == "ok":
            return NormalizedResponse(
                text="fake-ok", finish_reason="end_turn",
                input_tokens=10, output_tokens=5,
            )
        if action == "retriable":
            raise RetriableError("fake retriable")
        if action == "terminal":
            raise TerminalError("fake terminal")
        if action == "timeout":
            raise TimeoutError("fake timeout")
        raise ValueError(f"bad script action: {action}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fake_provider.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/providers/ tests/test_fake_provider.py
git commit -m "feat(providers): Provider 协议 + 离线 FakeProvider"
```

---

### Task 5: CostMeter

**Files:**
- Create: `agentctl/core/cost.py`
- Create: `tests/test_cost.py`

**Interfaces:**
- Produces: `CostMeter(prices: dict[str, tuple[float, float]])`;`cost(model: str, input_tokens: int, output_tokens: int) -> float | None`。价表单位 = 每 100 万 token 美元。未知模型返回 `None`(并 warn)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost.py
from agentctl.core.cost import CostMeter

def test_cost_known_model():
    m = CostMeter({"opus": (5.0, 25.0)})  # $/1M
    # 1000 in, 500 out → 1000/1e6*5 + 500/1e6*25 = 0.005 + 0.0125
    assert m.cost("opus", 1000, 500) == 0.0175

def test_cost_unknown_model_returns_none():
    m = CostMeter({})
    assert m.cost("mystery", 1000, 500) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cost.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.core.cost'`

- [ ] **Step 3: Write implementation**

```python
# agentctl/core/cost.py
from __future__ import annotations

import logging

log = logging.getLogger("agentctl.cost")


class CostMeter:
    """按价表(每 1M token 美元)算调用成本;未知模型返回 None。"""

    def __init__(self, prices: dict[str, tuple[float, float]]) -> None:
        self._prices = prices

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self._prices.get(model)
        if price is None:
            log.warning("unknown model for pricing: %s (cost=None)", model)
            return None
        in_price, out_price = price
        return round(input_tokens / 1_000_000 * in_price
                     + output_tokens / 1_000_000 * out_price, 6)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cost.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/core/cost.py tests/test_cost.py
git commit -m "feat(cost): 价表成本核算(未知模型 None+warn)"
```

---

### Task 6: 脱敏

**Files:**
- Create: `agentctl/store/__init__.py`
- Create: `agentctl/store/redaction.py`
- Create: `tests/test_redaction.py`

**Interfaces:**
- Produces: `redact(text: str) -> str`(遮蔽 sk-/bearer token、DSN 口令、JWT、邮箱、手机号);`redact_messages(messages: list[dict]) -> list[dict]`(对每条 content 文本脱敏)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_redaction.py
from agentctl.store.redaction import redact, redact_messages

def test_redact_token_and_email():
    out = redact("key=sk-ant-abc123XYZ contact a@b.com")
    assert "sk-ant-abc123XYZ" not in out
    assert "a@b.com" not in out
    assert "[REDACTED]" in out

def test_redact_messages_preserves_structure():
    msgs = [{"role": "user", "content": "my token sk-ant-secret999"}]
    out = redact_messages(msgs)
    assert out[0]["role"] == "user"
    assert "sk-ant-secret999" not in out[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_redaction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.store.redaction'`

- [ ] **Step 3: Write implementation**

```python
# agentctl/store/__init__.py
```

```python
# agentctl/store/redaction.py
from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{8,}"),            # api keys
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.]+"),     # bearer
    re.compile(r"eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"),  # JWT
    re.compile(r"://[^:/@\s]+:([^@/\s]+)@"),          # DSN password (group 1)
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"\b1[3-9]\d{9}\b"),                   # CN mobile
]
_MASK = "[REDACTED]"


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _PATTERNS:
        out = pat.sub(_MASK, out)
    return out


def redact_messages(messages: list[dict]) -> list[dict]:
    result = []
    for m in messages:
        copy = dict(m)
        content = copy.get("content")
        if isinstance(content, str):
            copy["content"] = redact(content)
        result.append(copy)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_redaction.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/store/__init__.py agentctl/store/redaction.py tests/test_redaction.py
git commit -m "feat(redaction): 落库前脱敏(token/DSN/JWT/邮箱/手机)"
```

---

### Task 7: CaptureStore(SQLite)

**Files:**
- Create: `agentctl/store/base.py`
- Create: `agentctl/store/sqlite_store.py`
- Create: `tests/test_sqlite_store.py`

**Interfaces:**
- Consumes: `CallRecord`(Task 2)。
- Produces:
  - `CaptureStore`(Protocol):`save(record: CallRecord) -> None`、`list_recent(limit: int) -> list[CallRecord]`、`cost_summary() -> dict`(`{"calls": int, "total_cost_usd": float, "total_input_tokens": int, "total_output_tokens": int}`)。
  - `SqliteCaptureStore(db_path: str)`:建表幂等;`db_path=":memory:"` 走内存。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sqlite_store.py
from agentctl.models import CallRecord
from agentctl.store.sqlite_store import SqliteCaptureStore

def _rec(cid, cost):
    return CallRecord(id=cid, consumer="t", status="success",
                      input_tokens=10, output_tokens=5, cost_usd=cost)

def test_save_and_list_recent(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    store.save(_rec("a", 0.01))
    store.save(_rec("b", 0.02))
    recent = store.list_recent(10)
    assert {r.id for r in recent} == {"a", "b"}

def test_cost_summary(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    store.save(_rec("a", 0.01))
    store.save(_rec("b", 0.02))
    s = store.cost_summary()
    assert s["calls"] == 2
    assert round(s["total_cost_usd"], 4) == 0.03
    assert s["total_input_tokens"] == 20

def test_concurrent_writes_thread_safe(tmp_path):
    import threading
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    def writer(start):
        for i in range(20):
            store.save(_rec(f"{start}-{i}", 0.001))
    threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert store.cost_summary()["calls"] == 100  # 5 线程 × 20,无 'database is locked'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sqlite_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.store.sqlite_store'`

- [ ] **Step 3: Write base.py + sqlite_store.py**

```python
# agentctl/store/base.py
from __future__ import annotations

from typing import Protocol

from agentctl.models import CallRecord


class CaptureStore(Protocol):
    def save(self, record: CallRecord) -> None: ...
    def list_recent(self, limit: int) -> list[CallRecord]: ...
    def cost_summary(self) -> dict: ...
```

```python
# agentctl/store/sqlite_store.py
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from agentctl.models import CallRecord


class SqliteCaptureStore:
    """SQLite 捕获存储:一行一条 CallRecord(JSON 整存 + 关键列冗余便于聚合)。

    线程安全:check_same_thread=False 允许跨线程复用连接,WAL 提升并发读,
    写入加 Lock 串行化(SQLite 单写),避免消费者多线程并发调用时 'database is locked'。
    """

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS call_record ("
                " id TEXT PRIMARY KEY, ts REAL, consumer TEXT, status TEXT,"
                " input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,"
                " doc TEXT NOT NULL)"
            )
            self._conn.commit()

    def save(self, record: CallRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO call_record"
                " (id, ts, consumer, status, input_tokens, output_tokens, cost_usd, doc)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (record.id, record.ts, record.consumer, record.status,
                 record.input_tokens, record.output_tokens, record.cost_usd,
                 record.model_dump_json()),
            )
            self._conn.commit()

    def list_recent(self, limit: int) -> list[CallRecord]:
        rows = self._conn.execute(
            "SELECT doc FROM call_record ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [CallRecord(**json.loads(r["doc"])) for r in rows]

    def cost_summary(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) c, COALESCE(SUM(cost_usd),0) cost,"
            " COALESCE(SUM(input_tokens),0) it, COALESCE(SUM(output_tokens),0) ot"
            " FROM call_record"
        ).fetchone()
        return {"calls": row["c"], "total_cost_usd": row["cost"],
                "total_input_tokens": row["it"], "total_output_tokens": row["ot"]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sqlite_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/store/base.py agentctl/store/sqlite_store.py tests/test_sqlite_store.py
git commit -m "feat(store): SQLite 捕获存储 + 成本汇总"
```

---

### Task 8: Cache(精确匹配 + TTL)

**Files:**
- Create: `agentctl/core/cache.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Consumes: `NormalizedRequest`、`NormalizedResponse`(Task 2)。
- Produces:
  - `make_key(request: NormalizedRequest) -> str`(对 model+归一化 messages+max_tokens+temperature 取 sha256)。
  - `Cache`(Protocol):`get(key: str) -> NormalizedResponse | None`、`set(key: str, resp: NormalizedResponse, ttl_s: int) -> None`。
  - `MemoryCache()`:进程内 dict + 过期戳(本期默认实现,测试与单机均用)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache.py
from agentctl.models import NormalizedRequest, NormalizedResponse
from agentctl.core.cache import make_key, MemoryCache

REQ = NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}])

def test_make_key_stable_and_distinct():
    k1 = make_key(REQ)
    k2 = make_key(NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}]))
    k3 = make_key(NormalizedRequest(model="default", messages=[{"role": "user", "content": "bye"}]))
    assert k1 == k2
    assert k1 != k3

def test_cache_get_set_and_miss():
    c = MemoryCache()
    assert c.get("k") is None
    c.set("k", NormalizedResponse(text="cached"), ttl_s=60)
    assert c.get("k").text == "cached"

def test_cache_expiry(monkeypatch):
    import agentctl.core.cache as mod
    now = {"t": 1000.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: now["t"])
    c = mod.MemoryCache()
    c.set("k", NormalizedResponse(text="x"), ttl_s=10)
    now["t"] = 1005.0
    assert c.get("k") is not None
    now["t"] = 1011.0
    assert c.get("k") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.core.cache'`

- [ ] **Step 3: Write implementation**

```python
# agentctl/core/cache.py
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Protocol

from agentctl.models import NormalizedRequest, NormalizedResponse


def make_key(request: NormalizedRequest) -> str:
    payload = json.dumps(
        {"model": request.model, "messages": request.messages,
         "max_tokens": request.max_tokens, "temperature": request.temperature},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache(Protocol):
    def get(self, key: str) -> NormalizedResponse | None: ...
    def set(self, key: str, resp: NormalizedResponse, ttl_s: int) -> None: ...


class MemoryCache:
    """进程内精确匹配缓存 + TTL。加锁保证多线程并发调用安全。"""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, NormalizedResponse]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> NormalizedResponse | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, resp = entry
            if time.monotonic() > expires_at:
                self._data.pop(key, None)
                return None
            return resp

    def set(self, key: str, resp: NormalizedResponse, ttl_s: int) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl_s, resp)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/core/cache.py tests/test_cache.py
git commit -m "feat(cache): 精确匹配 + TTL 内存缓存"
```

---

### Task 9: Gateway — 单目标调用(超时 + 重试 + 分类)

**Files:**
- Create: `agentctl/core/gateway.py`
- Create: `tests/test_gateway_retry.py`

**Interfaces:**
- Consumes: `Provider`、`Target`、`NormalizedRequest`、`NormalizedResponse`、`Attempt`、`RetriableError`、`TerminalError`、`RetryConfig`(Tasks 1/2/4)。
- Produces:`Gateway._invoke_target(provider: Provider, target: Target, request: NormalizedRequest, attempts: list[Attempt]) -> NormalizedResponse`;**把每次尝试 append 到调用方传入的 `attempts` 列表(成功/失败都 append),再返回或抛出**——这样失败目标的尝试痕迹不会随异常丢失(修复"attempts 丢失" bug)。终态错误立即抛 `TerminalError`;可重试错误退避重试至 `max_attempts_per_target`,耗尽抛 `RetriableError`。`Gateway.__init__(router, providers: dict[str, Provider], cost_meter, store, cache=None, retry=RetryConfig(), cache_enabled=True, cache_ttl_s=600)`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gateway_retry.py
import pytest
from agentctl.core.gateway import Gateway
from agentctl.core.router import Router
from agentctl.core.cost import CostMeter
from agentctl.config import RetryConfig
from agentctl.providers.fake import FakeProvider
from agentctl.models import Target, NormalizedRequest
from agentctl.errors import RetriableError, TerminalError

REQ = NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}])
T = Target(provider="fake", model="m")

def _gw(provider, retry=RetryConfig(max_attempts_per_target=2, base_backoff_s=0.0, timeout_s=1.0)):
    return Gateway(router=Router({"default": ["fake/m"]}),
                   providers={"fake": provider}, cost_meter=CostMeter({}),
                   store=None, cache=None, retry=retry)

def test_retriable_then_success_within_target():
    gw = _gw(FakeProvider(["retriable", "ok"]))
    attempts = []
    resp = gw._invoke_target(FakeProvider(["retriable", "ok"]), T, REQ, attempts)
    assert resp.text == "fake-ok"
    assert [a.outcome for a in attempts] == ["retriable", "success"]

def test_terminal_not_retried_but_attempt_recorded():
    p = FakeProvider(["terminal", "ok"])
    gw = _gw(p)
    attempts = []
    with pytest.raises(TerminalError):
        gw._invoke_target(p, T, REQ, attempts)
    assert len(p.calls) == 1  # 终态不重试
    assert [a.outcome for a in attempts] == ["terminal"]  # 失败也留痕

def test_retriable_exhausted_records_all_attempts():
    p = FakeProvider(["retriable", "retriable"])
    gw = _gw(p)
    attempts = []
    with pytest.raises(RetriableError):
        gw._invoke_target(p, T, REQ, attempts)
    assert len(p.calls) == 2
    assert [a.outcome for a in attempts] == ["retriable", "retriable"]  # 失败也留痕
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gateway_retry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.core.gateway'`

- [ ] **Step 3: Write gateway.py (单目标部分)**

```python
# agentctl/core/gateway.py
from __future__ import annotations

import time

from agentctl.config import RetryConfig
from agentctl.core.cost import CostMeter
from agentctl.core.router import Router
from agentctl.errors import RetriableError, TerminalError
from agentctl.models import Attempt, NormalizedRequest, NormalizedResponse, Target
from agentctl.providers.base import Provider


class Gateway:
    def __init__(self, router: Router, providers: dict[str, Provider],
                 cost_meter: CostMeter, store=None, cache=None,
                 retry: RetryConfig | None = None,
                 cache_enabled: bool = True, cache_ttl_s: int = 600) -> None:
        self._router = router
        self._providers = providers
        self._cost = cost_meter
        self._store = store
        self._cache = cache
        self._retry = retry or RetryConfig()
        self._cache_enabled = cache_enabled
        self._cache_ttl_s = cache_ttl_s

    def _invoke_target(
        self, provider: Provider, target: Target, request: NormalizedRequest,
        attempts: list[Attempt],
    ) -> NormalizedResponse:
        """对单目标尝试(含重试)。每次尝试都 append 到调用方的 attempts(成功/失败均留痕)。"""
        last_exc: Exception = RetriableError("no attempt made")
        for n in range(self._retry.max_attempts_per_target):
            started = time.monotonic()
            try:
                resp = provider.invoke(target, request, self._retry.timeout_s)
                attempts.append(Attempt(provider=target.provider, model=target.model,
                                        outcome="success",
                                        latency_ms=int((time.monotonic() - started) * 1000)))
                return resp
            except TerminalError as exc:
                attempts.append(Attempt(provider=target.provider, model=target.model,
                                        outcome="terminal",
                                        latency_ms=int((time.monotonic() - started) * 1000),
                                        error=str(exc)))
                raise
            except (RetriableError, TimeoutError) as exc:
                outcome = "timeout" if isinstance(exc, TimeoutError) else "retriable"
                attempts.append(Attempt(provider=target.provider, model=target.model,
                                        outcome=outcome,
                                        latency_ms=int((time.monotonic() - started) * 1000),
                                        error=str(exc)))
                last_exc = exc
                if n < self._retry.max_attempts_per_target - 1:
                    time.sleep(self._retry.base_backoff_s * (2 ** n))
        raise RetriableError(str(last_exc))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gateway_retry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/core/gateway.py tests/test_gateway_retry.py
git commit -m "feat(gateway): 单目标超时+重试+错误分类"
```

---

### Task 10: Gateway — 回退 + 全编排(缓存/成本/捕获/状态)

**Files:**
- Modify: `agentctl/core/gateway.py`
- Create: `tests/test_gateway_invoke.py`

**Interfaces:**
- Consumes: Task 9 的 `_invoke_target`、`Router.resolve`、`CostMeter.cost`、`Cache`、`CaptureStore`、`make_key`、`redact_messages`/`redact`、`AllTargetsFailed`。
- Produces: `Gateway.invoke(request: NormalizedRequest) -> NormalizedResponse`。流程:缓存查 → 路由 → 逐目标 `_invoke_target`(可重试失败则回退下一目标)→ 算成本 → 写捕获(脱敏)→ 返回。`status`:首目标成功=`success`;回退后成功=`fallback_success`;全失败=抛 `AllTargetsFailed` 且写 `status=error`。治理侧(cache/store/cost)异常 fail-open(warn,不阻断)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gateway_invoke.py
import pytest
from agentctl.core.gateway import Gateway
from agentctl.core.router import Router
from agentctl.core.cost import CostMeter
from agentctl.core.cache import MemoryCache
from agentctl.config import RetryConfig
from agentctl.providers.fake import FakeProvider
from agentctl.store.sqlite_store import SqliteCaptureStore
from agentctl.models import NormalizedRequest
from agentctl.errors import AllTargetsFailed

REQ = NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}],
                        metadata={"consumer": "t"})
RETRY = RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0)

def _gw(provider, store, cache=None):
    return Gateway(router=Router({"default": ["fake/a", "fake/b"]}),
                   providers={"fake": provider}, cost_meter=CostMeter({"a": (5.0, 25.0)}),
                   store=store, cache=cache, retry=RETRY)

def test_primary_success(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    resp = _gw(FakeProvider(["ok"]), store).invoke(REQ)
    assert resp.text == "fake-ok"
    rec = store.list_recent(1)[0]
    assert rec.status == "success"
    assert rec.model_resolved == "fake/a"
    assert rec.cost_usd is not None  # model 'a' 有价表

def test_fallback_to_second_target(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    # 第一目标 retriable(耗尽 1 次)→ 回退第二目标 ok
    resp = _gw(FakeProvider(["retriable", "ok"]), store).invoke(REQ)
    assert resp.text == "fake-ok"
    rec = store.list_recent(1)[0]
    assert rec.status == "fallback_success"
    assert len(rec.attempts) == 2

def test_all_targets_fail_records_error_with_all_attempts(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    with pytest.raises(AllTargetsFailed):
        _gw(FakeProvider(["retriable", "retriable"]), store).invoke(REQ)
    rec = store.list_recent(1)[0]
    assert rec.status == "error"
    # 修复验证:两个目标各 1 次尝试都留痕(此前会丢)
    assert len(rec.attempts) == 2
    assert all(a.outcome == "retriable" for a in rec.attempts)

def test_cache_hit_skips_provider_and_costs_zero(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    cache = MemoryCache()
    p = FakeProvider(["ok", "ok"])
    gw = _gw(p, store, cache)
    gw.invoke(REQ)
    gw.invoke(REQ)
    assert len(p.calls) == 1  # 第二次命中缓存
    hit = store.list_recent(1)[0]
    assert hit.cache_hit is True
    assert hit.cost_usd == 0.0  # 命中=省下的开销

def test_store_failure_is_fail_open(tmp_path):
    class BadStore:
        def save(self, record): raise RuntimeError("disk full")
    resp = _gw(FakeProvider(["ok"]), BadStore()).invoke(REQ)
    assert resp.text == "fake-ok"  # 捕获写失败不影响主调用
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gateway_invoke.py -v`
Expected: FAIL — `AttributeError: 'Gateway' object has no attribute 'invoke'`

- [ ] **Step 3: Add invoke() to gateway.py**

```python
# 追加 import 到 agentctl/core/gateway.py 顶部
import logging
import uuid

from agentctl.core.cache import make_key
from agentctl.errors import AllTargetsFailed
from agentctl.models import CallRecord
from agentctl.store.redaction import redact, redact_messages

log = logging.getLogger("agentctl.gateway")
```

```python
# 追加方法到 Gateway 类
    def invoke(self, request: NormalizedRequest) -> NormalizedResponse:
        started = time.monotonic()
        meta = request.metadata or {}
        cache_key = make_key(request) if (self._cache and self._cache_enabled) else None

        if cache_key:
            cached = self._safe_cache_get(cache_key)
            if cached is not None:
                self._capture(request, meta, started, model_resolved=None,
                              attempts=[], resp=cached, status="success",
                              cache_hit=True, cache_key=cache_key, error_type=None)
                return cached

        targets = self._router.resolve(request.model)
        all_attempts: list[Attempt] = []  # 共享:_invoke_target 往里 append,失败也留痕
        for idx, target in enumerate(targets):
            provider = self._providers[target.provider]
            try:
                resp = self._invoke_target(provider, target, request, all_attempts)
                status = "success" if idx == 0 else "fallback_success"
                if cache_key:
                    self._safe_cache_set(cache_key, resp)
                self._capture(request, meta, started, model_resolved=target.name,
                              attempts=all_attempts, resp=resp, status=status,
                              cache_hit=False, cache_key=cache_key, error_type=None)
                return resp
            except TerminalError:
                # 终态(鉴权/参数):不回退,attempts 已由 _invoke_target 记入 all_attempts
                self._capture(request, meta, started, model_resolved=target.name,
                              attempts=all_attempts, resp=None, status="error",
                              cache_hit=False, cache_key=cache_key,
                              error_type="terminal")
                raise
            except RetriableError:
                continue  # 可重试耗尽 → 回退下一目标(attempts 已记入)

        self._capture(request, meta, started, model_resolved=None,
                      attempts=all_attempts, resp=None, status="error",
                      cache_hit=False, cache_key=cache_key, error_type="all_failed")
        raise AllTargetsFailed(f"all targets failed for model {request.model!r}")

    def _safe_cache_get(self, key):
        try:
            return self._cache.get(key)
        except Exception as exc:  # fail-open
            log.warning("cache get failed (fail-open): %s", exc)
            return None

    def _safe_cache_set(self, key, resp) -> None:
        try:
            self._cache.set(key, resp, self._cache_ttl_s)
        except Exception as exc:
            log.warning("cache set failed (fail-open): %s", exc)

    def _capture(self, request, meta, started, *, model_resolved, attempts, resp,
                 status, cache_hit, cache_key, error_type) -> None:
        if self._store is None:
            return
        try:
            cost = None
            if cache_hit:
                cost = 0.0  # 命中缓存=省下真实开销
            elif resp is not None and model_resolved:
                model_only = model_resolved.split("/", 1)[-1]
                cost = self._cost.cost(model_only, resp.input_tokens, resp.output_tokens)
            rec = CallRecord(
                id=str(uuid.uuid4()), ts=time.time(),
                latency_ms=int((time.monotonic() - started) * 1000),
                consumer=meta.get("consumer", "unknown"),
                call_site=meta.get("call_site"), trace_id=meta.get("trace_id"),
                model_requested=request.model,
                params={"max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "has_tools": bool(request.tools)},
                messages_redacted=redact_messages(request.messages),
                prompt_version=meta.get("prompt_version"),
                model_resolved=model_resolved, attempts=attempts,
                output_redacted=redact(resp.text) if resp else None,
                finish_reason=resp.finish_reason if resp else None,
                tool_calls=resp.tool_calls if resp else 0,
                input_tokens=resp.input_tokens if resp else 0,
                output_tokens=resp.output_tokens if resp else 0,
                cost_usd=cost, cache_hit=cache_hit, cache_key=cache_key,
                status=status, error_type=error_type,
            )
            self._store.save(rec)
        except Exception as exc:  # fail-open:捕获绝不打断主调用
            log.warning("capture failed (fail-open): %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gateway_invoke.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run full suite**

Run: `pytest -v`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add agentctl/core/gateway.py tests/test_gateway_invoke.py
git commit -m "feat(gateway): 回退+缓存+成本+捕获全编排(治理 fail-open)"
```

---

### Task 11: GatewayClient(库形态门面)

**Files:**
- Create: `agentctl/client/__init__.py`
- Create: `agentctl/client/gateway_client.py`
- Create: `tests/test_client.py`

**Interfaces:**
- Consumes: `Gateway`、`load_config`、`Router`、`CostMeter`、`MemoryCache`、`SqliteCaptureStore`、`FakeProvider`(可注入)。
- Produces: `validate_routes(routes: dict[str, list[str]], providers: dict[str, Provider]) -> list[str]`(返回问题列表:每条路由目标的 provider 必须已注册;空列表=通过)。`GatewayClient.from_config(config: Config, providers: dict[str, Provider]) -> GatewayClient`(装配前调 `validate_routes`,有问题抛 `ValueError`——启动期 fail-fast);`messages(model: str, messages: list[dict], max_tokens: int = 1024, temperature=None, tools=None, **metadata) -> NormalizedResponse`(metadata 透传 consumer/call_site/trace_id/prompt_version)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client.py
from agentctl.client.gateway_client import GatewayClient
from agentctl.config import Config, RetryConfig
from agentctl.providers.fake import FakeProvider

def test_client_messages_routes_and_returns(tmp_path):
    cfg = Config(routes={"default": ["fake/m"]}, prices={"m": (5.0, 25.0)},
                 cache_enabled=False, db_path=str(tmp_path / "c.db"),
                 retry=RetryConfig(max_attempts_per_target=1, base_backoff_s=0.0, timeout_s=1.0))
    client = GatewayClient.from_config(cfg, providers={"fake": FakeProvider(["ok"])})
    resp = client.messages("default", [{"role": "user", "content": "hi"}], consumer="ops-agent")
    assert resp.text == "fake-ok"

def test_from_config_rejects_unregistered_provider(tmp_path):
    import pytest
    cfg = Config(routes={"default": ["openai/gpt"]}, db_path=str(tmp_path / "c.db"))
    with pytest.raises(ValueError, match="openai"):
        GatewayClient.from_config(cfg, providers={"fake": FakeProvider(["ok"])})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.client.gateway_client'`

- [ ] **Step 3: Write implementation**

```python
# agentctl/client/__init__.py
```

```python
# agentctl/client/gateway_client.py
from __future__ import annotations

from agentctl.config import Config
from agentctl.core.cache import MemoryCache
from agentctl.core.cost import CostMeter
from agentctl.core.gateway import Gateway
from agentctl.core.router import Router
from agentctl.models import NormalizedRequest, NormalizedResponse
from agentctl.models import Target
from agentctl.providers.base import Provider
from agentctl.store.sqlite_store import SqliteCaptureStore


def validate_routes(routes: dict[str, list[str]],
                    providers: dict[str, Provider]) -> list[str]:
    """每条路由目标的 provider 必须已注册。返回问题列表(空=通过)。"""
    problems: list[str] = []
    for logical, targets in routes.items():
        for spec in targets:
            try:
                target = Target.parse(spec)
            except ValueError as exc:
                problems.append(f"route {logical!r}: {exc}")
                continue
            if target.provider not in providers:
                problems.append(
                    f"route {logical!r} → {spec!r}: provider {target.provider!r} 未注册")
    return problems


class GatewayClient:
    """库形态门面:消费者(如 ops-agent)直接调这个。"""

    def __init__(self, gateway: Gateway) -> None:
        self._gateway = gateway

    @classmethod
    def from_config(cls, config: Config, providers: dict[str, Provider]) -> "GatewayClient":
        problems = validate_routes(config.routes, providers)
        if problems:
            raise ValueError("路由配置校验失败:\n  - " + "\n  - ".join(problems))
        gateway = Gateway(
            router=Router(config.routes),
            providers=providers,
            cost_meter=CostMeter(config.prices),
            store=SqliteCaptureStore(config.db_path),
            cache=MemoryCache() if config.cache_enabled else None,
            retry=config.retry,
            cache_enabled=config.cache_enabled,
            cache_ttl_s=config.cache_ttl_s,
        )
        return cls(gateway)

    def messages(self, model: str, messages: list[dict], max_tokens: int = 1024,
                 temperature: float | None = None, tools: list | None = None,
                 **metadata) -> NormalizedResponse:
        request = NormalizedRequest(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, tools=tools, metadata=metadata,
        )
        return self._gateway.invoke(request)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_client.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/client/ tests/test_client.py
git commit -m "feat(client): 库形态 GatewayClient 门面"
```

---

### Task 12: AnthropicProvider(真 provider)

**Files:**
- Create: `agentctl/providers/anthropic_provider.py`
- Create: `tests/test_anthropic_provider.py`

**Interfaces:**
- Consumes: `Provider` 协议、`Target`、`NormalizedRequest`、`NormalizedResponse`、`RetriableError`、`TerminalError`。
- Produces: `AnthropicProvider(client)`(client 可注入,便于测试 mock);`invoke(...)` 把 anthropic SDK 调用映射为 `NormalizedResponse`;按异常类型分类(`overloaded`/`rate_limit`/5xx → `RetriableError`;`authentication`/`invalid_request`/4xx → `TerminalError`)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_anthropic_provider.py
import pytest
from agentctl.providers.anthropic_provider import AnthropicProvider, classify_status
from agentctl.models import Target, NormalizedRequest
from agentctl.errors import RetriableError, TerminalError

REQ = NormalizedRequest(model="default", messages=[{"role": "user", "content": "hi"}], max_tokens=64)
T = Target(provider="anthropic", model="claude-opus-4-8")

class _FakeMessages:
    def __init__(self, behavior):
        self._b = behavior
        self.last_kwargs = None
    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._b == "ok":
            class R:
                content = [type("B", (), {"type": "text", "text": "hello"})()]
                stop_reason = "end_turn"
                usage = type("U", (), {"input_tokens": 7, "output_tokens": 3})()
            return R()
        raise RuntimeError(self._b)

class _FakeClient:
    def __init__(self, behavior): self.messages = _FakeMessages(behavior)

def test_invoke_maps_response():
    p = AnthropicProvider(_FakeClient("ok"))
    resp = p.invoke(T, REQ, timeout=5.0)
    assert resp.text == "hello"
    assert resp.input_tokens == 7
    assert resp.finish_reason == "end_turn"

def test_invoke_passes_timeout_to_sdk():
    client = _FakeClient("ok")
    AnthropicProvider(client).invoke(T, REQ, timeout=12.5)
    assert client.messages.last_kwargs["timeout"] == 12.5

def test_classify_status():
    assert classify_status(429) == "retriable"
    assert classify_status(529) == "retriable"
    assert classify_status(500) == "retriable"
    assert classify_status(401) == "terminal"
    assert classify_status(400) == "terminal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_anthropic_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.providers.anthropic_provider'`

- [ ] **Step 3: Write implementation**

```python
# agentctl/providers/anthropic_provider.py
from __future__ import annotations

from agentctl.errors import RetriableError, TerminalError
from agentctl.models import NormalizedRequest, NormalizedResponse, Target


def classify_status(status: int) -> str:
    """HTTP 状态 → retriable/terminal。429 与 5xx 可重试,其余 4xx 终态。"""
    if status == 429 or status >= 500:
        return "retriable"
    return "terminal"


class AnthropicProvider:
    """把 anthropic SDK 适配为 Provider 协议。client 可注入便于测试。"""

    def __init__(self, client) -> None:
        self._client = client

    def invoke(self, target: Target, request: NormalizedRequest,
               timeout: float) -> NormalizedResponse:
        try:
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
            msg = self._client.messages.create(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is not None and classify_status(status) == "retriable":
                raise RetriableError(str(exc)) from exc
            if status is not None:
                raise TerminalError(str(exc)) from exc
            raise RetriableError(str(exc)) from exc  # 网络/未知 → 可重试

        text = "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", None) == "text")
        tool_calls = sum(1 for b in msg.content if getattr(b, "type", None) == "tool_use")
        return NormalizedResponse(
            text=text, finish_reason=getattr(msg, "stop_reason", None),
            tool_calls=tool_calls,
            input_tokens=getattr(msg.usage, "input_tokens", 0),
            output_tokens=getattr(msg.usage, "output_tokens", 0),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_anthropic_provider.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/providers/anthropic_provider.py tests/test_anthropic_provider.py
git commit -m "feat(providers): AnthropicProvider + 状态分类"
```

---

### Task 13: server 骨架(为"任意 agent 接入"留门)

**Files:**
- Create: `agentctl/server/__init__.py`
- Create: `agentctl/server/app.py`
- Create: `tests/test_server_skeleton.py`

**Interfaces:**
- Consumes: `Gateway`。
- Produces: `build_server(gateway: Gateway)`;本期未实现,调用抛 `NotImplementedError("proxy server reserved for a later sub-project")`。仅锁定接入缝,后续子项目落地 HTTP 代理。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_skeleton.py
import pytest
from agentctl.server.app import build_server

def test_server_reserved_not_implemented():
    with pytest.raises(NotImplementedError):
        build_server(gateway=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server_skeleton.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.server.app'`

- [ ] **Step 3: Write implementation**

```python
# agentctl/server/__init__.py
```

```python
# agentctl/server/app.py
from __future__ import annotations


def build_server(gateway):
    """代理形态(HTTP,Anthropic/OpenAI 兼容)预留入口。

    本期不实现:第一刀聚焦库形态 + 捕获。代理形态留给后续子项目,
    届时让任意语言/任意 agent 改 base_url 即可接入同一治理引擎。
    """
    raise NotImplementedError("proxy server reserved for a later sub-project")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_server_skeleton.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add agentctl/server/ tests/test_server_skeleton.py
git commit -m "feat(server): 代理形态骨架(留口,本期 NotImplemented)"
```

---

### Task 14: CLI(captures / cost / doctor)

**Files:**
- Create: `agentctl/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config`、`SqliteCaptureStore`。
- Produces: `main(argv: list[str] | None = None) -> int`;子命令 `captures --limit N`(打印近期记录)、`cost`(打印成本汇总)、`doctor`(检查配置:路由非空、每条目标可解析为 `provider/model` 且 provider ∈ 已知内建集 `{"anthropic"}`、prod profile 下价表非空)。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from agentctl.cli import main
from agentctl.config import Config
from agentctl.store.sqlite_store import SqliteCaptureStore
from agentctl.models import CallRecord

def test_cost_command_reports(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "c.db")
    SqliteCaptureStore(db).save(CallRecord(id="a", consumer="t", status="success",
                                           input_tokens=10, output_tokens=5, cost_usd=0.01))
    monkeypatch.setattr("agentctl.cli.load_config",
                        lambda path=None: Config(db_path=db))
    rc = main(["cost"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.01" in out
    assert "calls" in out.lower()

def test_doctor_flags_empty_routes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("agentctl.cli.load_config",
                        lambda path=None: Config(routes={}, db_path=str(tmp_path / "c.db")))
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "routes" in out.lower()

def test_doctor_flags_unknown_provider(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("agentctl.cli.load_config",
                        lambda path=None: Config(routes={"default": ["openai/gpt"]},
                                                 db_path=str(tmp_path / "c.db")))
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "openai" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentctl.cli'`

- [ ] **Step 3: Write implementation**

```python
# agentctl/cli.py
from __future__ import annotations

import argparse
import json

from agentctl.config import load_config
from agentctl.models import Target
from agentctl.store.sqlite_store import SqliteCaptureStore

KNOWN_PROVIDERS = {"anthropic"}  # 本期内建;新增 provider 时同步扩充


def _cmd_captures(cfg, args) -> int:
    store = SqliteCaptureStore(cfg.db_path)
    for rec in store.list_recent(args.limit):
        print(f"{rec.id[:8]} {rec.status:16} {rec.model_resolved or '-':28} "
              f"in={rec.input_tokens} out={rec.output_tokens} cost={rec.cost_usd}")
    return 0


def _cmd_cost(cfg, args) -> int:
    summary = SqliteCaptureStore(cfg.db_path).cost_summary()
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_doctor(cfg, args) -> int:
    problems = []
    if not cfg.routes:
        problems.append("routes 为空:至少配一个逻辑模型 → 目标链")
    for logical, targets in cfg.routes.items():
        for spec in targets:
            try:
                target = Target.parse(spec)
            except ValueError as exc:
                problems.append(f"route {logical!r}: {exc}")
                continue
            if target.provider not in KNOWN_PROVIDERS:
                problems.append(
                    f"route {logical!r} → {spec!r}: 未知 provider {target.provider!r}"
                    f"(已知 {sorted(KNOWN_PROVIDERS)})")
    if cfg.profile == "prod" and not cfg.prices:
        problems.append("prod profile 下 prices 为空:成本将全为 None")
    if problems:
        for p in problems:
            print("FAIL:", p)
        return 1
    print("OK: 配置自检通过")
    return 0


_COMMANDS = {"captures": _cmd_captures, "cost": _cmd_cost, "doctor": _cmd_doctor}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentctl")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p_cap = sub.add_parser("captures")
    p_cap.add_argument("--limit", type=int, default=20)
    sub.add_parser("cost")
    sub.add_parser("doctor")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return _COMMANDS[args.command](cfg, args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run full suite + lint**

Run: `pytest -v && ruff check . && ruff format --check .`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add agentctl/cli.py tests/test_cli.py
git commit -m "feat(cli): captures/cost/doctor 子命令"
```

---

### Task 15: ops-agent 接入(首个真实消费者)

**Files:**
- Modify: `../ops-agent/ops_agent/llm.py`(改为可选走 agentctl 网关)
- Modify: `../ops-agent/.env.example`(加 `OPS_USE_GATEWAY` 说明)
- Create: `../ops-agent/tests/test_gateway_integration.py`

**前置:** 在 ops-agent 环境装 agentctl:`pip install -e ../agentctl`。

**Interfaces:**
- Consumes: `agentctl.client.gateway_client.GatewayClient`、`agentctl.config.load_config`、`agentctl.providers.anthropic_provider.AnthropicProvider`。
- Produces: ops-agent 的 `llm.py` 暴露一个工厂,当 `OPS_USE_GATEWAY=true` 时所有 LLM 调用经 agentctl(治理 + 捕获);否则保持原 Anthropic 直连(零行为变化)。

- [ ] **Step 1: 先看现有 llm.py 结构**

Run: `sed -n '1,80p' ../ops-agent/ops_agent/llm.py`
Expected: 看到现有"共享 Anthropic 客户端工厂"——确认改造点(把"返回 client"的工厂加一条网关分支)。

- [ ] **Step 2: Write the failing test**

```python
# ../ops-agent/tests/test_gateway_integration.py
import os
from ops_agent import llm

def test_gateway_disabled_returns_native(monkeypatch):
    monkeypatch.delenv("OPS_USE_GATEWAY", raising=False)
    # 关闭网关时,工厂返回原生路径(behaves as before)
    assert llm.gateway_enabled() is False

def test_gateway_enabled_flag(monkeypatch):
    monkeypatch.setenv("OPS_USE_GATEWAY", "true")
    assert llm.gateway_enabled() is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ../ops-agent && pytest tests/test_gateway_integration.py -v`
Expected: FAIL — `AttributeError: module 'ops_agent.llm' has no attribute 'gateway_enabled'`

- [ ] **Step 4: 在 ops-agent/ops_agent/llm.py 增加网关分支**

```python
# 追加到 ../ops-agent/ops_agent/llm.py
import os


def gateway_enabled() -> bool:
    return os.getenv("OPS_USE_GATEWAY", "").lower() == "true"


def build_gateway_client():
    """当 OPS_USE_GATEWAY=true:构造 agentctl 库形态 client(治理 + 捕获)。"""
    from agentctl.client.gateway_client import GatewayClient
    from agentctl.config import load_config
    from agentctl.providers.anthropic_provider import AnthropicProvider
    import anthropic

    cfg = load_config(os.getenv("AGENTCTL_CONFIG"))
    providers = {"anthropic": AnthropicProvider(anthropic.Anthropic())}
    return GatewayClient.from_config(cfg, providers)
```

> 说明:本步只新增"开关 + 工厂",不改动 ops-agent 既有直连路径,保证 `OPS_USE_GATEWAY` 未开时**零行为变化**(其余调用点逐步迁移到 `messages()` 留待后续小步)。

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ../ops-agent && pytest tests/test_gateway_integration.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: 跑 ops-agent 全量离线测试,确认零回归**

Run: `cd ../ops-agent && pytest -q`
Expected: 原有测试全 PASS(网关默认关,行为不变)

- [ ] **Step 7: 更新 .env.example**

```bash
# 追加到 ../ops-agent/.env.example
# 走 agentctl 网关(路由/回退/成本/缓存 + 全量捕获);默认 false=原生直连
OPS_USE_GATEWAY=false
AGENTCTL_CONFIG=../agentctl/agentctl.example.yaml
```

- [ ] **Step 8: Commit(在 ops-agent 仓库)**

```bash
cd ../ops-agent
git add ops_agent/llm.py .env.example tests/test_gateway_integration.py
git commit -m "feat: 可选经 agentctl 网关调用(治理+捕获,默认关零回归)"
```

---

## 完成标准(Definition of Done)

- agentctl:Tasks 1-14 全绿(`pytest && ruff check . && ruff format --check .`)。
- 库形态可用:`GatewayClient.messages()` 走路由/回退/超时重试/成本/缓存,每次调用落 `CallRecord`。
- `agentctl cost` / `captures` / `doctor` 可跑。
- ops-agent:`OPS_USE_GATEWAY=true` 时调用经网关并被捕获;关闭时零回归。
- server 形态留口(NotImplemented),为后续子项目铺路。

## 自检结果(spec 覆盖核对)

- §2 范围五件事 → Tasks 3/9/10(路由/回退/重试)、5(成本)、8(缓存)、7+10(捕获);限流/eval/大盘/prompt 版本/语义缓存/**流式**均按非目标未纳入。✓
- §2 网关=单次调用边界 → Task 10(一次 invoke=一次模型调用,tool_calls 只计数不串多步)。✓
- §3 CallRecord 全字段 + 缓存命中 cost=0 → Task 2 定义 + Task 10 落库(prompt_version 经 metadata 预留;`_capture` cache_hit→0.0)。✓
- §4 存储 SQLite + 脱敏 → Tasks 6/7/10。✓
- §5 容错 fail-open + 错误透传 → Task 10(`_safe_cache_*`/`_capture` fail-open;TerminalError 透传)。✓
- §5 并发/线程安全 → Task 7(WAL+Lock+多线程测)/ Task 8(MemoryCache Lock)。✓
- §5 启动期校验(route↔provider)→ Task 11(`validate_routes` + `from_config` 抛错)/ Task 14(`doctor`)。✓
- §6 ops-agent 仅改 llm.py → Task 15。✓
- §7 失败目标 attempts 留痕(修 bug)→ Task 9(共享 attempts 列表)+ Task 10(`test_all_targets_fail_records_error_with_all_attempts`)。✓
- §8 离线 FakeProvider TDD → Task 4 起全程。✓
- 超时真正生效 → Task 12(`timeout` 透传 anthropic SDK + 测)。✓
- 接入形态 C(库 + server 留口)→ Tasks 11/13。✓
