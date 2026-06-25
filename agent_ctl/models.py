# agent_ctl/models.py
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
    # 工具调用型消费者(如 ops-agent)需要这两项:system 提示 + 强制工具选择。
    # 缺省 None=不传,对纯文本路由消费者完全向后兼容。
    system: str | None = None
    tool_choice: dict | None = None
    metadata: dict = {}


class NormalizedResponse(BaseModel):
    text: str
    finish_reason: str | None = None
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict | None = None


class StreamChunk(BaseModel):
    """流式增量。text=本次增量文本;done=True 的终块携带 finish_reason + 最终 token 计量。"""

    text: str = ""
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    done: bool = False


class EmbeddingResponse(BaseModel):
    vectors: list[list[float]]
    input_tokens: int = 0
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
    error_message_redacted: str | None = None
    last_error: str | None = None
