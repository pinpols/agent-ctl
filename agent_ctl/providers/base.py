from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from agent_ctl.models import (
    EmbeddingResponse,
    NormalizedRequest,
    NormalizedResponse,
    StreamChunk,
    Target,
)


class Provider(Protocol):
    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse: ...


@runtime_checkable
class StreamingProvider(Protocol):
    """可选能力:原生流式。stream() 逐块产出文本增量,末块 done=True 带最终计量。

    无此能力的 provider 在 invoke_stream 里退化为缓冲式(跑非流式再切块)。
    开流前的失败可回退下一目标;一旦首块已出则提交该目标,不再回退。
    """

    def stream(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> Iterator[StreamChunk]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """可选能力:并非所有 provider 都支持 embeddings(如 Anthropic 无此 API)。

    runtime_checkable → 网关用 isinstance 判定某 provider 是否能 embed,
    不能的目标在 embed 回退链里被跳过(留痕 no_embed)。
    """

    def embed(
        self, target: Target, inputs: list[str], timeout: float
    ) -> EmbeddingResponse: ...
