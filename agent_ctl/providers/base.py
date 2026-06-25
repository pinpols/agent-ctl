from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_ctl.models import (
    EmbeddingResponse,
    NormalizedRequest,
    NormalizedResponse,
    Target,
)


class Provider(Protocol):
    def invoke(
        self, target: Target, request: NormalizedRequest, timeout: float
    ) -> NormalizedResponse: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """可选能力:并非所有 provider 都支持 embeddings(如 Anthropic 无此 API)。

    runtime_checkable → 网关用 isinstance 判定某 provider 是否能 embed,
    不能的目标在 embed 回退链里被跳过(留痕 no_embed)。
    """

    def embed(
        self, target: Target, inputs: list[str], timeout: float
    ) -> EmbeddingResponse: ...
