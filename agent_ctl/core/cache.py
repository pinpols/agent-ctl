# agent_ctl/core/cache.py
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Protocol

from agent_ctl.models import NormalizedRequest, NormalizedResponse


def make_key(request: NormalizedRequest) -> str:
    payload = json.dumps(
        {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        },
        sort_keys=True,
        ensure_ascii=False,
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
