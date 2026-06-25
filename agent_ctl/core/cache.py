# agent_ctl/core/cache.py
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Protocol

from agent_ctl.models import NormalizedRequest, NormalizedResponse


def make_key(request: NormalizedRequest) -> str:
    # 缓存键必须涵盖一切会改变响应的入参。tools/system/tool_choice 直接决定响应形状
    # (纯文本 vs 强制 tool_use),漏掉它们会让不同形状的请求命中同一缓存 → 返回错误结构。
    payload = json.dumps(
        {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "tools": request.tools,
            "system": request.system,
            "tool_choice": request.tool_choice,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache(Protocol):
    def get(self, key: str) -> NormalizedResponse | None: ...
    def set(self, key: str, resp: NormalizedResponse, ttl_s: int) -> None: ...


class MemoryCache:
    """进程内精确匹配缓存 + TTL,**有界 LRU**。加锁保证多线程并发调用安全。

    max_entries 硬上界 + 最久未用淘汰:LLM 请求 key 高度发散(过期项几乎不会被再次
    命中而惰性删除),无界字典会在长驻 server 里只增不减。set 时超界即从最旧端淘汰,
    O(1) 摊还,内存被钉死在 max_entries。
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._data: OrderedDict[str, tuple[float, NormalizedResponse]] = OrderedDict()
        self._max = max_entries
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
            self._data.move_to_end(key)  # LRU:命中即"最近用"
            return resp.model_copy(deep=True)

    def set(self, key: str, resp: NormalizedResponse, ttl_s: int) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl_s, resp.model_copy(deep=True))
            self._data.move_to_end(key)
            while self._max > 0 and len(self._data) > self._max:
                self._data.popitem(last=False)  # 淘汰最久未用

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
