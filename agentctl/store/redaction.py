from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{8,}"),  # api keys
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.]+"),  # bearer
    re.compile(r"eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"),  # JWT
    re.compile(r"://[^:/@\s]+:([^@/\s]+)@"),  # DSN password (group 1)
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"\b1[3-9]\d{9}\b"),  # CN mobile
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
