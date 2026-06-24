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


def redact_value(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    return value


def redact_messages(messages: list[dict]) -> list[dict]:
    return [redact_value(m) for m in messages]
