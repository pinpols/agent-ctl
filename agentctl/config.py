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
