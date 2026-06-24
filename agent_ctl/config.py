from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    max_attempts_per_target: int = Field(default=2, ge=1)
    base_backoff_s: float = Field(default=0.2, ge=0.0)
    timeout_s: float = Field(default=60.0, gt=0.0)
    jitter_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class Config(BaseModel):
    routes: dict[str, list[str]] = {"default": ["anthropic/claude-sonnet-4-6"]}
    # 裸模型名 → "provider/model"(OpenAI 兼容 server 用,如 deepseek-chat→deepseek/deepseek-chat)
    model_aliases: dict[str, str] = {}
    prices: dict[str, tuple[float, float]] = {}
    cache_enabled: bool = True
    cache_ttl_s: int = 600
    cache_tool_responses: bool = False
    profile: str = "dev"
    db_path: str = ".agent_ctl/capture.db"
    retry: RetryConfig = RetryConfig()


def load_config(path: str | None = None) -> Config:
    """从 yaml 读配置;path 为 None 时尝试 ./agent_ctl.yaml,无则用默认。env 不覆盖结构,仅 profile。"""
    data: dict = {}
    candidate = path or ("agent_ctl.yaml" if Path("agent_ctl.yaml").exists() else None)
    if candidate:
        with open(candidate, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    cfg = Config(**data)
    if env_profile := os.getenv("AGENT_CTL_PROFILE"):
        cfg = cfg.model_copy(update={"profile": env_profile})
    return cfg
