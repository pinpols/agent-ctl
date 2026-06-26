from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, NonNegativeFloat


class RetryConfig(BaseModel):
    max_attempts_per_target: int = Field(default=2, ge=1)
    base_backoff_s: float = Field(default=0.2, ge=0.0)
    timeout_s: float = Field(default=60.0, gt=0.0)
    jitter_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class Config(BaseModel):
    routes: dict[str, list[str]] = {"default": ["anthropic/claude-sonnet-4-6"]}
    # 裸模型名 → "provider/model"(OpenAI 兼容 server 用,如 deepseek-chat→deepseek/deepseek-chat)
    model_aliases: dict[str, str] = {}
    prices: dict[str, tuple[NonNegativeFloat, NonNegativeFloat]] = {}
    cache_enabled: bool = True
    cache_ttl_s: int = Field(default=600, ge=0)
    cache_tool_responses: bool = False
    # 缓存条目硬上界(LRU 淘汰),防长驻 server 内存只增不减。0=不限(不建议)。
    cache_max_entries: int = Field(default=10_000, ge=0)
    # 捕获落库移出请求主路径(后台线程 + 有界队列)。关闭则同步落库(测试/确定性可读)。
    capture_async: bool = True
    # 单次调用墙钟总预算(秒);跨"重试×回退×单目标超时"封顶,0=不封顶。
    request_deadline_s: float = Field(default=120.0, ge=0.0)
    # 成本预算闸(进程内累计,USD):per-consumer 上限 + 全局上限。空=不限。
    budgets: dict[str, NonNegativeFloat] = {}
    budget_global: NonNegativeFloat | None = None
    # 熔断:某 provider 连续失败达阈值则开路冷却,期间回退跳过它。0=关闭。
    circuit_failure_threshold: int = Field(default=5, ge=0)
    circuit_cooldown_s: float = Field(default=30.0, ge=0.0)
    # server 是否允许请求用 "provider/model" 直连未在 routes/aliases 登记的目标。
    # 默认禁(成本治理):否则调用方可绕过路由白名单调任意已注册 provider 的任意模型。
    allow_direct_models: bool = False
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
