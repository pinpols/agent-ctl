from __future__ import annotations

import argparse
import json

from agent_ctl.config import load_config
from agent_ctl.models import Target
from agent_ctl.store.sqlite_store import SqliteCaptureStore

# KNOWN_PROVIDERS 是静态 lint 集合:仅列举 agent_ctl 当前随包附带内建适配器的 provider 名。
# 它不是运行时接线检查——运行时权威校验由 GatewayClient.from_config 中的
# validate_routes 负责。doctor 命令使用此集合给出"无内建适配器"的早期提示。
KNOWN_PROVIDERS = {
    "anthropic",
    "openai",
}  # 内建适配器(静态校验用);运行时以注入的 providers 为准


def _cmd_captures(cfg, args) -> int:
    store = SqliteCaptureStore(cfg.db_path)
    for rec in store.list_recent(args.limit):
        print(
            f"{rec.id[:8]} {rec.status:16} {rec.model_resolved or '-':28} "
            f"in={rec.input_tokens} out={rec.output_tokens} cost={rec.cost_usd}"
        )
    return 0


def _cmd_cost(cfg, args) -> int:
    summary = SqliteCaptureStore(cfg.db_path).cost_summary()
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_doctor(cfg, args) -> int:
    problems = []
    if not cfg.routes:
        problems.append("routes 为空:至少配一个逻辑模型 → 目标链")
    for logical, targets in cfg.routes.items():
        for spec in targets:
            try:
                target = Target.parse(spec)
            except ValueError as exc:
                problems.append(f"route {logical!r}: {exc}")
                continue
            if target.provider not in KNOWN_PROVIDERS:
                problems.append(
                    f"route {logical!r} → {spec!r}: provider {target.provider!r} 无内建适配器 "
                    f"(内建适配器: {sorted(KNOWN_PROVIDERS)})"
                )
    if cfg.profile == "prod" and not cfg.prices:
        problems.append("prod profile 下 prices 为空:成本将全为 None")
    if problems:
        for p in problems:
            print("FAIL:", p)
        return 1
    print("OK: 配置自检通过")
    return 0


def _cmd_serve(cfg, args) -> int:
    """起 OpenAI 兼容网关:按目录构造 providers(仅有 key 的)+ Gateway + FastAPI server。"""
    import uvicorn

    from agent_ctl.core.cache import MemoryCache
    from agent_ctl.core.cost import CostMeter
    from agent_ctl.core.gateway import Gateway
    from agent_ctl.core.router import Router
    from agent_ctl.providers.catalog import available_providers, build_providers
    from agent_ctl.server.app import build_server
    from agent_ctl.store.sqlite_store import SqliteCaptureStore

    avail = available_providers()
    if not avail:
        print(
            "FAIL: 没有任何 provider 的 api key(设 ANTHROPIC_API_KEY / OPENAI_API_KEY / "
            "DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / GLM_API_KEY 之一)"
        )
        return 1
    print(f"已启用 provider: {avail}")
    providers = build_providers()
    gateway = Gateway(
        router=Router(cfg.routes, cfg.model_aliases),
        providers=providers,
        cost_meter=CostMeter(cfg.prices),
        store=SqliteCaptureStore(cfg.db_path),
        cache=MemoryCache() if cfg.cache_enabled else None,
        retry=cfg.retry,
        cache_enabled=cfg.cache_enabled,
        cache_ttl_s=cfg.cache_ttl_s,
    )
    app = build_server(gateway, models=sorted(cfg.model_aliases) or avail)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


_COMMANDS = {
    "captures": _cmd_captures,
    "cost": _cmd_cost,
    "doctor": _cmd_doctor,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-ctl")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p_cap = sub.add_parser("captures")
    p_cap.add_argument("--limit", type=int, default=20)
    sub.add_parser("cost")
    sub.add_parser("doctor")
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8400)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return _COMMANDS[args.command](cfg, args)
