from __future__ import annotations

import argparse
import json
import sys
import time

from agent_ctl import __version__
from agent_ctl.config import load_config
from agent_ctl.models import Target
from agent_ctl.providers.catalog import PROVIDER_CATALOG, provider_capabilities
from agent_ctl.store.sqlite_store import SqliteCaptureStore

# KNOWN_PROVIDERS 是静态 lint 集合:仅列举 agent_ctl 当前随包附带内建适配器的 provider 名。
# 它不是运行时接线检查——运行时权威校验由 GatewayClient.from_config 中的
# validate_routes 负责。doctor 命令使用此集合给出"无内建适配器"的早期提示。
KNOWN_PROVIDERS = set(PROVIDER_CATALOG)  # 运行时以注入的 providers 为准


def _cmd_captures(cfg, args) -> int:
    since = _parse_since(args.since)
    with SqliteCaptureStore(cfg.db_path) as store:
        records = store.list_recent(
            args.limit,
            consumer=args.consumer,
            status=args.status,
            model=args.model,
            since=since,
        )
    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
        return 0
    for rec in records:
        print(
            f"{rec.id[:8]} {rec.status:16} {rec.consumer:16} "
            f"{rec.model_resolved or rec.model_requested or '-':28} "
            f"in={rec.input_tokens} out={rec.output_tokens} cost={rec.cost_usd}"
        )
    return 0


def _cmd_cost(cfg, args) -> int:
    since = _parse_since(args.since)
    with SqliteCaptureStore(cfg.db_path) as store:
        summary = store.cost_summary(
            consumer=args.consumer,
            status=args.status,
            model=args.model,
            since=since,
            group_by=args.group_by,
        )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_export(cfg, args) -> int:
    """流式导出捕获为 JSONL(逐行一条 CallRecord),供 eval/replay。stdout 保持纯净。"""
    since = _parse_since(args.since)
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    n = 0
    try:
        with SqliteCaptureStore(cfg.db_path) as store:
            for rec in store.iter_all(
                consumer=args.consumer,
                status=args.status,
                model=args.model,
                since=since,
                ascending=True,  # 时序,replay 友好
            ):
                out.write(
                    json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n"
                )
                n += 1
    finally:
        if args.out:
            out.close()
    print(f"exported {n} records", file=sys.stderr)  # 计数走 stderr,不污染 JSONL
    return 0


def _cmd_config_schema(cfg, args) -> int:
    from agent_ctl.config import Config

    schema = Config.model_json_schema()
    text = json.dumps(schema, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


def _cmd_doctor(cfg, args) -> int:
    problems: list[str] = []
    warnings: list[str] = []
    cap_lines: list[str] = []
    if not cfg.routes:
        problems.append("routes 为空:至少配一个逻辑模型 → 目标链")
    for logical, targets in cfg.routes.items():
        parsed: list[tuple[str, set[str]]] = []
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
                continue
            parsed.append((spec, provider_capabilities(target.provider)))
        cap_lines.append(f"  route {logical!r}:")
        for spec, caps in parsed:
            cap_lines.append(f"    {spec:34} {','.join(sorted(caps)) or '(none)'}")
        # 回退能力一致性:某能力部分目标支持、部分不支持 → 该能力的回退会静默失败
        if len(parsed) > 1:
            cap_sets = [c for _, c in parsed]
            union: set[str] = set().union(*cap_sets)
            common: set[str] = set(cap_sets[0])
            for c in cap_sets[1:]:
                common &= c
            for cap in sorted(union - common):
                lack = [s for s, c in parsed if cap not in c]
                warnings.append(
                    f"route {logical!r}: 能力 {cap!r} 在目标间不一致,回退到 {lack} 时该类请求会失败"
                )
    if cfg.profile == "prod" and not cfg.prices:
        problems.append("prod profile 下 prices 为空:成本将全为 None")
    if problems:
        for p in problems:
            print("FAIL:", p)
        return 1
    for w in warnings:
        print("WARN:", w)
    print("能力矩阵(chat/stream/embed/tools):")
    for line in cap_lines:
        print(line)
    print("OK: 配置自检通过")
    return 0


def _cmd_serve(cfg, args) -> int:
    """起 OpenAI 兼容网关:按目录构造 providers(仅有 key 的)+ Gateway + FastAPI server。"""
    import uvicorn

    from agent_ctl.core.budget import BudgetGuard
    from agent_ctl.core.cache import MemoryCache
    from agent_ctl.core.circuit import CircuitBreaker
    from agent_ctl.core.cost import CostMeter
    from agent_ctl.core.gateway import Gateway
    from agent_ctl.core.router import Router
    from agent_ctl.client.gateway_client import build_store
    from agent_ctl.providers.catalog import available_providers, build_providers
    from agent_ctl.server.app import build_server

    if args.host not in {"127.0.0.1", "localhost", "::1"} and (
        not args.api_token or args.api_token == "change-me"
    ):
        print("FAIL: non-local serve requires a non-default --api-token")
        return 1
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
        store=build_store(cfg),
        cache=MemoryCache(cfg.cache_max_entries) if cfg.cache_enabled else None,
        retry=cfg.retry,
        cache_enabled=cfg.cache_enabled,
        cache_ttl_s=cfg.cache_ttl_s,
        cache_tool_responses=cfg.cache_tool_responses,
        circuit=CircuitBreaker(cfg.circuit_failure_threshold, cfg.circuit_cooldown_s),
        request_deadline_s=cfg.request_deadline_s,
        budget=BudgetGuard(cfg.budgets, cfg.budget_global),
    )
    app = build_server(
        gateway,
        models=sorted(cfg.model_aliases) or avail,
        api_token=args.api_token,
        max_request_bytes=args.max_request_bytes,
        rate_limit_per_minute=args.rate_limit_per_minute,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_version(cfg, args) -> int:
    print(__version__)
    return 0


def _parse_since(value: str | None) -> float | None:
    if value is None:
        return None
    if value.endswith("h"):
        return time.time() - float(value[:-1]) * 3600
    if value.endswith("d"):
        return time.time() - float(value[:-1]) * 86400
    return float(value)


_COMMANDS = {
    "captures": _cmd_captures,
    "cost": _cmd_cost,
    "export": _cmd_export,
    "config-schema": _cmd_config_schema,
    "doctor": _cmd_doctor,
    "serve": _cmd_serve,
    "version": _cmd_version,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-ctl")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p_cap = sub.add_parser("captures")
    p_cap.add_argument("--limit", type=int, default=20)
    p_cap.add_argument("--consumer")
    p_cap.add_argument("--status")
    p_cap.add_argument("--model")
    p_cap.add_argument("--since", help="Unix timestamp, Nh, or Nd")
    p_cap.add_argument("--json", action="store_true")
    p_cost = sub.add_parser("cost")
    p_cost.add_argument("--consumer")
    p_cost.add_argument("--status")
    p_cost.add_argument("--model")
    p_cost.add_argument("--since", help="Unix timestamp, Nh, or Nd")
    p_cost.add_argument("--group-by", choices=["model", "consumer", "status", "day"])
    p_exp = sub.add_parser("export", help="流式导出捕获为 JSONL(eval/replay)")
    p_exp.add_argument("--out", help="输出文件;省略则写 stdout")
    p_exp.add_argument("--consumer")
    p_exp.add_argument("--status")
    p_exp.add_argument("--model")
    p_exp.add_argument("--since", help="Unix timestamp, Nh, or Nd")
    p_schema = sub.add_parser("config-schema", help="输出 Config JSON Schema")
    p_schema.add_argument("--out", help="输出文件;省略则写 stdout")
    sub.add_parser("doctor")
    sub.add_parser("version")
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8400)
    p_serve.add_argument("--api-token", default=None)
    p_serve.add_argument("--max-request-bytes", type=int, default=1_000_000)
    p_serve.add_argument("--rate-limit-per-minute", type=int, default=120)
    args = parser.parse_args(argv)
    cfg = (
        None
        if args.command in {"version", "config-schema"}
        else load_config(args.config)
    )
    return _COMMANDS[args.command](cfg, args)
