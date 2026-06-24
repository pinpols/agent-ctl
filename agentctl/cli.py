from __future__ import annotations

import argparse
import json

from agentctl.config import load_config
from agentctl.models import Target
from agentctl.store.sqlite_store import SqliteCaptureStore

# KNOWN_PROVIDERS 是静态 lint 集合:仅列举 agentctl 当前随包附带内建适配器的 provider 名。
# 它不是运行时接线检查——运行时权威校验由 GatewayClient.from_config 中的
# validate_routes 负责。doctor 命令使用此集合给出"无内建适配器"的早期提示。
KNOWN_PROVIDERS = {"anthropic"}  # 本期内建;新增 provider 时同步扩充


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


_COMMANDS = {"captures": _cmd_captures, "cost": _cmd_cost, "doctor": _cmd_doctor}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentctl")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p_cap = sub.add_parser("captures")
    p_cap.add_argument("--limit", type=int, default=20)
    sub.add_parser("cost")
    sub.add_parser("doctor")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return _COMMANDS[args.command](cfg, args)
