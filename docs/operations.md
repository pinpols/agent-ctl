# Production Operations Runbook

This runbook covers single-instance production-style operation. Distributed state
for budget, cache, circuit breaker, and capture storage is intentionally outside
the current release.

## Preflight

```bash
agent-ctl --config agent_ctl.yaml doctor
agent-ctl config-schema --out /tmp/agent_ctl.schema.json
```

Required checks:

- At least one provider API key is present.
- Non-local `serve` uses `--api-token`.
- `profile: prod` has a non-empty `prices` table.
- Route fallback chains have expected provider capabilities.

## Run With Docker Compose

```bash
export AGENT_CTL_API_TOKEN="$(openssl rand -hex 24)"
export ANTHROPIC_API_KEY=...
docker compose up --build -d
curl -H "Authorization: Bearer $AGENT_CTL_API_TOKEN" http://127.0.0.1:8400/healthz
```

The compose template intentionally fails fast when `AGENT_CTL_API_TOKEN` is unset.

## Observability

- Health: `GET /healthz`
- Metrics: `GET /metrics`
- Capture inspection: `agent-ctl --config agent_ctl.yaml captures --limit 20`
- Cost summary: `agent-ctl --config agent_ctl.yaml cost --group-by model`
- Export traces: `agent-ctl --config agent_ctl.yaml export --out traces.jsonl`

## Release Checklist

1. Update `CHANGELOG.md`.
2. Confirm version in `pyproject.toml` and `agent_ctl/__init__.py`.
3. Run `pytest`, `ruff`, and `mypy`.
4. Run `agent-ctl --config agent-ctl.example.yaml doctor`.
5. Build image: `docker build -t agent-ctl:<version> .`.
6. Tag source: `git tag v<version>`.

## Rollback

Capture data lives in `db_path`. Before rollback, copy the SQLite file or mounted
volume. Config migrations are backward-compatible within the same major version;
if rollback crosses a major version, follow that release's migration notes.
