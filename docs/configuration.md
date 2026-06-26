# Configuration Schema and Migration

`agent-ctl` loads YAML into `agent_ctl.config.Config`.

Generate the machine-readable JSON Schema:

```bash
agent-ctl config-schema --out docs/config.schema.json
```

Validate operational config before deployment:

```bash
agent-ctl --config agent_ctl.yaml doctor
```

## Compatibility Policy

- New optional fields are backward-compatible and get defaults in `Config`.
- Required field additions or renamed fields require a minor migration guide entry.
- Removed fields require a major version bump.
- Runtime capture schema migrations must be idempotent and covered by tests against legacy tables.

## Config Migration Strategy

1. Add the new field to `Config` with a conservative default.
2. Add it to `agent-ctl.example.yaml`.
3. Add or update `doctor` checks when a production misconfiguration is likely.
4. Update this document and `CHANGELOG.md`.
5. Add tests for default loading, YAML loading, and invalid values.

## Current Operational Fields

- `routes`: logical model names to ordered fallback chains.
- `model_aliases`: OpenAI-compatible model names to `provider/model` targets.
- `prices`: non-negative cost table in USD per 1M input/output tokens.
- `cache_enabled`, `cache_ttl_s`, `cache_tool_responses`, `cache_max_entries`: response cache controls; numeric limits are non-negative.
- `capture_async`: enable background capture writes.
- `request_deadline_s`: non-negative wall-clock deadline per request; `0` disables.
- `budgets`, `budget_global`: non-negative in-process USD budget caps.
- `circuit_failure_threshold`, `circuit_cooldown_s`: provider circuit breaker controls; `0` disables.
- `retry`: bounded retry behavior per target.

In `profile: prod`, each configured route and alias must have either a
`provider/model` price key or a bare model price key. Unknown configured prices
fail closed before a provider call is attempted.
