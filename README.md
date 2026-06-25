# agent-ctl

`agent-ctl` is a local AgentOps gateway for LLM calls. It gives agents one governed path for model routing, fallback, retry, cost accounting, response caching, and redacted call capture.

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,anthropic,openai,server]"
```

For development, the expected checks are:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy agent_ctl
```

## Configure

Copy the example and edit routes, prices, and aliases:

```bash
cp agent-ctl.example.yaml agent_ctl.yaml
```

Provider credentials are read from environment variables:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...
export GLM_API_KEY=...
```

Run a config check:

```bash
.venv/bin/agent-ctl --config agent_ctl.yaml doctor
```

## CLI

Recent captures:

```bash
.venv/bin/agent-ctl --config agent_ctl.yaml captures --limit 20 --status error --json
```

Cost summary:

```bash
.venv/bin/agent-ctl --config agent_ctl.yaml cost --group-by model
```

Available `cost --group-by` values are `model`, `consumer`, `status`, and `day`.

Streaming export of captures to JSONL (time-ordered, for eval/replay; streams via a separate read connection without holding the write lock):

```bash
.venv/bin/agent-ctl --config agent_ctl.yaml export --consumer ops --since 7d > traces.jsonl
```

`doctor` reports a per-route capability matrix (chat/stream/embed/tools, derived statically per adapter) and warns when a fallback chain mixes targets with different capabilities (e.g. an `embed` request would fail when it falls back to a provider with no embeddings API).

## Library Usage

```python
from agent_ctl.client.gateway_client import GatewayClient
from agent_ctl.config import load_config
from agent_ctl.providers.catalog import build_providers

client = GatewayClient.from_config(load_config("agent_ctl.yaml"), build_providers())
resp = client.messages(
    "default",
    [{"role": "user", "content": "hello"}],
    consumer="my-agent",
)
print(resp.text)
```

## OpenAI-Compatible Server

By default the server binds to localhost. Pass an API token for any non-local use.

```bash
.venv/bin/agent-ctl --config agent_ctl.yaml serve --host 127.0.0.1 --port 8400 --api-token "$AGENT_CTL_API_TOKEN"
```

Then call:

```bash
curl http://127.0.0.1:8400/v1/chat/completions \
  -H "Authorization: Bearer $AGENT_CTL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'
```

### Surface

- `POST /v1/chat/completions` — chat, including real `stream:true` SSE (passthrough; capture/cost are aggregated at stream end). Fallback applies only before the first byte; once streaming starts the chosen target is committed.
- `POST /v1/embeddings` — `input` as a string or array of strings (OpenAI-compatible providers only).
- `GET /v1/models`, `GET /healthz`, `GET /metrics` (Prometheus).

### Governance knobs (config)

- `circuit_failure_threshold` / `circuit_cooldown_s` — per-provider circuit breaker; the fallback chain skips an open provider until cooldown.
- `request_deadline_s` — wall-clock budget per call; caps the worst case of retries × fallback × per-target timeout.
- `budgets` (per-consumer USD) / `budget_global` — cost budget gate; an exhausted budget short-circuits before hitting a provider and returns HTTP 402.
- `capture_async` — capture writes run off the request path on a background thread (fail-open; the request never blocks on storage I/O).

See [ADR-0001](docs/adr/0001-gateway-maturity-and-hardening.md) for the maturity/hardening decisions and the remaining (intentional) non-goals: distributed circuit/cache, Postgres capture store, persistent/shared budget windows, tiered cost modeling, and multi-tenant auth.

## Capture Storage

Captures are stored in SQLite at `db_path`. The store initializes schema metadata and indexes automatically. Request and response text are redacted before persistence, including nested content blocks and tool payloads.

## Production Notes

- Keep `serve` bound to `127.0.0.1` unless an auth token and network controls are in place.
- Tool-call responses are not cached by default because they often depend on external state.
- Retries use exponential backoff with jitter to avoid synchronized retry bursts.
- Real-provider integration tests should be run manually with API keys and low `max_tokens`; unit tests avoid network calls.
- See [operations.md](docs/operations.md) for Docker Compose, release, rollback, and runtime checks.
- See [configuration.md](docs/configuration.md) for config schema generation and migration policy.
- See [release.md](docs/release.md) for versioning, tagging, and real-provider smoke tests.
