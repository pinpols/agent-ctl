# Changelog

All notable changes to `agent-ctl` are documented here.

The project follows semantic versioning for public CLI/API behavior:

- **MAJOR**: incompatible config, CLI, HTTP, or library API changes.
- **MINOR**: backward-compatible features and new provider capabilities.
- **PATCH**: bug fixes, docs, tests, and internal refactors.

## [Unreleased]

### Changed

- Split auth-failure and business request rate limit buckets; successful `/metrics` scrapes no longer consume business quota.
- Added trusted proxy CIDR filtering for `X-Forwarded-For` handling; proxy headers now default to local-only trust.
- Updated prod `doctor` alias price checks to respect currently available providers, with `--strict-alias-prices` for shared config validation.
- Expanded `constraints.txt` to a fully resolved runtime dependency constraints file.

## [0.1.0] - 2026-06-25

Initial local AgentOps gateway release.

### Added

- Library gateway for routed chat calls with retry, fallback, deadline, budget, circuit breaker, cache, capture, and metrics.
- OpenAI-compatible server with `/v1/chat/completions`, passthrough SSE streaming, `/v1/embeddings`, `/v1/models`, `/healthz`, and `/metrics`.
- Anthropic and OpenAI-compatible provider adapters, including cross-provider tool-call translation.
- SQLite capture store with async write wrapper, schema migration, indexes, JSONL export, and cost summaries.
- CLI commands: `doctor`, `serve`, `captures`, `cost`, `export`, `config-schema`, and `version`.
- Production templates: Dockerfile, docker compose, configuration schema docs, operations runbook, and release checklist.
