# Release and Versioning

`agent-ctl` uses semantic versioning for public behavior.

## Version Sources

Keep these in sync for every release:

- `pyproject.toml` → `[project].version`
- `agent_ctl/__init__.py` → `__version__`
- `CHANGELOG.md`
- Docker image tag

Check the installed CLI version:

```bash
agent-ctl version
```

## Release Flow

1. Update version sources.
2. Update `CHANGELOG.md`.
3. Run local verification:

   ```bash
   python -m pytest -q
   python -m ruff check .
   python -m mypy agent_ctl
   agent-ctl --config agent-ctl.example.yaml doctor
   ```

4. Build and smoke-check the container:

   ```bash
   docker build -t agent-ctl:<version> .
   docker compose up -d
   curl http://127.0.0.1:8400/healthz
   ```

5. Tag source:

   ```bash
   git tag v<version>
   git push origin main v<version>
   ```

## Real Provider Smoke Tests

Unit tests never require network access. To exercise real providers manually:

```bash
export AGENT_CTL_RUN_PROVIDER_SMOKE=1
export OPENAI_API_KEY=...
python -m pytest tests/test_provider_smoke.py -q
```
