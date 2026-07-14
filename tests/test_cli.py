import json

from fastapi.testclient import TestClient

from agent_ctl.cli import main
from agent_ctl.config import Config
from agent_ctl.models import CallRecord
from agent_ctl.providers.fake import FakeProvider
from agent_ctl.store.sqlite_store import SqliteCaptureStore


def test_cost_command_reports(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "c.db")
    SqliteCaptureStore(db).save(
        CallRecord(
            id="a",
            consumer="t",
            status="success",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        )
    )
    monkeypatch.setattr(
        "agent_ctl.cli.load_config", lambda path=None: Config(db_path=db)
    )
    rc = main(["cost"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.01" in out
    assert "calls" in out.lower()


def test_cost_command_group_by_model(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "c.db")
    store = SqliteCaptureStore(db)
    store.save(
        CallRecord(
            id="a",
            ts=10,
            consumer="t",
            status="success",
            model_requested="fake/a",
            model_resolved="fake/a",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.01,
        )
    )
    monkeypatch.setattr(
        "agent_ctl.cli.load_config", lambda path=None: Config(db_path=db)
    )
    rc = main(["cost", "--group-by", "model"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "fake/a" in out
    assert "groups" in out


def test_captures_command_filters_and_json(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "c.db")
    store = SqliteCaptureStore(db)
    store.save(
        CallRecord(
            id="a",
            ts=10,
            consumer="ops",
            status="error",
            model_requested="fake/a",
            model_resolved="fake/a",
        )
    )
    store.save(
        CallRecord(
            id="b",
            ts=20,
            consumer="web",
            status="success",
            model_requested="fake/b",
            model_resolved="fake/b",
        )
    )
    monkeypatch.setattr(
        "agent_ctl.cli.load_config", lambda path=None: Config(db_path=db)
    )
    rc = main(["captures", "--consumer", "ops", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"id": "a"' in out
    assert '"id": "b"' not in out


def test_export_streams_jsonl_in_time_order(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "c.db")
    store = SqliteCaptureStore(db)
    store.save(CallRecord(id="b", ts=20, consumer="ops", status="success"))
    store.save(CallRecord(id="a", ts=10, consumer="ops", status="success"))
    store.save(CallRecord(id="c", ts=30, consumer="web", status="error"))
    monkeypatch.setattr(
        "agent_ctl.cli.load_config", lambda path=None: Config(db_path=db)
    )
    rc = main(["export", "--consumer", "ops"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert [r["id"] for r in lines] == ["a", "b"]  # 时序升序、按 consumer 过滤
    assert all(r["consumer"] == "ops" for r in lines)


def test_export_to_file(tmp_path, monkeypatch):
    db = str(tmp_path / "c.db")
    SqliteCaptureStore(db).save(
        CallRecord(id="x", ts=1, consumer="t", status="success")
    )
    out_file = tmp_path / "traces.jsonl"
    monkeypatch.setattr(
        "agent_ctl.cli.load_config", lambda path=None: Config(db_path=db)
    )
    assert main(["export", "--out", str(out_file)]) == 0
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "x"


def test_config_schema_command_outputs_schema(capsys, monkeypatch):
    monkeypatch.setattr("agent_ctl.cli.load_config", lambda path=None: Config())
    assert main(["config-schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["title"] == "Config"
    assert "routes" in schema["properties"]


def test_config_schema_command_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_ctl.cli.load_config", lambda path=None: Config())
    out_file = tmp_path / "schema.json"
    assert main(["config-schema", "--out", str(out_file)]) == 0
    assert json.loads(out_file.read_text(encoding="utf-8"))["title"] == "Config"


def test_version_command(capsys, monkeypatch):
    monkeypatch.setattr("agent_ctl.cli.load_config", lambda path=None: Config())
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


def test_doctor_flags_empty_routes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(routes={}, db_path=str(tmp_path / "c.db")),
    )
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "routes" in out.lower()


def test_doctor_flags_unknown_provider(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"default": ["cohere/command"]}, db_path=str(tmp_path / "c.db")
        ),
    )
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "cohere" in out.lower()


def test_doctor_allows_catalog_provider(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"default": ["deepseek/deepseek-chat"]},
            db_path=str(tmp_path / "c.db"),
        ),
    )
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_doctor_prints_capability_matrix(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"chat": ["deepseek/deepseek-chat"]},
            db_path=str(tmp_path / "c.db"),
        ),
    )
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "能力矩阵" in out
    assert "deepseek/deepseek-chat" in out
    assert "embed" in out  # deepseek 走 OpenAI 兼容 → 有 embed


def test_doctor_warns_on_inconsistent_fallback_capabilities(
    tmp_path, capsys, monkeypatch
):
    # 回退链:openai(有 embed)→ anthropic(无 embed)→ embed 请求回退会失败
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"mix": ["openai/gpt-4o", "anthropic/claude-sonnet-4-6"]},
            db_path=str(tmp_path / "c.db"),
        ),
    )
    assert main(["doctor"]) == 0  # 警告不致失败
    out = capsys.readouterr().out
    assert "WARN" in out
    assert "embed" in out  # 指出 embed 能力在目标间不一致


def test_doctor_prod_skips_alias_prices_for_unavailable_providers(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"default": ["openai/gpt-4o"]},
            model_aliases={"mini": "openai/gpt-4o-mini"},
            prices={"gpt-4o": (1.0, 2.0)},
            profile="prod",
            db_path=str(tmp_path / "c.db"),
        ),
    )
    monkeypatch.setattr("agent_ctl.providers.catalog.available_providers", lambda: [])
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_doctor_prod_requires_alias_prices_for_available_providers(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"default": ["openai/gpt-4o"]},
            model_aliases={"mini": "openai/gpt-4o-mini"},
            prices={"gpt-4o": (1.0, 2.0)},
            profile="prod",
            db_path=str(tmp_path / "c.db"),
        ),
    )
    monkeypatch.setattr(
        "agent_ctl.providers.catalog.available_providers", lambda: ["openai"]
    )
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "gpt-4o-mini" in out


def test_doctor_strict_alias_prices_checks_unavailable_providers(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"default": ["openai/gpt-4o"]},
            model_aliases={"mini": "anthropic/claude-sonnet-4-6"},
            prices={"gpt-4o": (1.0, 2.0)},
            profile="prod",
            db_path=str(tmp_path / "c.db"),
        ),
    )
    monkeypatch.setattr("agent_ctl.providers.catalog.available_providers", lambda: [])
    rc = main(["doctor", "--strict-alias-prices"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "claude-sonnet-4-6" in out


def test_serve_rejects_non_local_without_token_before_provider_setup(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(db_path=str(tmp_path / "c.db")),
    )

    def boom():
        raise AssertionError("provider setup should not be reached")

    monkeypatch.setattr("agent_ctl.providers.catalog.available_providers", boom)
    rc = main(["serve", "--host", "0.0.0.0"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "api-token" in out


def test_serve_rejects_non_local_default_token_before_provider_setup(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(db_path=str(tmp_path / "c.db")),
    )

    def boom():
        raise AssertionError("provider setup should not be reached")

    monkeypatch.setattr("agent_ctl.providers.catalog.available_providers", boom)
    rc = main(["serve", "--host", "0.0.0.0", "--api-token", "change-me"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "non-default" in out


def test_serve_models_include_routes_and_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"default": ["fake/a"]},
            model_aliases={"alias-a": "fake/a"},
            db_path=str(tmp_path / "c.db"),
        ),
    )
    monkeypatch.setattr(
        "agent_ctl.providers.catalog.available_providers", lambda: ["fake"]
    )
    monkeypatch.setattr(
        "agent_ctl.providers.catalog.build_providers",
        lambda: {"fake": FakeProvider(["ok"])},
    )
    seen = {}

    def fake_run(app, host, port):
        seen["models"] = TestClient(app).get("/v1/models").json()

    monkeypatch.setattr("uvicorn.run", fake_run)

    assert main(["serve"]) == 0
    ids = {m["id"] for m in seen["models"]["data"]}
    assert ids == {"default", "alias-a"}


def test_serve_fails_friendly_on_bad_default_max_tokens_env(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("AGENT_CTL_DEFAULT_MAX_TOKENS", "bad")
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(
            routes={"default": ["fake/a"]},
            db_path=str(tmp_path / "c.db"),
        ),
    )
    def boom():
        raise AssertionError("provider setup should not be reached")

    monkeypatch.setattr("agent_ctl.providers.catalog.available_providers", boom)

    assert main(["serve"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "AGENT_CTL_DEFAULT_MAX_TOKENS" in out


def test_serve_fails_friendly_on_bad_trusted_proxy_cidr(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        "agent_ctl.cli.load_config",
        lambda path=None: Config(db_path=str(tmp_path / "c.db")),
    )

    def boom():
        raise AssertionError("provider setup should not be reached")

    monkeypatch.setattr("agent_ctl.providers.catalog.available_providers", boom)

    assert main(["serve", "--trusted-proxy-cidr", "not-a-cidr"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "trusted-proxy-cidr" in out


def test_cli_fails_friendly_on_unknown_config_key(tmp_path, capsys):
    """P2-b:doctor 等命令对拼错的配置键给 FAIL 提示 + 退出码 1,非裸 traceback。"""
    from agent_ctl.cli import main

    cfg = tmp_path / "bad.yaml"
    cfg.write_text("routez:\n  default: [anthropic/x]\n", encoding="utf-8")
    assert main(["--config", str(cfg), "doctor"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "routez" in out


def test_parse_since_friendly_error():
    """P3:非法 --since 给友好 FAIL,而非裸 ValueError traceback。"""
    import pytest

    from agent_ctl.cli import _parse_since

    with pytest.raises(SystemExit, match="since"):
        _parse_since("yesterday")
    with pytest.raises(SystemExit, match="since"):
        _parse_since("xxh")
    assert _parse_since(None) is None
    assert _parse_since("0") == 0.0
