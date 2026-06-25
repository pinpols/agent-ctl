import json

from agent_ctl.cli import main
from agent_ctl.config import Config
from agent_ctl.store.sqlite_store import SqliteCaptureStore
from agent_ctl.models import CallRecord


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
