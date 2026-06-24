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
