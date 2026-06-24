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
