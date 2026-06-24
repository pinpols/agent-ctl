# tests/test_sqlite_store.py
import sqlite3

from agent_ctl.models import CallRecord
from agent_ctl.store.sqlite_store import SqliteCaptureStore


def _rec(cid, cost, *, consumer="t", status="success", model="fake/m", ts=0.0):
    return CallRecord(
        id=cid,
        ts=ts,
        consumer=consumer,
        status=status,
        model_requested=model,
        model_resolved=model,
        input_tokens=10,
        output_tokens=5,
        cost_usd=cost,
    )


def test_save_and_list_recent(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    store.save(_rec("a", 0.01))
    store.save(_rec("b", 0.02))
    recent = store.list_recent(10)
    assert {r.id for r in recent} == {"a", "b"}


def test_cost_summary(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    store.save(_rec("a", 0.01))
    store.save(_rec("b", 0.02))
    s = store.cost_summary()
    assert s["calls"] == 2
    assert round(s["total_cost_usd"], 4) == 0.03
    assert s["total_input_tokens"] == 20


def test_filters_and_grouped_cost_summary(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    store.save(_rec("a", 0.01, consumer="ops", status="success", model="fake/a", ts=10))
    store.save(_rec("b", 0.02, consumer="ops", status="error", model="fake/b", ts=20))
    store.save(_rec("c", 0.03, consumer="web", status="success", model="fake/a", ts=30))

    assert [r.id for r in store.list_recent(10, consumer="ops")] == ["b", "a"]
    assert [r.id for r in store.list_recent(10, status="success", model="fake/a")] == [
        "c",
        "a",
    ]
    assert [r.id for r in store.list_recent(10, since=25)] == ["c"]

    grouped = store.cost_summary(group_by="consumer")
    assert grouped["groups"] == [
        {
            "bucket": "ops",
            "calls": 2,
            "total_cost_usd": 0.03,
            "total_input_tokens": 20,
            "total_output_tokens": 10,
        },
        {
            "bucket": "web",
            "calls": 1,
            "total_cost_usd": 0.03,
            "total_input_tokens": 10,
            "total_output_tokens": 5,
        },
    ]


def test_store_migrates_legacy_table(tmp_path):
    db = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE call_record ("
        " id TEXT PRIMARY KEY, ts REAL, consumer TEXT, status TEXT,"
        " input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,"
        " doc TEXT NOT NULL)"
    )
    rec = _rec("a", 0.01, model="fake/m")
    conn.execute(
        "INSERT INTO call_record"
        " (id, ts, consumer, status, input_tokens, output_tokens, cost_usd, doc)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            rec.id,
            rec.ts,
            rec.consumer,
            rec.status,
            rec.input_tokens,
            rec.output_tokens,
            rec.cost_usd,
            rec.model_dump_json(),
        ),
    )
    conn.commit()
    conn.close()

    store = SqliteCaptureStore(db)
    assert store.list_recent(1, model="fake/m")[0].id == "a"
    assert store.cost_summary(group_by="model")["groups"][0]["bucket"] == "fake/m"


def test_concurrent_read_write_thread_safe(tmp_path):
    import threading

    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    n_writers = 4
    writes_per_thread = 25  # total = 100
    errors: list[Exception] = []

    def writer(start):
        try:
            for i in range(writes_per_thread):
                store.save(_rec(f"w{start}-{i}", 0.001))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(30):
                store.list_recent(10)
                store.cost_summary()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_writers)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == [], f"Thread errors: {errors}"
    assert store.cost_summary()["calls"] == n_writers * writes_per_thread


def test_concurrent_writes_thread_safe(tmp_path):
    import threading

    store = SqliteCaptureStore(str(tmp_path / "c.db"))

    def writer(start):
        for i in range(20):
            store.save(_rec(f"{start}-{i}", 0.001))

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert store.cost_summary()["calls"] == 100  # 5 线程 × 20,无 'database is locked'


def test_store_context_manager_closes_connection(tmp_path):
    with SqliteCaptureStore(str(tmp_path / "c.db")) as store:
        store.save(_rec("a", 0.01))

    import pytest

    with pytest.raises(Exception):
        store.cost_summary()
