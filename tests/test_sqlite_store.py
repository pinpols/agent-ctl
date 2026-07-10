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


def test_iter_all_streams_in_time_order_and_filters(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    store.save(_rec("b", 0.0, ts=20))
    store.save(_rec("a", 0.0, ts=10))
    store.save(_rec("c", 0.0, consumer="other", ts=30))
    got = list(store.iter_all(consumer="t"))
    assert [r.id for r in got] == ["a", "b"]  # 升序时序 + 过滤
    desc = list(store.iter_all(ascending=False))
    assert [r.id for r in desc] == ["c", "b", "a"]


def test_iter_all_memory_db():
    store = SqliteCaptureStore(":memory:")
    store.save(_rec("a", 0.0, ts=1))
    store.save(_rec("b", 0.0, ts=2))
    assert [r.id for r in store.iter_all()] == ["a", "b"]


# ── 深审 round4 P2-12:启动回填只跑一次(marker),不每次全量重扫 error 记录 ──


def test_backfill_not_rerun_on_reopen(tmp_path, monkeypatch):
    """error 记录 model_resolved 合法为 NULL,老实现每次启动都全量重扫重写;
    回填完成后应打标记,再次打开不再扫。"""
    path = str(tmp_path / "c.db")
    with SqliteCaptureStore(path) as s:
        s.save(
            CallRecord(
                id="err1",
                ts=1.0,
                consumer="t",
                status="error",
                model_requested="m",
                model_resolved=None,
            )
        )
    calls = []
    monkeypatch.setattr(
        SqliteCaptureStore,
        "_backfill_model_columns",
        lambda self: calls.append(1),
    )
    with SqliteCaptureStore(path):
        pass
    assert calls == []  # 标记已在 → 不再重扫


def test_backfill_runs_for_legacy_db_then_marks(tmp_path):
    """老库(无 model_* 列、无标记)→ 补列 + 回填 + 打标;再开不重跑。"""
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE call_record (id TEXT PRIMARY KEY, ts REAL, consumer TEXT,"
        " status TEXT, input_tokens INTEGER, output_tokens INTEGER,"
        " cost_usd REAL, doc TEXT NOT NULL)"
    )
    doc = CallRecord(
        id="old1",
        ts=1.0,
        consumer="t",
        status="success",
        model_requested="fake/m",
        model_resolved="fake/m",
    ).model_dump_json()
    conn.execute(
        "INSERT INTO call_record (id, ts, consumer, status, input_tokens,"
        " output_tokens, cost_usd, doc) VALUES ('old1',1.0,'t','success',0,0,0,?)",
        (doc,),
    )
    conn.commit()
    conn.close()
    with SqliteCaptureStore(path) as s:
        rows = s._conn.execute(
            "SELECT model_requested, model_resolved FROM call_record"
        ).fetchall()
        assert rows[0]["model_requested"] == "fake/m"
        assert rows[0]["model_resolved"] == "fake/m"
        marked = s._conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 2"
        ).fetchone()
        assert marked is not None


# ── 深审 round2(P2-c):双进程并发 schema 升级容错 ──────────────────────────


def test_concurrent_upgrade_duplicate_column_tolerated(tmp_path):
    """模拟第二进程场景:本进程读到"列缺失"后、ALTER 前,另一进程已把列加上
    (旧库 → 双方同时升级)。后到的 ALTER 得 duplicate column,应视为已升级继续。"""
    import sqlite3 as _sq

    db = str(tmp_path / "legacy.db")
    conn = _sq.connect(db)
    conn.execute(
        "CREATE TABLE call_record ("
        " id TEXT PRIMARY KEY, ts REAL, consumer TEXT, status TEXT,"
        " input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,"
        " doc TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    class RacingStore(SqliteCaptureStore):
        def _add_column(self, name):
            # 抢在本进程 ALTER 之前,"另一进程"先把列加好 → 触发 duplicate column
            other = _sq.connect(self._path)
            other.execute(f"ALTER TABLE call_record ADD COLUMN {name} TEXT")
            other.commit()
            other.close()
            super()._add_column(name)

    store = RacingStore(db)  # 不抛 → 升级容错生效
    cols = {
        r["name"]
        for r in store._conn.execute("PRAGMA table_info(call_record)").fetchall()
    }
    assert {"model_requested", "model_resolved"} <= cols
    store.save(_rec("r1", 0.1))  # 升级后的库照常可写
    assert store.list_recent(1)[0].id == "r1"
    store.close()


def test_other_operational_errors_still_raise(tmp_path):
    """duplicate column 之外的 OperationalError 不得被吞。"""
    import pytest as _pytest
    import sqlite3 as _sq

    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    with _pytest.raises(_sq.OperationalError):
        store._add_column("doc)")  # 非法列名 → syntax error 照常抛
    store.close()


def test_busy_timeout_configured(tmp_path):
    store = SqliteCaptureStore(str(tmp_path / "c.db"))
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    store.close()
