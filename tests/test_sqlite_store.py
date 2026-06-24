# tests/test_sqlite_store.py
from agent_ctl.models import CallRecord
from agent_ctl.store.sqlite_store import SqliteCaptureStore


def _rec(cid, cost):
    return CallRecord(
        id=cid,
        consumer="t",
        status="success",
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
