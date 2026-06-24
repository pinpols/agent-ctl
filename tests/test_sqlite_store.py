# tests/test_sqlite_store.py
from agentctl.models import CallRecord
from agentctl.store.sqlite_store import SqliteCaptureStore


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
