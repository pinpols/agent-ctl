# agentctl/store/sqlite_store.py
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from agentctl.models import CallRecord


class SqliteCaptureStore:
    """SQLite 捕获存储:一行一条 CallRecord(JSON 整存 + 关键列冗余便于聚合)。

    线程安全:check_same_thread=False 允许跨线程复用连接,WAL 提升并发读,
    写入加 Lock 串行化(SQLite 单写),避免消费者多线程并发调用时 'database is locked'。
    """

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS call_record ("
                " id TEXT PRIMARY KEY, ts REAL, consumer TEXT, status TEXT,"
                " input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,"
                " doc TEXT NOT NULL)"
            )
            self._conn.commit()

    def save(self, record: CallRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO call_record"
                " (id, ts, consumer, status, input_tokens, output_tokens, cost_usd, doc)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.ts,
                    record.consumer,
                    record.status,
                    record.input_tokens,
                    record.output_tokens,
                    record.cost_usd,
                    record.model_dump_json(),
                ),
            )
            self._conn.commit()

    def list_recent(self, limit: int) -> list[CallRecord]:
        rows = self._conn.execute(
            "SELECT doc FROM call_record ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [CallRecord(**json.loads(r["doc"])) for r in rows]

    def cost_summary(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) c, COALESCE(SUM(cost_usd),0) cost,"
            " COALESCE(SUM(input_tokens),0) it, COALESCE(SUM(output_tokens),0) ot"
            " FROM call_record"
        ).fetchone()
        return {
            "calls": row["c"],
            "total_cost_usd": row["cost"],
            "total_input_tokens": row["it"],
            "total_output_tokens": row["ot"],
        }
