# agent_ctl/store/sqlite_store.py
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from agent_ctl.models import CallRecord

SCHEMA_VERSION = 1
# v2 = model_requested/model_resolved 列回填完成标记(见 _init_schema)
_BACKFILL_MARKER_VERSION = 2


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
        # 显式 busy_timeout:多进程同时打开同一 db(如 CLI 与 server 并存)时,
        # schema 升级/写入遇锁等待而非立刻 'database is locked'。
        self._conn.execute("PRAGMA busy_timeout=5000")
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS call_record ("
                " id TEXT PRIMARY KEY, ts REAL, consumer TEXT, status TEXT,"
                " model_requested TEXT, model_resolved TEXT,"
                " input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,"
                " doc TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version INTEGER PRIMARY KEY, applied_at REAL DEFAULT (unixepoch()))"
            )
            self._ensure_columns()
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_call_record_ts ON call_record(ts)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_call_record_consumer "
                "ON call_record(consumer)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_call_record_status "
                "ON call_record(status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_call_record_model_status "
                "ON call_record(model_resolved, status)"
            )
            self._conn.commit()

    def _ensure_columns(self) -> None:
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(call_record)").fetchall()
        }
        for name in ("model_requested", "model_resolved"):
            if name not in existing:
                self._add_column(name)
        # 回填只跑一次:error 记录的 model_resolved 合法为 NULL,按 "IS NULL" 判断
        # 会让每次启动都全量重扫重写这批行。回填完成即打 v2 标记,后续启动跳过。
        marked = self._conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?",
            (_BACKFILL_MARKER_VERSION,),
        ).fetchone()
        if marked is None:
            self._backfill_model_columns()
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (_BACKFILL_MARKER_VERSION,),
            )

    def _add_column(self, name: str) -> None:
        """ALTER ADD COLUMN 容忍并发升级:两个进程同时打开旧库,双方都看到列缺失、
        都发 ALTER,后到者得 'duplicate column'——这等价于"已被别人升级完",
        吞掉继续;其他 OperationalError 照常抛。"""
        try:
            self._conn.execute(f"ALTER TABLE call_record ADD COLUMN {name} TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise

    def _backfill_model_columns(self) -> None:
        rows = self._conn.execute(
            "SELECT id, doc FROM call_record "
            "WHERE model_requested IS NULL OR model_resolved IS NULL"
        ).fetchall()
        for row in rows:
            try:
                data = json.loads(row["doc"])
            except json.JSONDecodeError:
                continue
            self._conn.execute(
                "UPDATE call_record SET model_requested = ?, model_resolved = ? "
                "WHERE id = ?",
                (
                    data.get("model_requested", ""),
                    data.get("model_resolved"),
                    row["id"],
                ),
            )

    def save(self, record: CallRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO call_record"
                " (id, ts, consumer, status, model_requested, model_resolved,"
                " input_tokens, output_tokens, cost_usd, doc)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.ts,
                    record.consumer,
                    record.status,
                    record.model_requested,
                    record.model_resolved,
                    record.input_tokens,
                    record.output_tokens,
                    record.cost_usd,
                    record.model_dump_json(),
                ),
            )
            self._conn.commit()

    def list_recent(
        self,
        limit: int,
        *,
        consumer: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since: float | None = None,
    ) -> list[CallRecord]:
        where, params = self._filters(
            consumer=consumer, status=status, model=model, since=since
        )
        with self._lock:
            rows = self._conn.execute(
                f"SELECT doc FROM call_record{where} ORDER BY ts DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [CallRecord(**json.loads(r["doc"])) for r in rows]

    def iter_all(
        self,
        *,
        consumer: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since: float | None = None,
        ascending: bool = True,
    ):
        """流式逐条产出匹配记录(按 ts 时序),供大批量导出/replay。

        不把全表读进内存(对照 list_recent):file 库另开只读连接惰性游标迭代(WAL 允许
        并发读,不抢写锁);:memory: 库无法跨连接,退回锁内快照迭代。
        """
        where, params = self._filters(
            consumer=consumer, status=status, model=model, since=since
        )
        order = "ASC" if ascending else "DESC"
        sql = f"SELECT doc FROM call_record{where} ORDER BY ts {order}"
        if self._path == ":memory:":
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            for row in rows:
                yield CallRecord(**json.loads(row["doc"]))
            return
        uri = f"{Path(self._path).resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            for row in conn.execute(sql, params):  # 游标惰性取,真流式
                yield CallRecord(**json.loads(row["doc"]))
        finally:
            conn.close()

    def cost_summary(
        self,
        *,
        consumer: str | None = None,
        status: str | None = None,
        model: str | None = None,
        since: float | None = None,
        group_by: str | None = None,
    ) -> dict:
        where, params = self._filters(
            consumer=consumer, status=status, model=model, since=since
        )
        if group_by is not None:
            group_expr = {
                "consumer": "consumer",
                "status": "status",
                "model": "COALESCE(model_resolved, model_requested, '')",
                "day": "date(ts, 'unixepoch')",
            }.get(group_by)
            if group_expr is None:
                raise ValueError(f"bad group_by: {group_by!r}")
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT {group_expr} bucket, COUNT(*) calls,"
                    " COALESCE(SUM(cost_usd),0) cost,"
                    " COALESCE(SUM(input_tokens),0) it,"
                    " COALESCE(SUM(output_tokens),0) ot"
                    f" FROM call_record{where} GROUP BY bucket ORDER BY bucket",
                    params,
                ).fetchall()
            return {
                "groups": [
                    {
                        "bucket": row["bucket"],
                        "calls": row["calls"],
                        "total_cost_usd": row["cost"],
                        "total_input_tokens": row["it"],
                        "total_output_tokens": row["ot"],
                    }
                    for row in rows
                ]
            }
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) c, COALESCE(SUM(cost_usd),0) cost,"
                " COALESCE(SUM(input_tokens),0) it, COALESCE(SUM(output_tokens),0) ot"
                f" FROM call_record{where}",
                params,
            ).fetchone()
        return {
            "calls": row["c"],
            "total_cost_usd": row["cost"],
            "total_input_tokens": row["it"],
            "total_output_tokens": row["ot"],
        }

    def _filters(
        self,
        *,
        consumer: str | None,
        status: str | None,
        model: str | None,
        since: float | None,
    ) -> tuple[str, tuple]:
        clauses: list[str] = []
        params: list[object] = []
        if consumer:
            clauses.append("consumer = ?")
            params.append(consumer)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if model:
            clauses.append("(model_resolved = ? OR model_requested = ?)")
            params.extend([model, model])
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return where, tuple(params)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SqliteCaptureStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
