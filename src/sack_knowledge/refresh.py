import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from .embedding import embed
from .operations import operation_lock
from .sync import sync


def record_state(config, error):
    database = Path(config["storage"]["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database, timeout=60)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS sync_state (
                source TEXT NOT NULL, account TEXT NOT NULL, token TEXT,
                last_success TEXT, last_error TEXT, PRIMARY KEY(source, account)
            )
        """)
        db.execute("""
            INSERT INTO sync_state(source, account, last_success, last_error)
            VALUES ('refresh', 'local', ?, ?)
            ON CONFLICT(source, account) DO UPDATE SET
                last_success=coalesce(excluded.last_success, sync_state.last_success),
                last_error=excluded.last_error
        """, (None if error else datetime.now(UTC).isoformat(), error))


def refresh(config, root=Path("local")):
    start = perf_counter()
    with operation_lock(root, "refresh") as acquired:
        if not acquired:
            return {"status": "busy"}
        stats = {}
        phase = "sync"
        try:
            if "discord" in config:
                result = sync(config)
                stats.update({key: result[key] for key in
                              ("inserted", "updated", "deleted", "chunks_built", "chunks_removed")})
                stats["deletions_held"] = sum(item["missing"] for item in result["deletions_held"])
            phase = "embed"
            stats["embedded"] = embed(config)["embedded"]
            phase = "state"
            record_state(config, None)
            stats["status"] = "success"
        except Exception as error:
            stats.update(status="failed", phase=phase)
            try:
                record_state(config, f"{phase}:{type(error).__name__}")
            except (OSError, sqlite3.Error):
                stats["state_written"] = False
        stats["elapsed_ms"] = round((perf_counter() - start) * 1000, 2)
        return stats
