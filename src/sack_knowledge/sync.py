import json
import sqlite3
from collections import defaultdict
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from .discord import DiscordSource, connect


def utc(value):
    return value.astimezone(UTC).isoformat(timespec="microseconds") if value else None


def person_id(db, alias, name=None):
    existing = db.execute(
        "SELECT person_id FROM person_aliases WHERE source='discord' AND source_id=?", (alias,)
    ).fetchone()
    if existing:
        if name:
            db.execute("UPDATE persons SET display_name=? WHERE id=? AND display_name=''",
                       (name, existing[0]))
        return existing[0]
    person = db.execute("INSERT INTO persons(display_name) VALUES (?)", (name or "",)).lastrowid
    db.execute("INSERT INTO person_aliases VALUES ('discord', ?, ?)", (alias, person))
    return person


def upsert(db, guild, message, fingerprint, now):
    source_id = message["source_id"]
    channel = message["channel_id"]
    record = db.execute("""
        INSERT INTO records(source, account, source_id, kind, created_at, fetched_at, edited_at,
                            title, body, original_ref, channel_id, fingerprint, reply_source_id)
        VALUES ('discord', ?, ?, 'message', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, account, source_id) DO UPDATE SET
            created_at=excluded.created_at, fetched_at=excluded.fetched_at, edited_at=excluded.edited_at,
            title=excluded.title, body=excluded.body, original_ref=excluded.original_ref,
            channel_id=excluded.channel_id, fingerprint=excluded.fingerprint,
            reply_source_id=excluded.reply_source_id, deleted=0
        RETURNING id
    """, (guild, source_id, utc(message["created_at"]), now, utc(message["edited_at"]),
          message["title"], message["body"] or "", f"https://discord.com/channels/{guild}/{channel}/{source_id}",
          channel, fingerprint, message["reply_id"])).fetchone()[0]
    db.execute("DELETE FROM record_person_links WHERE record_id=?", (record,))
    db.execute("DELETE FROM record_relations WHERE record_id=? AND kind='reply'", (record,))
    if message["author_id"]:
        db.execute("INSERT INTO record_person_links VALUES (?, ?, 'sender')",
                   (record, person_id(db, message["author_id"], message["author_name"])))
    db.executemany("INSERT OR IGNORE INTO record_person_links VALUES (?, ?, 'mention')", (
        (record, person_id(db, str(mention))) for mention in (message["mentions"] or [])
    ))
    return record


def groups(rows, gap, cap):
    batch = []
    size = 0
    previous = None
    for row in rows:
        timestamp = datetime.fromisoformat(row["created_at"])
        length = len(row["body"])
        if batch and ((timestamp - previous).total_seconds() > gap or size + length + 1 > cap):
            yield batch
            batch = []
            size = 0
        size += length + bool(batch)
        batch.append(row)
        previous = timestamp
    if batch:
        yield batch


def rebuild(db, guild, channel, gap, cap):
    old = defaultdict(list)
    for row in db.execute("""
        SELECT c.id, c.first_record_id, c.starts_at, c.ends_at, c.body, member.record_id
        FROM records r JOIN chunks c ON c.first_record_id=r.id
        JOIN chunk_records member ON member.chunk_id=c.id
        WHERE r.source='discord' AND r.account=? AND r.channel_id=?
    """, (guild, channel)):
        old[(row["id"], row["first_record_id"], row["starts_at"], row["ends_at"], row["body"])].append(
            row["record_id"]
        )
    existing = {(first, start, end, body, tuple(sorted(ids))): chunk
                for (chunk, first, start, end, body), ids in old.items()}
    rows = db.execute("""
        SELECT id, created_at, body FROM records
        WHERE source='discord' AND account=? AND channel_id=? AND deleted=0 ORDER BY created_at, id
    """, (guild, channel))
    proposed = []
    for batch in groups(rows, gap, cap):
        key = (batch[0]["id"], batch[0]["created_at"], batch[-1]["created_at"],
               "\n".join(row["body"] for row in batch), tuple(sorted(row["id"] for row in batch)))
        if existing.pop(key, None) is None:
            proposed.append(key)
    for chunk in existing.values():
        db.execute("DELETE FROM chunk_fts WHERE rowid=?", (chunk,))
        db.execute("DELETE FROM chunks WHERE id=?", (chunk,))
    for first, start, end, body, ids in proposed:
        chunk = db.execute("""
            INSERT INTO chunks(first_record_id, starts_at, ends_at, body) VALUES (?, ?, ?, ?)
        """, (first, start, end, body)).lastrowid
        db.execute("INSERT INTO chunk_fts(rowid, body) VALUES (?, ?)", (chunk, body))
        db.executemany("INSERT INTO chunk_records VALUES (?, ?)", ((chunk, record) for record in ids))
    return len(proposed), len(existing)


def sync_guild(db, source, guild, now, gap, cap, stats):
    manifest = source.manifest(guild)
    known = {row["source_id"]: row for row in db.execute("""
        SELECT id, source_id, channel_id, fingerprint, deleted FROM records
        WHERE source='discord' AND account=?
    """, (guild,))}
    changed = {key for key, fingerprint in manifest.items()
               if key not in known or known[key]["deleted"] or known[key]["fingerprint"] != fingerprint}
    channels = set()
    settings = db.execute("""
        SELECT token FROM sync_state WHERE source='discord' AND account=?
    """, (guild,)).fetchone()
    if settings is None or settings["token"] != f"{gap}:{cap}":
        channels.update(row["channel_id"] for row in known.values())
    affected_ids = []
    for message in source.messages(guild, sorted(changed)):
        key = message["source_id"]
        before = known.get(key)
        if before is not None:
            channels.add(before["channel_id"])
            # Remove chunks anchored to a record before its channel changes.
            if before["channel_id"] != message["channel_id"]:
                db.execute("DELETE FROM chunk_fts WHERE rowid IN (SELECT id FROM chunks WHERE first_record_id=?)",
                           (before["id"],))
                stats["chunks_removed"] += db.execute(
                    "DELETE FROM chunks WHERE first_record_id=?", (before["id"],)
                ).rowcount
        affected_ids.append(upsert(db, guild, message, manifest[key], now))
        stats["updated" if before is not None else "inserted"] += 1
        channels.add(message["channel_id"])
    if len(affected_ids) != len(changed):
        raise RuntimeError("Discord snapshot changed")
    for key in known.keys() - manifest.keys():
        row = known[key]
        if not row["deleted"]:
            db.execute("UPDATE records SET deleted=1 WHERE id=?", (row["id"],))
            db.execute("DELETE FROM record_relations WHERE kind='reply' AND (record_id=? OR related_record_id=?)",
                       (row["id"], row["id"]))
            channels.add(row["channel_id"])
            affected_ids.append(row["id"])
            stats["deleted"] += 1
    if affected_ids:
        db.execute("""
            INSERT OR IGNORE INTO record_relations
            SELECT child.id, parent.id, 'reply' FROM records child
            JOIN records parent ON (parent.source, parent.account, parent.source_id) =
                                   (child.source, child.account, child.reply_source_id)
            WHERE child.source='discord' AND child.account=? AND NOT child.deleted AND NOT parent.deleted
        """, (guild,))
        db.executemany("""
            UPDATE ai_extracted_items SET state='invalid'
            WHERE state <> 'invalid' AND (target_kind='record' AND target_id=?
                OR EXISTS (SELECT 1 FROM json_each(evidence_ids) WHERE value=?))
        """, ((record, record) for record in affected_ids))
    for channel in sorted(channels):
        built, removed = rebuild(db, guild, channel, gap, cap)
        stats["chunks_built"] += built
        stats["chunks_removed"] += removed


@contextmanager
def postgres_source(config):
    with connect(config) as connection:
        yield DiscordSource(connection)


def sync(config, open_source=postgres_source):
    start = perf_counter()
    options = config["discord"]
    guilds = sorted(set(options["guild_ids"]))
    gap, cap = options["gap_seconds"], options["chunk_characters"]
    if not guilds or any(not guild.isdecimal() or not guild.isascii() for guild in guilds) or gap < 0 or cap < 1:
        raise ValueError("Invalid Discord sync configuration")
    path = Path(config["storage"]["database"])
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = dict(inserted=0, updated=0, deleted=0, chunks_built=0, chunks_removed=0)
    with closing(sqlite3.connect(path, timeout=60)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(Path(__file__).with_name("schema.sql").read_text())
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                with open_source(options) as source:
                    for guild in guilds:
                        sync_guild(db, source, guild, utc(datetime.now(UTC)), gap, cap, stats)
                db.execute("""
                    UPDATE allow_list SET local_storage=0 WHERE source='discord' AND local_storage=1
                    AND target NOT IN (SELECT value FROM json_each(?))
                """, (json.dumps(guilds),))
                db.executemany("""
                    INSERT INTO allow_list(source, target, local_storage) VALUES ('discord', ?, 1)
                    ON CONFLICT(source, target) DO UPDATE SET local_storage=1 WHERE local_storage=0
                """, ((guild,) for guild in guilds))
                db.executemany("""
                    INSERT INTO sync_state(source, account, token, last_success) VALUES ('discord', ?, ?, ?)
                    ON CONFLICT(source, account) DO UPDATE SET
                        token=excluded.token, last_success=excluded.last_success, last_error=NULL
                """, ((guild, f"{gap}:{cap}", utc(datetime.now(UTC))) for guild in guilds))
        except Exception as error:
            with db:
                db.executemany("""
                    INSERT INTO sync_state(source, account, last_error) VALUES ('discord', ?, ?)
                    ON CONFLICT(source, account) DO UPDATE SET last_error=excluded.last_error
                """, ((guild, type(error).__name__) for guild in guilds))
            raise
    return stats | {"elapsed_ms": round((perf_counter() - start) * 1000, 2),
                    "sqlite_bytes": path.stat().st_size}
