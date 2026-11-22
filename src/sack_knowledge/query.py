import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import perf_counter

from .embedding import LocalEmbedder, vector_key
from .errors import LocalCommandError
from .ranking import cosine_rank, fuse, vector_rows
from .visibility import profile, visible_chunk, visible_record


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def open_readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


VISIBLE = visible_record()


def filters(start, end, person, project, source, *, external=False, scoped_name="NULL"):
    clauses = [visible_record(external=external)]
    params = []
    if start is not None:
        clauses.append("r.created_at >= ?")
        params.append(timestamp(start))
    if end is not None:
        clauses.append("r.created_at < ?")
        params.append(timestamp(end))
    if person is not None:
        name_match = f"l.role='sender' AND {scoped_name}=?" if external else "p.display_name=?"
        clauses.append(f"""EXISTS (
            SELECT 1 FROM record_person_links l JOIN persons p ON p.id=l.person_id
            WHERE l.record_id=r.id AND (CAST(p.id AS TEXT)=? OR ({name_match}))
        )""")
        params.extend((person, person))
    if project is not None:
        clauses.append("""EXISTS (
            SELECT 1 FROM projects p WHERE p.id=r.project_id AND (CAST(p.id AS TEXT)=? OR p.name=?)
        )""")
        params.extend((project, project))
    if source is not None:
        clauses.append("r.source=?")
        params.append(source)
    return " AND ".join(clauses), params


SENDER = """(SELECT p.display_name FROM record_person_links l JOIN persons p ON p.id=l.person_id
              WHERE l.record_id=r.id AND l.role='sender') AS person"""


def scoped_sender_name(db):
    if db.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='record_sender_names'").fetchone():
        return "(SELECT display_name FROM record_sender_names n WHERE n.record_id=r.id)"
    return "NULL"
RECORD = """'r:' || r.id AS id, r.created_at AS datetime, r.source,
            r.original_ref, r.title, r.body"""


def excerpt(body, query):
    offset = body.casefold().find(query.casefold()) if query else 0
    start = max(0, offset - 30)
    end = start + 100
    return ("…" if start else "") + body[start:end] + ("…" if end < len(body) else "")


def literal_like(value):
    return "%" + value.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"


def query_tool(function: Callable):
    @wraps(function)
    def run(self, *args, **kwargs):
        started = perf_counter()
        characters = 0
        try:
            with closing(open_readonly(self.database)) as db:
                db.create_function("contains_text", 2, lambda body, term: term.casefold() in body.casefold(),
                                   deterministic=True)
                db.execute("BEGIN")
                result = function(self, db, *args, **kwargs)
            characters = len(encode(result))
            return result
        finally:
            entry = {"tool": function.__name__, "characters": characters,
                     "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                     "profile": self.profile_name, "external": self.external}
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("a", encoding="utf-8") as file:
                file.write(encode(entry) + "\n")
    return run


def search_page(rows, cap, total, position, metadata=None):
    def page(items):
        next_position = position + len(items)
        has_more = bool(items) and next_position < total
        return {"items": items, "has_more": has_more, "total": total,
                "next_position": next_position if has_more else None} | (metadata or {})

    items = []
    for row in rows:
        candidate = [*items, row]
        if len(encode(page(candidate))) <= cap:
            items = candidate
        elif items:
            break
        else:
            date = row["datetime"].replace("+00:00", "Z").replace(".000000Z", "Z")
            minimal = {"id": row["id"], "datetime": date, "excerpt": ""}
            result = page([minimal])
            if len(encode(result)) > cap:
                # A fallback marker can require the more compact epoch-millisecond representation.
                minimal["datetime"] = int(datetime.fromisoformat(date).timestamp() * 1000)
            if len(encode(result)) > cap:
                raise LocalCommandError("The search limit is too small for pagination metadata. Raise it to 200 or more.")
            shorten(result, minimal, "excerpt", row["excerpt"], cap)
            return result
    return page(items)


def shorten(result, row, field, original, cap):
    low, high = 0, min(len(original), cap)
    while low < high:
        middle = (low + high + 1) // 2
        row[field] = original[:middle]
        if len(encode(result)) <= cap:
            low = middle
        else:
            high = middle - 1
    row[field] = original[:low]


def bounded(rows, cap, *, total=None, position=None, metadata=None):
    if position is not None:
        return search_page(rows, cap, total, position, metadata)
    result = {"items": [], "has_more": False} | (metadata or {})
    for row in rows:
        result["items"].append(row)
        result["has_more"] = True
        if len(encode(result)) <= cap:
            continue
        result["items"].pop()
        field = "body" if "body" in row else None
        if field and not result["items"]:
            original = row[field]
            row[field] = ""
            row["truncated"] = True
            result["items"].append(row)
            for optional in ("person", "title"):
                if optional in row and len(encode(result)) > cap:
                    row[optional] = None
            if len(encode(result)) <= cap:
                shorten(result, row, field, original, cap)
                return result
            result["items"].pop()
        break
    returned = len(result["items"])
    result["has_more"] = returned < len(rows)
    # The true/false and numeric/null variants can have different encoded lengths.
    while len(encode(result)) > cap and result["items"]:
        result["items"].pop()
        result["has_more"] = True
    return result


def lexical_rows(db, where, params, query, *, word=False, external=False):
    joins = ""
    match = ""
    score = "0.0"
    if word:
        from .evaluation import segment

        tokens = segment(query)
        if not tokens:
            return {}
        joins = """JOIN words ON words.rowid=r.id
                   JOIN word_fingerprints w ON w.id=r.id AND w.fingerprint=r.fingerprint"""
        score = "words.rank"
        cte = "WITH words AS MATERIALIZED (SELECT rowid, rank FROM word_fts WHERE word_fts MATCH ?) "
        params = ['"' + tokens.replace('"', '""') + '"', *params]
    elif len(query) >= 3 and "\x00" not in query:
        joins = "JOIN words ON words.rowid=c.id"
        score = "words.rank"
        cte = "WITH words AS MATERIALIZED (SELECT rowid, rank FROM chunk_fts WHERE chunk_fts MATCH ?) "
        match = " AND contains_text(r.body, ?)"
        params = ['"' + query.replace('"', '""') + '"', *params, query]
    else:
        cte = "WITH "
        if query:
            match = " AND r.body LIKE ? ESCAPE '!'"
            params = [*params, literal_like(query)]
    separator = ", " if joins else ""
    chunk_scope = " AND " + visible_chunk(external=True) if external else ""
    rows = db.execute(cte + separator + f"""
        eligible AS (
            SELECT c.id AS chunk, r.id AS record, r.created_at,
                   {score} AS score,
                   row_number() OVER (PARTITION BY c.id ORDER BY {score}, r.created_at, r.id) AS ordinal
            FROM records r JOIN chunk_records cr ON cr.record_id=r.id
            JOIN chunks c ON c.id=cr.chunk_id {joins}
            WHERE {where}{match}{chunk_scope}
        )
        SELECT chunk, record FROM eligible WHERE ordinal=1
        ORDER BY score, created_at DESC, chunk DESC
    """, params)
    return dict(rows)


EXTERNAL_EVIDENCE = f"""
    AND json_type(CASE WHEN json_valid(evidence_ids) THEN evidence_ids ELSE 'null' END)='array'
    AND json_array_length(CASE WHEN json_valid(evidence_ids) THEN evidence_ids ELSE '[]' END)>0
    AND NOT EXISTS (
        SELECT 1 FROM json_each(CASE WHEN json_valid(evidence_ids) THEN evidence_ids ELSE '[]' END) evidence
        WHERE evidence.type <> 'integer' OR NOT EXISTS (
            SELECT 1 FROM records e WHERE e.id=evidence.value AND {visible_record("e", external=True)}
        )
    )
"""


class Query:
    def __init__(self, config, embedder=None, *, profile_name=None, default_external=False):
        self.config = config
        self.profile_name, self.external = profile(config, profile_name, default_external=default_external)
        self.embedder = embedder if embedder is not None else LocalEmbedder(config)
        self.database = Path(config["storage"]["database"])
        self.log = Path(config["storage"]["query_log"])
        if self.log.resolve() == self.database.resolve() or (
            self.log.exists() and self.database.exists() and self.log.samefile(self.database)
        ):
            raise ValueError("Query log must use a separate file")
        self.limits = config["limits"]
        # Leave room for a minimal search item and pagination with 64-bit IDs and counts.
        if self.limits["search"] < 180 or any(self.limits[name] < 128 for name in ("get", "count")):
            raise ValueError("Minimum response limits are 180 for search and 128 for get and count")

    def metadata(self, db):
        if self.external and db.execute("""
            SELECT 1 FROM allow_list WHERE local_storage=1 AND external_ai=0 LIMIT 1
        """).fetchone():
            return {"withheld": True}
        return {}

    @query_tool
    def search(self, db, query="", start=None, end=None, person=None, project=None, source=None,
               target=None, position=0, mode=None, *, _ranked=None):
        if target is None:
            target = "summary" if mode is None else "body"
        if target not in ("summary", "body") or not 0 <= position < 2**63:
            raise ValueError("Invalid search target or position")
        if mode not in (None, "fts", "vector", "hybrid", "word") or (
            mode == "word" and (_ranked is None or self.external)
        ):
            raise ValueError("Invalid search mode")
        scoped_name = scoped_sender_name(db) if self.external else "NULL"
        where, params = filters(start, end, person, project, source, external=self.external, scoped_name=scoped_name)
        metadata = self.metadata(db)
        sender = f"{scoped_name} AS person" if self.external else SENDER
        if target == "body":
            if mode in ("vector", "hybrid"):
                eligible, unembedded, key = vector_rows(
                    db, lambda: vector_key(self.config), where, params, external=self.external
                )
                ranked = cosine_rank(eligible, self.embedder, query, key) if eligible and query.strip() else {}
                if not query.strip() or (not ranked and (eligible or unembedded)):
                    metadata["fallback"] = "fts"
                    mode = "fts"
                elif ranked:
                    unembedded += len(eligible) - len(ranked)
                    if unembedded:
                        metadata["unembedded"] = unembedded
            if mode != "vector":
                lexical = lexical_rows(db, where, params, query, word=mode == "word", external=self.external)
                ranked = fuse(lexical, ranked) if mode == "hybrid" else lexical
            total = len(ranked)
            ordered = list(ranked.items())[position:position + 10]
            rows = []
            for chunk, record in ordered:
                rows.append(db.execute(f"""
                    SELECT 'c:' || ? AS id, r.created_at AS datetime, r.source, r.original_ref,
                           r.title, {sender}, r.body FROM records r WHERE r.id=?
                """, (chunk, record)).fetchone())
        else:
            if mode in ("vector", "hybrid"):
                metadata["fallback"] = "fts"
            evidence = EXTERNAL_EVIDENCE if self.external else ""
            selection = f"""
                WITH matching AS (
                    SELECT target_id, body, row_number() OVER (
                        PARTITION BY target_id ORDER BY generated_at DESC, id DESC
                    ) AS rank
                    FROM ai_extracted_items WHERE target_kind='record' AND state='valid'
                      AND body LIKE ? ESCAPE '!' {evidence}
                )
                SELECT 'r:' || r.id AS id, r.created_at AS datetime, r.source, r.original_ref,
                       r.title, {sender}, item.body AS body
                FROM matching item JOIN records r ON r.id=item.target_id
                WHERE item.rank=1 AND {where}
            """
            params.insert(0, literal_like(query))
            total = db.execute(f"SELECT count(*) FROM ({selection})", params).fetchone()[0]
            rows = db.execute(selection + " ORDER BY datetime DESC, id DESC, body LIMIT 10 OFFSET ?",
                              [*params, position])
        items = []
        for row in rows:
            item = dict(row)
            item["excerpt"] = excerpt(item.pop("body"), query)
            items.append(item)
        if _ranked is not None:
            _ranked(db, [item["id"] for item in items])
        return search_page(items, self.limits["search"], total, position, metadata)

    @query_tool
    def get(self, db, ids, neighbors=0):
        if not 1 <= len(ids) <= 5 or neighbors < 0:
            raise ValueError("Provide one to five IDs and a nonnegative neighbor count")
        cap = self.limits["get"]
        visible = visible_record(external=self.external)
        rows = {}
        context = {}
        for identifier in ids:
            prefix, _, number = identifier.partition(":")
            if prefix not in ("r", "c") or not number.isascii() or not number.isdecimal() or len(number) > 19:
                raise ValueError("Invalid record or chunk ID")
            key = int(number)
            if key >= 2**63:
                raise ValueError("Invalid record or chunk ID")
            if prefix == "r":
                item = db.execute(f"SELECT {RECORD} FROM records r WHERE r.id=? AND {visible}",
                                  (key,)).fetchone()
            else:
                item = db.execute(f"""
                    SELECT 'c:' || c.id AS id, c.starts_at AS datetime, r.source,
                           r.original_ref, r.title, c.body
                    FROM chunks c JOIN records r ON r.id=c.first_record_id
                    WHERE c.id=? AND {visible_chunk(external=self.external)}
                """, (key,)).fetchone()
            if item is None:
                continue
            rows[item["id"]] = dict(item)
            if neighbors:
                anchor = "SELECT ?" if prefix == "r" else "SELECT record_id FROM chunk_records WHERE chunk_id=?"
                for op, order, edge_order in (("<", "DESC", "ASC"), (">", "ASC", "DESC")):
                    neighbors_sql = f"""
                        WITH members AS (SELECT r.* FROM records r WHERE r.id IN ({anchor})),
                        edge AS (SELECT * FROM members ORDER BY created_at {edge_order}, id {edge_order} LIMIT 1)
                        SELECT {RECORD} FROM edge e JOIN records r
                            ON (r.source,r.account,r.channel_id)=(e.source,e.account,e.channel_id)
                        WHERE {visible} AND (r.created_at,r.id) {op} (e.created_at,e.id)
                        ORDER BY r.created_at {order}, r.id {order} LIMIT ?
                    """
                    for row in db.execute(neighbors_sql, (key, min(neighbors, cap))):
                        context.setdefault(row["id"], dict(row))
        rows.update((identifier, row) for identifier, row in context.items() if identifier not in rows)
        return bounded(list(rows.values()), cap, metadata=self.metadata(db))

    @query_tool
    def count(self, db, group_by, start=None, end=None, person=None, project=None, source=None):
        scoped_name = scoped_sender_name(db) if self.external and (group_by == "person" or person is not None) else "NULL"
        person_label = f"min(CASE WHEN l.role='sender' THEN {scoped_name} END)" if self.external else "p.display_name"
        groups = {
            "month": ("substr(r.created_at,1,7)", "substr(r.created_at,1,7)", ""),
            "person": ("p.id", person_label, """JOIN record_person_links l ON l.record_id=r.id
                                                   JOIN persons p ON p.id=l.person_id"""),
            "project": ("p.id", "p.name", "LEFT JOIN projects p ON p.id=r.project_id"),
            "source": ("r.source", "r.source", ""),
        }
        if group_by not in groups:
            raise ValueError("Invalid count axis")
        key, label, join = groups[group_by]
        where, params = filters(start, end, person, project, source, external=self.external, scoped_name=scoped_name)
        rows = [dict(row) for row in db.execute(f"""
            SELECT {key} AS key, {label} AS label, count(DISTINCT r.id) AS count
            FROM records r {join} WHERE {where}
            GROUP BY {key} ORDER BY {key} LIMIT 51
        """, params)]
        result = bounded(rows[:50], self.limits["count"], metadata=self.metadata(db))
        if len(rows) == 51:
            result["has_more"] = True
        return result
