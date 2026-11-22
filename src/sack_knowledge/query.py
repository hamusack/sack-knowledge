import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import perf_counter


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


def filters(start, end, person, project, source):
    clauses = ["r.deleted=0", "(r.source <> 'discord' OR guild_allowed(r.account))"]
    params = []
    if start is not None:
        clauses.append("r.created_at >= ?")
        params.append(timestamp(start))
    if end is not None:
        clauses.append("r.created_at < ?")
        params.append(timestamp(end))
    if person is not None:
        clauses.append("""EXISTS (
            SELECT 1 FROM record_person_links l JOIN persons p ON p.id=l.person_id
            WHERE l.record_id=r.id AND (CAST(p.id AS TEXT)=? OR p.display_name=?)
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
                db.create_function("guild_allowed", 1, self.guilds.__contains__, deterministic=True)
                db.create_function("contains_text", 2, lambda body, term: term.casefold() in body.casefold(),
                                   deterministic=True)
                db.execute("BEGIN")
                result = function(self, db, *args, **kwargs)
            characters = len(encode(result))
            return result
        finally:
            entry = {"tool": function.__name__, "characters": characters,
                     "elapsed_ms": round((perf_counter() - started) * 1000, 2)}
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("a", encoding="utf-8") as file:
                file.write(encode(entry) + "\n")
    return run


def bounded(rows, cap, *, total=None, position=None):
    result = {"items": [], "has_more": False}
    if total is not None:
        result["total"] = total
        result["next_position"] = None
    for row in rows:
        result["items"].append(row)
        result["has_more"] = True
        if position is not None:
            result["next_position"] = position + len(result["items"])
        if len(encode(result)) <= cap:
            continue
        result["items"].pop()
        field = "body" if "body" in row else "excerpt" if "excerpt" in row else None
        if field and not result["items"]:
            original = row[field]
            row[field] = ""
            row["truncated"] = True
            result["items"].append(row)
            for optional in ("person", "title"):
                if optional in row and len(encode(result)) > cap:
                    row[optional] = None
            if len(encode(result)) <= cap:
                low, high = 0, min(len(original), cap)
                while low < high:
                    middle = (low + high + 1) // 2
                    row[field] = original[:middle]
                    if len(encode(result)) <= cap:
                        low = middle
                    else:
                        high = middle - 1
                row[field] = original[:low]
                return result
            result["items"].pop()
        break
    returned = len(result["items"])
    result["has_more"] = returned < len(rows) or (total is not None and (position or 0) + returned < total)
    if position is not None:
        result["next_position"] = position + max(1, returned) if result["has_more"] else None
    # The true/false and numeric/null variants can have different encoded lengths.
    while len(encode(result)) > cap and result["items"]:
        result["items"].pop()
        result["has_more"] = True
        if position is not None:
            result["next_position"] = position + max(1, len(result["items"]))
    return result


class Query:
    def __init__(self, config):
        self.database = Path(config["storage"]["database"])
        self.log = Path(config["storage"]["query_log"])
        if self.log.resolve() == self.database.resolve() or (
            self.log.exists() and self.database.exists() and self.log.samefile(self.database)
        ):
            raise ValueError("Query log must use a separate file")
        self.guilds = frozenset(config["discord"]["guild_ids"])
        self.limits = config["limits"]
        if any(self.limits[name] < 128 for name in ("search", "get", "count")):
            raise ValueError("Response limits must be at least 128 characters")

    @query_tool
    def search(self, db, query="", start=None, end=None, person=None, project=None, source=None,
               target="summary", position=0):
        if target not in ("summary", "body") or not 0 <= position < 2**63:
            raise ValueError("Invalid search target or position")
        where, params = filters(start, end, person, project, source)
        if target == "body":
            match = ""
            if len(query) >= 3 and "\x00" not in query:
                match = """ AND c.id IN (SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?)
                            AND contains_text(r.body, ?)"""
                params.extend(('"' + query.replace('"', '""') + '"', query))
            elif query:
                match = " AND r.body LIKE ? ESCAPE '!'"
                params.append(literal_like(query))
            selection = f"""
                WITH eligible AS (
                    SELECT c.id AS chunk, r.id AS record,
                           row_number() OVER (PARTITION BY c.id ORDER BY r.created_at, r.id) AS rank
                    FROM records r JOIN chunk_records cr ON cr.record_id=r.id
                    JOIN chunks c ON c.id=cr.chunk_id WHERE {where}{match}
                )
                SELECT 'c:' || c.id AS id, r.created_at AS datetime, r.source, r.original_ref,
                       r.title, {SENDER}, r.body AS body
                FROM eligible e JOIN chunks c ON c.id=e.chunk JOIN records r ON r.id=e.record
                WHERE e.rank=1
            """
        else:
            selection = f"""
                SELECT 'r:' || r.id AS id, r.created_at AS datetime, r.source, r.original_ref,
                       r.title, {SENDER}, item.body AS body
                FROM ai_extracted_items item JOIN records r ON r.id=item.target_id
                WHERE item.target_kind='record' AND item.state='valid' AND {where}
                  AND item.body LIKE ? ESCAPE '!'
            """
            params.append(literal_like(query))
        total = db.execute(f"SELECT count(*) FROM ({selection})", params).fetchone()[0]
        rows = db.execute(selection + " ORDER BY datetime DESC, id DESC, body LIMIT 10 OFFSET ?",
                          [*params, position])
        items = []
        for row in rows:
            item = dict(row)
            item["excerpt"] = excerpt(item.pop("body"), query)
            items.append(item)
        return bounded(items, self.limits["search"], total=total, position=position)

    @query_tool
    def get(self, db, ids, neighbors=0):
        if not 1 <= len(ids) <= 5 or neighbors < 0:
            raise ValueError("Provide one to five IDs and a nonnegative neighbor count")
        cap = self.limits["get"]
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
                item = db.execute(f"""SELECT {RECORD} FROM records r WHERE r.id=? AND NOT r.deleted
                                      AND (r.source <> 'discord' OR guild_allowed(r.account))""",
                                  (key,)).fetchone()
            else:
                item = db.execute("""
                    SELECT 'c:' || c.id AS id, c.starts_at AS datetime, r.source,
                           r.original_ref, r.title, c.body
                    FROM chunks c JOIN records r ON r.id=c.first_record_id
                    WHERE c.id=? AND NOT r.deleted
                      AND (r.source <> 'discord' OR guild_allowed(r.account))
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
                        WHERE NOT r.deleted AND (r.created_at,r.id) {op} (e.created_at,e.id)
                        ORDER BY r.created_at {order}, r.id {order} LIMIT ?
                    """
                    for row in db.execute(neighbors_sql, (key, min(neighbors, cap))):
                        context.setdefault(row["id"], dict(row))
        rows.update((identifier, row) for identifier, row in context.items() if identifier not in rows)
        return bounded(list(rows.values()), cap)

    @query_tool
    def count(self, db, group_by, start=None, end=None, person=None, project=None, source=None):
        groups = {
            "month": ("substr(r.created_at,1,7)", "substr(r.created_at,1,7)", ""),
            "person": ("p.id", "p.display_name", """JOIN record_person_links l ON l.record_id=r.id
                                                   JOIN persons p ON p.id=l.person_id"""),
            "project": ("p.id", "p.name", "LEFT JOIN projects p ON p.id=r.project_id"),
            "source": ("r.source", "r.source", ""),
        }
        if group_by not in groups:
            raise ValueError("Invalid count axis")
        key, label, join = groups[group_by]
        where, params = filters(start, end, person, project, source)
        rows = [dict(row) for row in db.execute(f"""
            SELECT {key} AS key, {label} AS label, count(DISTINCT r.id) AS count
            FROM records r {join} WHERE {where}
            GROUP BY {key} ORDER BY {key} LIMIT 51
        """, params)]
        result = bounded(rows[:50], self.limits["count"])
        if len(rows) == 51:
            result["has_more"] = True
        return result
