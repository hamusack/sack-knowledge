import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta

import pytest

from sack_knowledge.query import Query, encode, open_readonly
from sack_knowledge.sync import sync


class Source:
    def __init__(self):
        self.rows = {}
        self.fetched = []

    @contextmanager
    def open(self, options):
        yield self

    def manifest(self, guild):
        assert guild == "1"
        return {key: repr(row) for key, row in self.rows.items()}

    def messages(self, guild, ids):
        self.fetched.extend(ids)
        return (self.rows[key] for key in ids)

    def add(self, key, minute, body, *, channel="2", author="3", reply=None, mentions=()):
        self.rows[key] = dict(source_id=key, channel_id=channel, author_id=author,
                              author_name=f"Person {author}", reply_id=reply, body=body,
                              mentions=mentions, created_at=datetime(2025, 1, 1, tzinfo=UTC)
                              + timedelta(minutes=minute), edited_at=None, title="Channel")


@pytest.fixture
def setup(tmp_path):
    config = {
        "storage": {"database": tmp_path / "x.sqlite", "query_log": tmp_path / "queries.jsonl"},
        "discord": {"guild_ids": ["1"], "gap_seconds": 300, "chunk_characters": 2400},
        "limits": {"search": 2000, "get": 6000, "count": 2000},
    }
    return config, Source()


def test_sync_edit_delete_and_idempotency(setup):
    config, source = setup
    source.add("10", 0, "alpha", mentions=("4",))
    source.add("11", 1, "bravo", reply="10", author="4")
    source.add("12", 30, "charlie")
    assert sync(config, source.open)["chunks_built"] == 2
    source.fetched.clear()
    repeat = sync(config, source.open)
    assert all(repeat[key] == 0 for key in ("inserted", "updated", "deleted", "chunks_built", "chunks_removed"))
    assert source.fetched == []
    query = Query(config)
    stable = query.search("charlie", target="body")["items"][0]["id"]
    source.rows["11"]["body"] = "delta"
    edit = sync(config, source.open)
    assert (edit["updated"], edit["chunks_built"], edit["chunks_removed"]) == (1, 1, 1)
    assert source.fetched == ["11"]
    assert query.search("bravo", target="body")["total"] == 0
    assert query.search("delta", target="body")["total"] == 1
    assert query.search("charlie", target="body")["items"][0]["id"] == stable
    del source.rows["10"]
    assert sync(config, source.open)["deleted"] == 1
    assert query.search("alpha", target="body")["total"] == 0
    with closing(open_readonly(config["storage"]["database"])) as db:
        assert db.execute("SELECT deleted FROM records WHERE source_id='10'").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM record_relations").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM persons").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM record_person_links WHERE role='mention'").fetchone()[0] == 1
        db.execute("PRAGMA query_only=OFF")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM records")


def test_search_filters_caps_and_neighbors(setup):
    config, source = setup
    for key in range(12):
        source.add(str(key), key * 10, " abc %_ " + "x" * 190,
                   author="3" if key % 2 else "4")
    sync(config, source.open)
    with sqlite3.connect(config["storage"]["database"]) as db:
        db.execute("INSERT INTO projects(id, name) VALUES (1, 'Sample')")
        db.execute("UPDATE records SET project_id=1 WHERE source_id='11'")
    config["limits"] = {"search": 500, "get": 240, "count": 128}
    query = Query(config)
    first = query.search("abc", target="body")
    assert first["total"] == 12 and first["has_more"] and len(encode(first)) <= 500
    next_page = query.search("abc", target="body", position=first["next_position"])
    assert next_page["items"][0]["id"] != first["items"][0]["id"]
    assert query.search("%_", target="body", start="2025-01-01T01:00:00",
                        end="2025-01-01T02:00:00", project="Sample", person="Person 3",
                        source="discord")["total"] == 1
    assert query.search("abc", target="body", person="Person 4")["total"] == 6
    assert query.search("abc", target="body", source="other")["total"] == 0
    result = query.get([first["items"][0]["id"]], neighbors=1)
    assert result["has_more"] and result["items"][0]["truncated"] and len(encode(result)) <= 240
    assert query.count("month", source="discord")["items"][0]["count"] == 12
    with pytest.raises(ValueError):
        query.get(["r:1"] * 6)
    entries = [json.loads(line) for line in config["storage"]["query_log"].read_text().splitlines()]
    assert set(entries[-1]) == {"tool", "characters", "elapsed_ms"}


def test_get_chunk_neighbors_outside_both_edges(setup):
    config, source = setup
    for key, minute in enumerate((0, 10, 11, 30)):
        source.add(str(key), minute, "marker" if key in (1, 2) else "outside")
    sync(config, source.open)
    query = Query(config)
    chunk = query.search("marker", target="body")["items"][0]["id"]
    result = query.get([chunk], neighbors=1)
    assert [item["body"] for item in result["items"]] == ["marker\nmarker", "outside", "outside"]


def test_filtered_match_and_allow_list(setup):
    config, source = setup
    source.add("10", 0, "alpha")
    source.add("11", 1, "bravo", author="4")
    assert sync(config, source.open)["chunks_built"] == 1
    query = Query(config)
    assert query.search("bravo", target="body", person="Person 3")["total"] == 0
    hit = query.search("bravo", target="body", person="Person 4")["items"][0]
    assert hit["person"] == "Person 4" and hit["excerpt"] == "bravo"
    config["discord"]["gap_seconds"] = 10
    assert sync(config, source.open)["chunks_built"] == 2
    config["discord"]["guild_ids"] = []
    query = Query(config)
    assert query.search("bravo", target="body")["total"] == 0
    assert query.get(["r:1"])["items"] == []
    assert query.count("source")["items"] == []
    config["storage"]["query_log"] = config["storage"]["database"]
    with pytest.raises(ValueError, match="separate file"):
        Query(config)


def test_sync_failure_does_not_commit_partial_change(setup):
    config, source = setup
    source.add("1", 0, "test")
    sync(config, source.open)

    @contextmanager
    def fail(options):
        yield source
        raise RuntimeError("synthetic failure details")

    source.rows["1"]["body"] = "changed"
    with pytest.raises(RuntimeError):
        sync(config, fail)
    with closing(open_readonly(config["storage"]["database"])) as db:
        assert db.execute("SELECT body FROM records").fetchone()[0] == "test"
        assert db.execute("SELECT last_error FROM sync_state").fetchone()[0] == "RuntimeError"
