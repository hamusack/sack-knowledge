import asyncio
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta, timezone

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from sack_knowledge.mcp_server import server
from sack_knowledge.query import Query, bounded, encode, open_readonly
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
    def unexpected_connection(options):
        pytest.fail("Revoking all guilds must not connect")

    result = sync(config, unexpected_connection)
    assert result["deleted"] == 0 and not result["deletions_held"]
    assert query.search("bravo", target="body")["total"] == 0
    assert query.get(["r:1"])["items"] == []
    assert query.count("source")["items"] == []
    with closing(open_readonly(config["storage"]["database"])) as db:
        assert db.execute("SELECT sum(deleted), count(*) FROM records").fetchone()[:] == (0, 2)
    config["discord"]["guild_ids"] = ["1"]
    sync(config, source.open)
    assert query.count("source")["items"][0]["count"] == 2
    config["storage"]["query_log"] = config["storage"]["database"]
    with pytest.raises(ValueError, match="separate file"):
        Query(config)


@pytest.mark.parametrize(("size", "removed"), [(3, 3), (100, 51)])
def test_bulk_deletion_hold_and_confirmation(setup, size, removed):
    config, source = setup
    for key in range(size):
        source.add(str(key), key, "fixture")
    sync(config, source.open)
    for key in range(removed):
        del source.rows[str(key)]
    if removed < size:
        source.rows[str(removed)]["body"] = "updated fixture"
        source.add(str(size), size, "new fixture")
    result = sync(config, source.open)
    assert result["deleted"] == 0
    assert result["updated"] == result["inserted"] == int(removed < size)
    assert result["deletions_held"] == [{
        "status": "deletions held", "live_local": size, "source_rows": len(source.rows), "missing": removed
    }]
    with closing(open_readonly(config["storage"]["database"])) as db:
        assert json.loads(db.execute("SELECT last_error FROM sync_state").fetchone()[0]) == result["deletions_held"][0]
    result = sync(config, source.open, confirm_deletions=True)
    assert result["deleted"] == removed and not result["deletions_held"]


@pytest.mark.parametrize(("initial", "removed", "added"), [(100, 51, 160), (50, 49, 0)])
def test_deletion_hold_survives_new_rows_and_clears_on_reappearance(setup, initial, removed, added):
    config, source = setup
    for key in range(initial):
        source.add(str(key), key, "fixture")
    sync(config, source.open)
    original = source.rows["0"]
    for key in range(removed):
        del source.rows[str(key)]
    for key in range(initial, initial + added):
        source.add(str(key), key, "fixture")
    result = sync(config, source.open)
    assert result["inserted"] == added and result["deleted"] == 0
    assert result["deletions_held"][0]["live_local"] == initial
    assert result["deletions_held"][0]["missing"] == removed
    result = sync(config, source.open)
    assert result["inserted"] == result["deleted"] == 0
    assert result["deletions_held"][0]["missing"] == removed
    source.rows["0"] = original
    result = sync(config, source.open)
    assert result["deletions_held"][0]["missing"] == removed - 1
    assert result["updated"] == 0
    result = sync(config, source.open, confirm_deletions=True)
    assert result["deleted"] == removed - 1 and not result["deletions_held"]
    with closing(open_readonly(config["storage"]["database"])) as db:
        assert db.execute("SELECT count(*) FROM held_deletions").fetchone()[0] == 0
        assert db.execute("SELECT deleted FROM records WHERE source_id='0'").fetchone()[0] == 0


def test_sync_cli_reports_held_and_passes_confirmation(setup, monkeypatch, capsys):
    from sack_knowledge import __main__ as cli

    config, source = setup
    source.add("1", 0, "fixture")
    sync(config, source.open)
    source.rows.clear()
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "sync", lambda cfg, **options: sync(cfg, source.open, **options))
    args = ["sack_knowledge", "sync", "discord"]
    monkeypatch.setattr("sys.argv", args)
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out)["deletions_held"][0]["missing"] == 1
    monkeypatch.setattr("sys.argv", [*args, "--confirm-deletions"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["deleted"] == 1


def test_allow_list_revocation_survives_source_failure(setup):
    config, source = setup
    source.add("1", 0, "fixture")
    sync(config, source.open)
    query = Query(config)
    chunk = query.search("fixture", target="body")["items"][0]["id"]
    config["discord"]["guild_ids"] = ["2"]

    @contextmanager
    def fail(options):
        with closing(open_readonly(config["storage"]["database"])) as db:
            assert db.execute("SELECT target FROM allow_list WHERE local_storage=1").fetchone()[0] == "2"
        raise RuntimeError("synthetic source failure")
        yield

    with pytest.raises(RuntimeError):
        sync(config, fail)
    assert query.search("fixture", target="body")["items"] == []
    assert query.get(["r:1", chunk], neighbors=1)["items"] == []
    assert query.count("source")["items"] == []
    with closing(open_readonly(config["storage"]["database"])) as db:
        assert db.execute("SELECT deleted FROM records").fetchone()[0] == 0

def test_tight_search_cap_preserves_position(setup):
    config, source = setup
    for key in range(2):
        source.add(str(key), key * 10, "fixture " * 10)
        source.rows[str(key)]["title"] = "X" * 500
    sync(config, source.open)
    config["limits"]["search"] = 180
    query = Query(config)
    first = query.search("fixture", target="body")
    assert len(encode(first)) <= 180 and first["next_position"] == 1
    assert set(first["items"][0]) == {"id", "datetime", "excerpt"}
    second = query.search("fixture", target="body", position=first["next_position"])
    assert len(encode(second)) <= 180
    assert second["items"][0]["id"] != first["items"][0]["id"]
    max_id = 2**63 - 1
    largest = bounded([{"id": f"c:{max_id}", "datetime": "2025-01-01T00:00:00.123456+00:00",
                        "excerpt": "A" * 100}], 180, total=max_id, position=max_id - 2)
    assert largest["items"][0]["id"] == f"c:{max_id}"
    assert largest["next_position"] == max_id - 1 and len(encode(largest)) <= 180


def test_final_search_page_at_exact_cap():
    row = {"id": "c:1", "datetime": "2025-01-01T00:00:00.123456+00:00",
           "excerpt": "A" * 90, "title": ""}
    expected = {"items": [row], "has_more": False, "total": 1, "next_position": None}
    row["title"] = "X" * (500 - len(encode(expected)))
    assert len(encode(expected)) == 500
    assert bounded([row], 500, total=1, position=0) == expected
    compact = bounded([row], 499, total=1, position=0)
    assert [item["id"] for item in compact["items"]] == ["c:1"]
    assert not compact["has_more"] and compact["next_position"] is None


@pytest.mark.parametrize("cap", [180, 250, 500, 900, 2000])
def test_search_pages_return_every_id_once(setup, cap):
    config, source = setup
    for key in range(23):
        source.add(str(key), key * 10, "fixture " * 12)
        source.rows[str(key)]["title"] = "X" * (10 * key)
    sync(config, source.open)
    config["limits"]["search"] = cap
    query = Query(config)
    seen = []
    position = 0
    for _ in range(23):
        result = query.search("fixture", target="body", position=position)
        assert len(encode(result)) <= cap
        assert result["items"]
        seen.extend(item["id"] for item in result["items"])
        if not result["has_more"]:
            break
        assert result["next_position"] == len(seen)
        position = result["next_position"]
    else:
        pytest.fail("Pagination did not terminate")
    assert sorted(seen) == sorted(f"c:{index}" for index in range(1, 24))


def test_summary_search_returns_one_result_per_record(setup):
    config, source = setup
    source.add("1", 0, "fixture")
    sync(config, source.open)
    with sqlite3.connect(config["storage"]["database"]) as db:
        db.executemany("""
            INSERT INTO ai_extracted_items(target_kind, target_id, body, evidence_ids, model, generated_at, state)
            VALUES ('record', 1, ?, '[1]', 'fixture', ?, 'valid')
        """, [("older fixture", "2025-01-01"), ("newer fixture", "2025-01-02")])
    result = Query(config).search("fixture")
    assert result["total"] == 1
    assert result["items"][0]["excerpt"] == "newer fixture"


@pytest.mark.parametrize("term", ['"abc"', "aaa OR bbb", "NEAR(aaa bbb)", "body:aaa"])
def test_fts_special_characters_are_literal(setup, term):
    config, source = setup
    source.add("1", 0, term)
    source.add("2", 15, "aaa bbb abc")
    sync(config, source.open)
    assert Query(config).search(term, target="body")["total"] == 1


def test_deleted_records_and_neighbors_are_hidden_and_times_are_utc(setup):
    config, source = setup
    for key in range(3):
        source.add(str(key), key, "fixture")
    source.rows["0"]["created_at"] = datetime(2025, 1, 1, 9, tzinfo=timezone(timedelta(hours=9)))
    sync(config, source.open)
    query = Query(config)
    assert query.get(["r:1"])["items"][0]["datetime"] == "2025-01-01T00:00:00.000000+00:00"
    del source.rows["1"]
    sync(config, source.open)
    assert query.get(["r:2"])["items"] == []
    assert [item["id"] for item in query.get(["r:1"], neighbors=2)["items"]] == ["r:1", "r:3"]
    assert query.count("source")["items"][0]["count"] == 2


def test_mcp_errors_never_echo_tool_arguments(setup):
    config, _ = setup

    async def call_invalid_tools():
        async with create_connected_server_and_client_session(server(config)) as client:
            for tool, arguments in (
                ("search", {"query": {"synthetic_marker": 123}}),
                ("get", {"ids": ["synthetic_marker"]}),
            ):
                result = await client.call_tool(tool, arguments)
                assert result.isError
                assert [block.text for block in result.content] == ["Tool call failed."]

    asyncio.run(call_invalid_tools())


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
