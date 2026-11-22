import asyncio
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from sack_knowledge.embedding import embed
from sack_knowledge.errors import ModelIntegrityError
from sack_knowledge.mcp_server import server
from sack_knowledge.query import Query
from sack_knowledge.sync import sync


class Source:
    def __init__(self):
        self.rows = {"1": {}, "2": {}}

    def add(self, guild, key, minute, author):
        self.rows[guild][key] = {
            "source_id": key, "channel_id": "1", "author_id": author, "author_name": f"Fixture {author}",
            "reply_id": None, "body": "fixture", "mentions": [], "title": "Fixture",
            "created_at": datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=minute), "edited_at": None,
        }

    @contextmanager
    def open(self, options):
        yield self

    def manifest(self, guild):
        return {key: repr(row) for key, row in self.rows[guild].items()}

    def messages(self, guild, ids):
        return (self.rows[guild][key] for key in ids)

    def sender_names(self, guild, ids):
        return (self.rows[guild][key] for key in ids)


class Embedder:
    def lengths(self, texts, *, key):
        return [len(text) for text in texts]

    def encode(self, texts, *, key, query=False):
        return [[1, 0] for _ in texts]


@pytest.fixture
def corpus(tmp_path):
    config = {
        "storage": {"database": tmp_path / "test.sqlite3", "query_log": tmp_path / "log.jsonl"},
        "discord": {"guild_ids": ["1", "2"], "external_ai": {"1": True}, "gap_seconds": 300,
                    "chunk_characters": 2400},
        "profiles": {"onsite": {"external": False}, "remote": {"external": True}},
        "embedding": {"cache": str(tmp_path / "models")},
        "limits": {"search": 180, "get": 6000, "count": 2000},
    }
    source = Source()
    source.add("1", "10", 0, "3")
    source.add("1", "11", 15, "3")
    source.add("2", "20", 30, "4")
    source.add("2", "21", 45, "4")
    sync(config, source.open)
    model = Embedder()
    embed(config, model)
    return config, source, model


@pytest.mark.parametrize("mode", ["fts", "vector", "hybrid"])
def test_external_pages_and_totals_only_include_cleared_chunks(corpus, mode):
    config, source, model = corpus
    external = Query(config, model, profile_name="remote")
    before = external.search("fixture", mode=mode)
    assert before["withheld"] and before["total"] == 2 and before["has_more"]
    after = external.search("fixture", mode=mode, position=before["next_position"])
    assert after["total"] == 2 and not after["has_more"]
    assert {item["id"] for item in before["items"] + after["items"]} == {"c:1", "c:2"}
    source.add("2", "22", 60, "4")
    sync(config, source.open)
    assert external.search("fixture", mode=mode) == before
    hidden = external.search("fixture", mode=mode, person="Fixture 4")
    assert hidden == {"items": [], "total": 0, "has_more": False, "next_position": None, "withheld": True}
    local = Query(config, model, profile_name="onsite").search("fixture", mode=mode)
    assert local["total"] >= 4 and "withheld" not in local


def test_external_fts_matches_local_relevance_order_when_all_entries_are_cleared(corpus):
    config, source, _ = corpus
    config["discord"]["external_ai"] = {"1": True, "2": True}
    config["limits"]["search"] = 2000
    source.rows["1"]["10"]["body"] = "fixture fixture fixture fixture"
    sync(config, source.open)
    local = Query(config).search("fixture", mode="fts")
    external = Query(config, profile_name="remote").search("fixture", mode="fts")
    order = [row["id"] for row in local["items"]]
    assert local == external and len(order) == 4
    assert order[0] == Query(config).search("fixture fixture", mode="fts")["items"][0]["id"]


def test_external_fallback_and_pending_count_only_use_cleared_chunks(corpus):
    config, source, model = corpus
    source.add("2", "22", 60, "4")
    source.add("1", "12", 75, "5")
    sync(config, source.open)
    external = Query(config, model, profile_name="remote")
    for mode in ("vector", "hybrid"):
        whole = external.search("fixture", mode=mode)
        assert whole["unembedded"] == 1
        assert whole["total"] == (2 if mode == "vector" else 3)
        cleared_waiting = external.search("fixture", mode=mode, person="Fixture 5")
        assert cleared_waiting["fallback"] == "fts" and cleared_waiting["total"] == 1
        denied = external.search("fixture", mode=mode, person="Fixture 4")
        assert denied["total"] == 0 and "fallback" not in denied and "unembedded" not in denied


def test_get_neighbors_count_and_mixed_chunks_and_summaries(corpus):
    config, _, model = corpus
    query = Query(config, model, profile_name="remote")
    result = query.get(["r:1", "r:3", "c:4"], neighbors=5)
    assert {item["id"] for item in result["items"]} == {"r:1", "r:2"} and not result["has_more"]
    assert query.get(["r:3", "c:4", "r:999"], neighbors=5)["items"] == []
    assert query.count("source")["items"][0]["count"] == 2
    assert query.count("person", person="Fixture 4")["items"] == []
    with closing(sqlite3.connect(config["storage"]["database"])) as db, db:
        db.execute("INSERT INTO chunk_records(chunk_id, record_id) VALUES (1, 3)")
        db.executemany("""
            INSERT INTO ai_extracted_items(target_kind, target_id, body, evidence_ids, model, generated_at, state)
            VALUES ('record', ?, 'fixture', ?, 'fixture', '2025-01-01', 'valid')
        """, ((1, "[1,3]"), (2, "[2]"), (3, "[3]")))
    assert query.get(["c:1"], neighbors=5)["items"] == []
    for mode in ("fts", "vector", "hybrid"):
        assert [item["id"] for item in query.search("fixture", mode=mode)["items"]] == ["c:2"]
        assert query.search("fixture", mode=mode, person="Fixture 4")["items"] == []
    assert [item["id"] for item in query.search("fixture")["items"]] == ["r:2"]


def test_running_external_query_observes_revocation_even_when_sync_fails(corpus):
    config, source, model = corpus
    external = Query(config, model, profile_name="remote")
    local = Query(config, model)
    assert external.count("source")["items"][0]["count"] == 2
    config["discord"]["external_ai"] = {"2": True}

    @contextmanager
    def fail(options):
        assert external.get(["r:1", "c:1"], neighbors=5)["items"] == []
        raise RuntimeError("fixture")
        yield source

    with pytest.raises(RuntimeError):
        sync(config, fail)
    assert external.search("fixture", mode="hybrid")["total"] == 2
    assert {item["id"] for item in external.get(["r:3"], neighbors=5)["items"]} == {"r:3", "r:4"}
    assert local.count("source")["items"][0]["count"] == 4


def test_external_sender_names_only_use_visible_records(corpus):
    config, source, model = corpus
    config["discord"]["external_ai"] = {"2": True}
    source.rows["2"]["20"].update(author_id="3", author_name="Cleared fixture")
    source.rows["2"]["21"].update(author_id="3", author_name="Cleared fixture")
    sync(config, source.open)
    config["limits"]["search"] = 2000
    external = Query(config, model, profile_name="remote")
    assert external.search("fixture", mode="fts", person="Fixture 3")["total"] == 0
    allowed = external.search("fixture", mode="fts", person="Cleared fixture")
    assert allowed["total"] == 2 and {item["person"] for item in allowed["items"]} == {"Cleared fixture"}
    assert external.count("person")["items"][0]["label"] == "Cleared fixture"
    assert Query(config).count("person")["items"][0]["label"] == "Fixture 3"


def test_existing_sender_metadata_backfills_without_updating_records(corpus):
    config, source, _ = corpus
    with closing(sqlite3.connect(config["storage"]["database"])) as db, db:
        db.execute("DELETE FROM record_sender_names WHERE record_id=1")
    assert sync(config, source.open)["updated"] == 0
    external = Query(config, profile_name="remote")
    assert external.search("fixture", mode="fts", person="Fixture 3")["total"] == 2


def test_hidden_candidates_do_not_trigger_model_errors(corpus, monkeypatch):
    config, _, model = corpus

    def bad_model(config):
        raise ModelIntegrityError()

    monkeypatch.setattr("sack_knowledge.query.vector_key", bad_model)
    external = Query(config, model, profile_name="remote")
    assert external.search("fixture", mode="vector", person="Fixture 4")["total"] == 0
    with pytest.raises(ModelIntegrityError):
        external.search("fixture", mode="vector", person="Fixture 3")


@pytest.mark.parametrize(("requested", "logged", "external", "count"), [
    (None, "external", True, 2), ("unknown", "external", True, 2),
    ("onsite", "onsite", False, 4), ("remote", "remote", True, 2),
])
def test_mcp_profiles_are_resolved_at_start_and_logged(corpus, requested, logged, external, count):
    config, _, _ = corpus

    async def call():
        async with create_connected_server_and_client_session(server(config, profile_name=requested)) as client:
            result = await client.call_tool("count", {"group_by": "source"})
            assert not result.isError
            assert json.loads(result.content[0].text)["items"][0]["count"] == count

    asyncio.run(call())
    entry = json.loads(config["storage"]["query_log"].read_text().splitlines()[-1])
    assert set(entry) == {"tool", "characters", "elapsed_ms", "external", "profile"}
    assert (entry["profile"], entry["external"]) == (logged, external)
