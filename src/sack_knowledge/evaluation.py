import json
import math
import sqlite3
from contextlib import closing
from functools import cache
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from .query import Query, VISIBLE, encode, filters


METHODS = ("fts", "word", "vector", "hybrid")
FILTERS = {"start", "end", "person", "project", "source"}


@cache
def tokenizer():
    from tinysegmenter import TinySegmenter

    return TinySegmenter()


def segment(text):
    return " ".join(token for token in tokenizer().tokenize(text) if any(char.isalnum() for char in token))


def build_word_index(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True, timeout=60)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS word_fts USING fts5(body, tokenize='unicode61')")
        db.execute("CREATE TABLE IF NOT EXISTS word_fingerprints (id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL)")
        db.execute(f"""
            DELETE FROM word_fts WHERE rowid IN (
                SELECT w.id FROM word_fingerprints w LEFT JOIN records r ON r.id=w.id
                WHERE r.id IS NULL OR NOT ({VISIBLE}) OR w.fingerprint <> r.fingerprint
            )
        """)
        db.execute("""
            DELETE FROM word_fingerprints WHERE NOT EXISTS (SELECT 1 FROM word_fts WHERE rowid=word_fingerprints.id)
        """)
        rows = db.execute(f"""
            SELECT r.id, r.body, r.fingerprint FROM records r
            LEFT JOIN word_fingerprints w ON w.id=r.id WHERE w.id IS NULL AND {VISIBLE}
        """).fetchall()
        db.executemany("INSERT INTO word_fts(rowid, body) VALUES (?, ?)",
                       ((identifier, segment(body)) for identifier, body, _ in rows))
        db.executemany("INSERT INTO word_fingerprints VALUES (?, ?)",
                       ((identifier, fingerprint) for identifier, _, fingerprint in rows))


def questions(path):
    with Path(path).open(encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]
    if not entries:
        raise ValueError("Evaluation questions are empty")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid evaluation question")
        text, answers, filters = entry.get("question"), entry.get("answers"), entry.get("filters", {})
        if (not isinstance(text, str) or not text.strip() or not isinstance(answers, list) or not answers
                or not all(isinstance(value, str) and value[:2] in ("c:", "r:")
                           and value[2:].isascii() and value[2:].isdigit() for value in answers)
                or not isinstance(filters, dict) or filters.keys() - FILTERS
                or not all(isinstance(value, str) or value is None for value in filters.values())):
            raise ValueError("Invalid evaluation question")
        yield text, set(answers), filters


def hit_in_top_ten(db, identifiers, answers, options):
    if answers.intersection(identifiers):
        return True
    chunk_ids = json.dumps([int(identifier[2:]) for identifier in identifiers])
    where, params = filters(**dict.fromkeys(FILTERS) | options)
    return any(f"r:{row[0]}" in answers for row in db.execute(f"""
        SELECT r.id FROM chunk_records cr JOIN records r ON r.id=cr.record_id
        WHERE cr.chunk_id IN (SELECT value FROM json_each(?)) AND {where}
    """, [chunk_ids, *params]))


def evaluate(config, path, embedder=None):
    if not config.get("evaluation", {}).get("word_index"):
        raise ValueError("Enable evaluation.word_index to compare all four methods")
    entries = list(questions(path))
    build_word_index(config["storage"]["database"])
    search = Query(config, embedder)
    if "fallback" in search.search(entries[0][0], mode="vector"):
        raise ValueError("Build embeddings before evaluation")
    report = []
    for method in METHODS:
        hits = []
        elapsed = []
        characters = []
        for text, answers, options in entries:
            scoring_ms = 0

            def score(db, ids):
                nonlocal scoring_ms
                started = perf_counter()
                hits.append(hit_in_top_ten(db, ids, answers, options))
                scoring_ms = (perf_counter() - started) * 1000

            started = perf_counter()
            response = search.search(text, target="body", mode=method, **options, _ranked=score)
            elapsed.append((perf_counter() - started) * 1000 - scoring_ms)
            characters.append(len(encode(response)))
            if "fallback" in response:
                raise ValueError("Build embeddings before evaluation")
        report.append({
            "method": method, "hit_rate": mean(hits), "median_ms": median(elapsed),
            "p95_ms": sorted(elapsed)[math.ceil(0.95 * len(elapsed)) - 1], "mean_characters": mean(characters),
        })
    return report


def table(report):
    header = "| Method | Hit rate @10 | Median ms | P95 ms | Mean characters |"
    separator = "|---|---:|---:|---:|---:|"
    labels = {"fts": "Trigram FTS", "word": "Word FTS", "vector": "Vector", "hybrid": "Hybrid"}
    rows = [
        f"| {labels[row['method']]} | {row['hit_rate']:.1%} | {row['median_ms']:.2f} | "
        f"{row['p95_ms']:.2f} | {row['mean_characters']:.1f} |"
        for row in report
    ]
    return "\n".join([header, separator, *rows])
