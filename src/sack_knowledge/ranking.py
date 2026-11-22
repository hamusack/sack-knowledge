from .embedding import allowed_chunk, unit_vectors, vector_key


RRF_CONSTANT = 60


def vector_rows(db, config, where, params):
    if not any(row[1] == "embedding_key" for row in db.execute("PRAGMA table_info(chunks)")):
        return None
    key = vector_key(config)
    if not params:
        rows = db.execute(f"""
            SELECT c.id, c.first_record_id AS record, c.embedding_dimension, c.embedding
            FROM chunks c WHERE c.embedding_key=? AND c.embedding IS NOT NULL AND {allowed_chunk()}
        """, (key,)).fetchall()
        return rows or None
    rows = db.execute(f"""
        SELECT c.id, min(r.id) AS record, c.embedding_dimension, c.embedding
        FROM chunks c
        JOIN chunk_records cr ON cr.chunk_id=c.id JOIN records r ON r.id=cr.record_id
        WHERE c.embedding_key=? AND c.embedding IS NOT NULL AND {where}
        GROUP BY c.id
    """, [key, *params]).fetchall()
    if rows:
        return rows
    available = db.execute(f"""
        SELECT 1 FROM chunks c WHERE c.embedding_key=? AND c.embedding IS NOT NULL AND {allowed_chunk()}
        LIMIT 1
    """, (key,)).fetchone()
    return [] if available else None


def cosine_rank(rows, embedder, query):
    import numpy as np

    needle = unit_vectors(embedder.encode([query], query=True))[0]
    matching = [row for row in rows if row["embedding_dimension"] == len(needle)]
    if not matching:
        return {}
    matrix = np.frombuffer(b"".join(row["embedding"] for row in matching),
                           dtype="<f4").reshape(len(matching), len(needle))
    scores = np.einsum("ij,j->i", matrix, needle)
    order = sorted(zip(scores, matching), key=lambda pair: (-pair[0], pair[1]["id"]))
    return {row["id"]: row["record"] for _, row in order}


def fuse(lexical, semantic):
    scores = {}
    for ranking in (lexical, semantic):
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0) + 1 / (RRF_CONSTANT + rank)
    return {identifier: lexical.get(identifier, semantic.get(identifier))
            for identifier in sorted(scores, key=lambda identifier: (-scores[identifier], identifier))}
