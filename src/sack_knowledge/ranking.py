from .embedding import allowed_chunk, unit_vectors


RRF_CONSTANT = 60


def vector_rows(db, key, where, params):
    if not params:
        candidates = f"SELECT c.id, c.first_record_id AS record FROM chunks c WHERE {allowed_chunk()}"
    else:
        candidates = f"""
            SELECT c.id, min(r.id) AS record FROM chunks c
            JOIN chunk_records cr ON cr.chunk_id=c.id JOIN records r ON r.id=cr.record_id
            WHERE {where} GROUP BY c.id
        """
    if not any(row[1] == "embedding_key" for row in db.execute("PRAGMA table_info(chunks)")):
        return [], db.execute(f"SELECT count(*) FROM ({candidates})", params).fetchone()[0]
    rows = db.execute(f"""
        WITH eligible AS ({candidates})
        SELECT e.id, e.record, c.embedding_dimension, c.embedding
        FROM eligible e LEFT JOIN chunks c ON c.id=e.id AND c.embedding_key=?
    """, [*params, key]).fetchall()
    embedded = [row for row in rows if row["embedding"] is not None]
    return embedded, len(rows) - len(embedded)


def cosine_rank(rows, embedder, query, key):
    import numpy as np

    needle = unit_vectors(embedder.encode([query], query=True, key=key))[0]
    matching = [row for row in rows if row["embedding_dimension"] == len(needle)
                and len(row["embedding"]) == needle.nbytes]
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
