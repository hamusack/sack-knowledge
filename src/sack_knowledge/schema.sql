CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, activity TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, starts_at TEXT NOT NULL, ends_at TEXT, title TEXT
);
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL, account TEXT NOT NULL, source_id TEXT NOT NULL,
    kind TEXT NOT NULL, created_at TEXT NOT NULL, fetched_at TEXT NOT NULL,
    edited_at TEXT, title TEXT, body TEXT NOT NULL, activity TEXT,
    deleted INTEGER NOT NULL DEFAULT 0, original_ref TEXT NOT NULL,
    channel_id TEXT NOT NULL, fingerprint TEXT NOT NULL, reply_source_id TEXT,
    project_id INTEGER REFERENCES projects(id), event_id INTEGER REFERENCES events(id),
    UNIQUE(source, account, source_id)
);
CREATE INDEX IF NOT EXISTS record_channel ON records(source, account, channel_id, deleted, created_at, id);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_record_id INTEGER NOT NULL UNIQUE REFERENCES records(id),
    starts_at TEXT NOT NULL, ends_at TEXT NOT NULL, body TEXT NOT NULL,
    embedding BLOB, embedding_model TEXT, embedding_dimension INTEGER, embedding_key TEXT
);
CREATE TABLE IF NOT EXISTS chunk_records (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    record_id INTEGER NOT NULL REFERENCES records(id),
    PRIMARY KEY(chunk_id, record_id)
);
CREATE INDEX IF NOT EXISTS chunk_membership ON chunk_records(record_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(body, tokenize='trigram');
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY, display_name TEXT NOT NULL, note TEXT
);
CREATE TABLE IF NOT EXISTS person_aliases (
    source TEXT NOT NULL, source_id TEXT NOT NULL, person_id INTEGER NOT NULL REFERENCES persons(id),
    PRIMARY KEY(source, source_id)
);
CREATE TABLE IF NOT EXISTS project_assignment_rules (
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, target TEXT NOT NULL,
    project_id INTEGER NOT NULL REFERENCES projects(id)
);
CREATE TABLE IF NOT EXISTS record_person_links (
    record_id INTEGER NOT NULL REFERENCES records(id), person_id INTEGER NOT NULL REFERENCES persons(id),
    role TEXT NOT NULL, PRIMARY KEY(record_id, person_id, role)
);
CREATE TABLE IF NOT EXISTS record_relations (
    record_id INTEGER NOT NULL REFERENCES records(id), related_record_id INTEGER NOT NULL REFERENCES records(id),
    kind TEXT NOT NULL, PRIMARY KEY(record_id, related_record_id, kind)
);
CREATE TABLE IF NOT EXISTS ai_extracted_items (
    id INTEGER PRIMARY KEY, target_kind TEXT NOT NULL, target_id INTEGER NOT NULL,
    body TEXT NOT NULL, evidence_ids TEXT NOT NULL, model TEXT NOT NULL,
    generated_at TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('valid','invalid','pending'))
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY, corrected_at TEXT NOT NULL, target TEXT NOT NULL, rule TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sync_state (
    source TEXT NOT NULL, account TEXT NOT NULL, token TEXT,
    last_success TEXT, last_error TEXT, PRIMARY KEY(source, account)
);
CREATE TABLE IF NOT EXISTS held_deletions (
    record_id INTEGER PRIMARY KEY REFERENCES records(id)
);
CREATE TABLE IF NOT EXISTS allow_list (
    source TEXT NOT NULL, target TEXT NOT NULL, local_storage INTEGER NOT NULL,
    external_ai INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(source, target)
);
