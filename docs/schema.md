# SQLite schema

| README term | SQLite table |
|---|---|
| 記録 | `records` |
| まとまり | `chunks`, `chunk_records` |
| 人物 | `persons` |
| 人物の別名 | `person_aliases` |
| 案件 | `projects` |
| 案件の割り当て規則 | `project_assignment_rules` |
| 出来事 | `events` |
| 記録と人物 | `record_person_links` |
| 記録どうしの関係 | `record_relations` |
| AI抽出情報 | `ai_extracted_items` |
| 修正の記録 | `corrections` |
| 同期の状態 | `sync_state` |
| 許可リスト | `allow_list`; sync applies the Discord configuration |

Discord records are keyed by source, guild and message ID. Times are UTC.
An ID and fingerprint scan under a repeatable-read snapshot finds edits and
deletions, including content edits without an edit timestamp. Only changed rows
are fetched with their bodies. Discord remains the original source.
An empty source snapshot or a loss over 20% of the guild's live records,
with at least five missing records by default, holds all missing IDs while
upserts proceed. Each held ID stays held across syncs until it reappears or a
sync explicitly confirms deletions. The CLI prints a count-only
warning in its JSON output and returns a failure status. Use
`sync discord --confirm-deletions` to apply deletions from the current snapshot.
Sync commits allow-list changes before connecting to PostgreSQL and can
revoke the entire allow-list without connecting.
Revoked guilds stay stored and all three tools omit them until sync re-allows them.
Aliases are shared across guilds and the first nonempty author display name is
kept. Mentions can create aliases before those people post.

Chunks group messages in one channel using their creation times. The initial
five-minute gap keeps short exchanges together. The 2,400-character grouping
cap bounds long conversations and leaves room to return more than one chunk
within the default retrieval budget. A message that exceeds the grouping cap
forms a chunk by itself so its full body stays searchable. The sync compares
chunk contents and membership in affected channels, preserving unchanged IDs
and index entries. Vectors are nullable, unit-normalized little-endian float32
blobs with the model name, dimension, and inference configuration key.
`embed` upgrades older SQLite files when needed and processes new or changed
chunks and vectors with an outdated key. Sync removes a discarded chunk and its
vector together. Model files live under the configured local cache. Only the
`embed` command downloads model files. Query inference uses the local ONNX
graph, tokenizer, the model's search query/document prefixes, and masked mean
pooling across all tokens. Batches are sorted by token length; longer inputs
are split into windows and pooled across windows. The embedding key includes
the SHA-256 of both model files, the token limit, and the prefixes. The default
files are verified against pinned hashes at download and load. The sync
reports the count of visible chunks that need a current vector.

Body search uses literal FTS5 trigram matching. Queries shorter than three
characters use a SQLite LIKE scan. Period filters have inclusive starts and
exclusive ends. Person and project accept a local ID or an exact display name.
Chunk results represent a conversation with at least one record matching the
term and filters. The excerpt and metadata come from that same record. Counts
use distinct active records within each group. Retrieval accepts
`r:<id>` and `c:<id>` IDs, and neighbors come from the same channel.

Search defaults to summaries. Use `--target body` for Discord text, since this
slice generates no summaries. CLI and MCP return the same compact JSON payload
and log only tool name, payload character count and elapsed time.
The search cap must be at least 180 characters to fit a minimal result and
pagination fields even with the largest SQLite IDs and counts.

FTS uses SQLite BM25 when the trigram index can answer the query. Vector mode
scans eligible chunks for cosine similarity. Hybrid mode adds
`1 / (60 + rank)` from each method that found the chunk, using one-based ranks.
Chunk ID breaks score ties. The query reports `"fallback":"fts"` when it
cannot rank the eligible candidates with embeddings for the configured model.
Partial results report `unembedded` as the number of eligible chunks missing
a current vector. Hybrid still ranks lexical matches from those chunks.
In a minimal fallback result, the timestamp may use UTC epoch milliseconds
if the ISO timestamp and fallback marker together would exceed the cap.

The evaluation command builds and refreshes its separate word index only when
enabled. It uses [TinySegmenter](https://pypi.org/project/tinysegmenter/) and
SQLite's unicode61 tokenizer. Hit rate uses the first ten ranked IDs before
the character cap, counting a record ID when a returned chunk contains it.
Evaluation requires a current vector for every visible chunk and warms the
model before measuring either vector method. A SQLite data-version monitor
rejects the report if another connection writes to the database while evaluation
runs. The monitor is conservative and can reject a run after a no-change sync.
P95 uses the nearest-rank percentile. Character counts use the capped
response payload.

Scheduled refresh uses a local nonblocking lock. The sync state entry for
`refresh` records only a phase and exception type on failure. Deletion holds
remain in the per-source sync state and do not fail the refresh process.

Backup retention and verification use the backup tag, the source root, and a
host label derived from the machine ID. Snapshots made before host scoping
were introduced are retained separately and require explicit selection for
manual recovery. The managed snapshot count covers this source only.
