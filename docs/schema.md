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
| 許可リスト | `allow_list`; ingestion scope comes from the Discord configuration |

Discord records are keyed by source, guild and message ID. Times are UTC.
An ID and fingerprint scan under a repeatable-read snapshot finds edits and
deletions, including content edits without an edit timestamp. Only changed rows
are fetched with their bodies. Discord remains the original source.
Aliases are shared across guilds and the first nonempty author display name is
kept. Mentions can create aliases before those people post.

Chunks group messages in one channel using their creation times. The initial
five-minute gap keeps short exchanges together. The 2,400-character grouping
cap bounds long conversations and leaves room to return more than one chunk
within the default retrieval budget. A message that exceeds the grouping cap
forms a chunk by itself so its full body stays searchable. The sync compares
chunk contents and membership in affected channels, preserving unchanged IDs
and index entries. Embedding fields are nullable.

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
