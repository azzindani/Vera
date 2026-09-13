# migrations

! **`dev_tools/pre_embed/schema.sql` is the source of truth for the corpus
schema.** `ingest.py` applies it when a corpus is built. Nothing in this
directory is applied automatically, by the engine or by anything else.

What lives here is what a corpus needs *after* ingestion, applied by hand:

| | |
|---|---|
| `0002_provenance_immutable.sql` | trigger forbidding UPDATEs to provenance columns |
| `0003_rum_text_index.sql` | the RUM index the text arm's ordering is served from |

```bash
psql "$DATABASE_URL" -f migrations/0003_rum_text_index.sql
```

`0001_init.sql` was deleted. It described a system that was never built —
`halfvec(4096)`, a `domains` table with pre-embedded anchors, `chunks`
partitioned by `HASH(cluster_id)`, `NOT NULL` provenance. The live corpus is
`halfvec(1024)`, has no `domains` table, is not partitioned, and has nullable
`source_url`. Keeping a schema file that contradicts the database in every
particular is worse than having none: it is the first thing someone reads.
