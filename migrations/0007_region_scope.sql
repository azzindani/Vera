-- The region a question names, as something the lexical arm can filter on.
--
-- A question that names a region ("retribusi reklame di Kabupaten Bangka Tengah") is
-- answered by that region's regulation, and the global arms rarely find it: hundreds of
-- regions enact near-identical clauses, and the region's name in the query pulls in that
-- region's OTHER chunks. Measured on the 1.7M corpus, 604 cases (`ROADMAP.md` Stage 3):
-- 137 cases name the answer's region and only 54 have it in the top 5, against 62% for the
-- rest; in 73 of the 83 misses the named region IS in the top 10, with the wrong clause.
-- A search restricted to the region puts the answer in the top 5 for 29 of the 83.
--
-- Two things, both facts about the corpus:
--
-- * `chunks_bm25` is rebuilt with `enacting_body` as a literal (untokenized) field, so the
--   engine's region arm is one Tantivy query: `body` matched AND `enacting_body` in the
--   named set, top-k served from the index. ✗ A heap filter: measured at 940 ms for the
--   `indexable` filter, against 35 ms for the index's own top-k (`store/src/search.rs`).
--   The `body` field is indexed exactly as before, so the sparse arm's scores do not move.
--
-- * `corpus_regions` lists the enacting bodies of regional instruments, which is what the
--   engine matches a question against. Generic words the extractor left in the column
--   (`KABUPATEN`, `KOTA`, ...) and two-letter artefacts (`MA`, `RI`) are excluded: they
--   would scope every question that says "kabupaten" (`SCORING.md` on `enacting_body`).
--
-- ! OPTIONAL, like 0004: without the field or the table the region arm does not run, and
-- the engine answers exactly as before. Apply it AFTER 0004, where pg_search is installed.
-- ! Re-apply after a corpus load or a delta that adds regions.
--
--     psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/0007_region_scope.sql

DROP INDEX IF EXISTS chunks_bm25;
CREATE INDEX chunks_bm25
    ON chunks USING bm25 (id, body, (enacting_body::pdb.literal))
    WITH (key_field = 'id');

DROP TABLE IF EXISTS corpus_regions;
CREATE TABLE corpus_regions AS
SELECT enacting_body, count(*)::bigint AS chunks
  FROM chunks
 WHERE enacting_body IS NOT NULL
   AND length(enacting_body) >= 4
   AND coalesce(regulation_type, '') !~ '(UNDANG|PEMERINTAH|PRESIDEN|MENTERI|BADAN)'
   AND enacting_body !~ '^(REPUBLIK INDONESIA|INDONESIA|KABUPATEN|KOTA|PROVINSI|DAERAH|PEMERINTAH|NEGARA|N/A)$'
 GROUP BY enacting_body;
ALTER TABLE corpus_regions ADD PRIMARY KEY (enacting_body);

ANALYZE chunks;
