//! SQLite backend.
//!
//! Not the production store — Postgres + pgvector is (`migrations/`). This
//! exists because the corpus under test ships as SQLite and because it lets the
//! routing experiment run with no server, no container and no network, which is
//! what makes the layer-by-layer latency numbers reproducible.
//!
//! Vectors are stored as little-endian f32 BLOBs and BM25 comes from FTS5,
//! which SQLite implements natively. Neither choice leaks past [`ChunkStore`].

use crate::{
    Centroid, ChunkStore, DomainAnchor, Edge, KeywordHit, ScannedRow, StoreError,
    decode_vector_into,
};
use rusqlite::{Connection, OpenFlags, params, params_from_iter};
use std::sync::Mutex;
use vera_core::{Chunk, EmbeddingSpace};

impl From<rusqlite::Error> for StoreError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Backend(e.to_string())
    }
}

impl From<rusqlite::types::FromSqlError> for StoreError {
    fn from(e: rusqlite::types::FromSqlError) -> Self {
        Self::Malformed(e.to_string())
    }
}

/// DDL for a Vera SQLite corpus · mirrors `migrations/0001_init.sql`.
///
/// ! `chunks_fts` is an FTS5 **external-content** table: it indexes `body`
/// without storing a second copy of it. On a 750K-row corpus that is the
/// difference between a few hundred MB of index and doubling the whole file.
pub const SCHEMA: &str = r"
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS domains (
    id          TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    anchor      BLOB NOT NULL,
    row_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS clusters (
    id         INTEGER PRIMARY KEY,
    domain_id  TEXT NOT NULL REFERENCES domains(id),
    centroid   BLOB NOT NULL,
    row_count  INTEGER NOT NULL DEFAULT 0,
    generation INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS clusters_domain_idx ON clusters (domain_id, generation);

CREATE TABLE IF NOT EXISTS chunks (
    rowid           INTEGER PRIMARY KEY,
    id              TEXT NOT NULL UNIQUE,
    domain_id       TEXT NOT NULL,
    cluster_id      INTEGER NOT NULL,
    body            TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    source_title    TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    locator_page    INTEGER,
    locator_section TEXT,
    heading_path    TEXT,
    identifier      TEXT,
    -- LOOPHOLES.md §8 · digest of the source as ingested, so a source that has
    -- since moved or changed can be detected rather than silently re-cited.
    source_hash     TEXT
);
-- The layer-3 access path: every leaf scan is a range over this index.
CREATE INDEX IF NOT EXISTS chunks_cluster_idx ON chunks (cluster_id);
CREATE INDEX IF NOT EXISTS chunks_identifier_idx ON chunks (identifier)
    WHERE identifier IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61'
);
";

/// A SQLite-backed corpus.
///
/// ! The connection sits behind a `Mutex` because SQLite connections are not
/// `Sync`. Concurrency is governed a layer up by the engine's semaphore
/// (`MCP_ENGINE.md` §4), so this lock is a correctness guard, ✗ the throughput
/// control — and it means the measured per-query latency is single-connection
/// latency, which is what the leaf-scan benchmark wants to report.
#[derive(Debug)]
pub struct SqliteStore {
    conn: Mutex<Connection>,
    space: EmbeddingSpace,
}

impl SqliteStore {
    /// Open an existing corpus read-only and read back the space it declares.
    ///
    /// # Errors
    /// Backend failure, or [`StoreError::NoRecordedSpace`] when the corpus
    /// never recorded what it was embedded with.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, StoreError> {
        let display = path.as_ref().display().to_string();
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

        // ! Refuse a build that never finished. Because the loader commits in
        // batches (`vera_index::stream`), an interrupted import leaves a corpus
        // with a valid schema, a working FTS index and *some* of the rows — it
        // opens, it answers, and it is silently short. That is the failure this
        // project is shaped around, and the marker is the only thing that
        // distinguishes it.
        //
        // ! Absent means **complete**, not incomplete. Corpora built before the
        // marker existed record nothing, and treating those as broken would
        // reject every existing index to catch a state none of them can be in:
        // the unmarked builds used a single transaction, so they are atomic by
        // construction.
        let state: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'build_state'",
                [],
                |r| r.get(0),
            )
            .ok();
        if let Some(state) = state
            && state != "complete"
        {
            return Err(StoreError::IncompleteBuild {
                path: display,
                state,
            });
        }
        Self::from_connection(conn)
    }

    /// Create (or reopen) a writable corpus and apply [`SCHEMA`].
    ///
    /// Used by ingest and by tests. The query path opens read-only.
    ///
    /// # Errors
    /// Backend failure.
    pub fn create(
        path: impl AsRef<std::path::Path>,
        space: &EmbeddingSpace,
    ) -> Result<Self, StoreError> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_space', ?1)",
            params![serde_json::to_string(space).map_err(|e| StoreError::Malformed(e.to_string()))?],
        )?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self, StoreError> {
        // Read-heavy, single writer: WAL and a generous page cache are free wins
        // and change no semantics.
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        conn.pragma_update(None, "synchronous", "NORMAL").ok();
        let space = read_space(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            space,
        })
    }

    /// Borrow the connection for ingest-side work (bulk insert, clustering).
    ///
    /// # Panics
    /// If another thread panicked while holding the lock.
    pub fn with_connection<T>(&self, f: impl FnOnce(&Connection) -> T) -> T {
        f(&self.conn.lock().expect("store lock poisoned"))
    }

    /// Every chunk id grouped by the identifier it carries, up to `limit`
    /// distinct identifiers.
    ///
    /// ! Inherent on the backend, ✗ on [`ChunkStore`], because it is **ground
    /// truth for the eval harness**, not a query-path operation. It reads the
    /// whole identifier column, which is precisely the corpus-wide scan the
    /// routing architecture exists to avoid — exposing it through the trait
    /// would put it one autocomplete away from the request path.
    ///
    /// It exists so exact-match recall (`EVAL.md` §3) can be measured without
    /// asking the engine's own lookup what the right answer is. What is under
    /// test lives above the store: identifier extraction, the routing bypass
    /// ordering, and whether fusion keeps the hit.
    ///
    /// # Errors
    /// Backend failure.
    pub fn identifier_index(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, Vec<String>)>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT identifier, id FROM chunks WHERE identifier IS NOT NULL \
             ORDER BY identifier, id",
        )?;
        let mut rows = stmt.query([])?;
        let mut out: Vec<(String, Vec<String>)> = Vec::new();
        while let Some(row) = rows.next()? {
            let identifier: String = row.get(0)?;
            let id: String = row.get(1)?;
            match out.last_mut() {
                // Ordered by identifier, so equal keys arrive together and the
                // grouping needs no map — and no map means no memory
                // proportional to the corpus when `limit` is small.
                Some((key, ids)) if *key == identifier => ids.push(id),
                _ => {
                    if out.len() >= limit {
                        break;
                    }
                    out.push((identifier, vec![id]));
                }
            }
        }
        Ok(out)
    }
}

fn read_space(conn: &Connection) -> Result<EmbeddingSpace, StoreError> {
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'embedding_space'",
            [],
            |r| r.get(0),
        )
        .ok();
    let raw = raw.ok_or(StoreError::NoRecordedSpace)?;
    serde_json::from_str(&raw).map_err(|e| StoreError::Malformed(e.to_string()))
}

/// Turn arbitrary user text into an FTS5 MATCH expression that cannot be a
/// syntax error.
///
/// ! Every token is quoted and joined with `OR`. Quoting is what stops a query
/// containing `-`, `"`, `*` or `NEAR` from being read as FTS5 syntax — with a
/// raw query string, `UU 28/2007` is a parse error, not a search. `OR` rather
/// than the default `AND` because this is the recall half of a hybrid engine:
/// dropping a document for missing one term is the fusion's job to weigh, not
/// the tokenizer's to decide.
///
/// Returns `None` when nothing searchable survives, so callers skip the query
/// instead of running a match against an empty expression.
#[must_use]
pub fn fts_match_expression(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t.to_lowercase()))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

/// Columns every `Chunk` read selects, in the order [`row_to_chunk`] expects.
const CHUNK_COLUMNS: &str = "id, domain_id, cluster_id, body, source_title, source_url, \
                             locator_page, locator_section, heading_path, identifier, \
                             source_hash";

fn row_to_chunk(row: &rusqlite::Row<'_>) -> rusqlite::Result<Chunk> {
    Ok(Chunk {
        id: row.get(0)?,
        domain_id: row.get(1)?,
        cluster_id: row.get(2)?,
        body: row.get(3)?,
        source_title: row.get(4)?,
        source_url: row.get(5)?,
        locator_page: row.get(6)?,
        locator_section: row.get(7)?,
        heading_path: row.get(8)?,
        identifier: row.get(9)?,
        source_hash: row.get(10)?,
    })
}

impl ChunkStore for SqliteStore {
    fn corpus_space(&self) -> Result<EmbeddingSpace, StoreError> {
        Ok(self.space.clone())
    }

    fn domains(&self) -> Result<Vec<DomainAnchor>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let mut stmt =
            conn.prepare("SELECT id, description, anchor, row_count FROM domains ORDER BY id")?;
        let mut out = Vec::new();
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(2)?;
            let mut anchor = Vec::new();
            decode_vector_into(&id, &blob, self.space.dim, &mut anchor)?;
            out.push(DomainAnchor {
                id,
                description: row.get(1)?,
                anchor,
                row_count: row.get(3)?,
            });
        }
        Ok(out)
    }

    fn centroids(&self, domain_id: &str) -> Result<Vec<Centroid>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        // ! Pinned to a single generation. `CLUSTER_MAINTENANCE.md` §3 requires a
        // query to see one cluster-version for its lifetime: a Tier-3 re-cluster
        // publishes a new generation alongside the live one, so selecting
        // without this filter would mix old and new centroids and probe exactly
        // the half-updated index the atomic swap exists to prevent. The engine
        // loads centroids once at startup, so "the newest complete generation"
        // is the version it pins until it reloads.
        let mut stmt = conn.prepare(
            "SELECT id, domain_id, centroid, row_count FROM clusters \
             WHERE domain_id = ?1 \
               AND generation = (SELECT MAX(generation) FROM clusters WHERE domain_id = ?1) \
             ORDER BY id",
        )?;
        let mut out = Vec::new();
        let mut rows = stmt.query(params![domain_id])?;
        while let Some(row) = rows.next()? {
            let id: i32 = row.get(0)?;
            let blob: Vec<u8> = row.get(2)?;
            let mut centroid = Vec::new();
            decode_vector_into(&id.to_string(), &blob, self.space.dim, &mut centroid)?;
            out.push(Centroid {
                id,
                domain_id: row.get(1)?,
                centroid,
                row_count: row.get(3)?,
            });
        }
        Ok(out)
    }

    fn scan_cluster(
        &self,
        cluster_id: i32,
        visit: &mut dyn FnMut(ScannedRow<'_>),
    ) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let mut stmt = conn.prepare_cached(
            "SELECT id, embedding FROM chunks WHERE cluster_id = ?1",
        )?;
        let mut rows = stmt.query(params![cluster_id])?;

        // ! One buffer for the whole cluster. This is the allocation the OOM
        // argument rests on: the working set is one row wide, ✗ one cluster.
        let mut vector = Vec::with_capacity(self.space.dim);
        let mut scanned = 0usize;
        while let Some(row) = rows.next()? {
            // ! Borrow the id, ✗ `row.get::<_, String>(0)`. That allocates a
            // String for every row scanned — on a 10K-row cluster that is 10K
            // allocations per probe, spent entirely on rows that will be
            // rejected. The visitor only sees `&str`, so nothing needs an owned
            // copy until TopK actually admits a candidate.
            let id = row.get_ref(0)?.as_str()?;
            let blob = row.get_ref(1)?.as_blob()?;
            decode_vector_into(id, blob, self.space.dim, &mut vector)?;
            visit(ScannedRow {
                id,
                vector: &vector,
            });
            scanned += 1;
        }
        Ok(scanned)
    }

    fn keyword_search(
        &self,
        query: &str,
        cluster_id: Option<i32>,
        limit: usize,
    ) -> Result<Vec<KeywordHit>, StoreError> {
        let Some(expr) = fts_match_expression(query) else {
            return Ok(Vec::new());
        };
        let conn = self.conn.lock().expect("store lock poisoned");

        // ! FTS5's bm25() returns a value that is *more negative* the better the
        // match. Negating is what makes "higher is better" true for every score
        // the fusion layer sees, so RRF never has to special-case a backend.
        let (sql, has_cluster) = match cluster_id {
            Some(_) => (
                "SELECT c.id, -bm25(chunks_fts) FROM chunks_fts \
                 JOIN chunks c ON c.rowid = chunks_fts.rowid \
                 WHERE chunks_fts MATCH ?1 AND c.cluster_id = ?3 \
                 ORDER BY bm25(chunks_fts) LIMIT ?2",
                true,
            ),
            None => (
                "SELECT c.id, -bm25(chunks_fts) FROM chunks_fts \
                 JOIN chunks c ON c.rowid = chunks_fts.rowid \
                 WHERE chunks_fts MATCH ?1 \
                 ORDER BY bm25(chunks_fts) LIMIT ?2",
                false,
            ),
        };
        let mut stmt = conn.prepare_cached(sql)?;
        let collect = |rows: &mut rusqlite::Rows<'_>| -> Result<Vec<KeywordHit>, StoreError> {
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(KeywordHit {
                    id: row.get(0)?,
                    score: row.get::<_, f64>(1)? as f32,
                });
            }
            Ok(out)
        };
        if has_cluster {
            let mut rows = stmt.query(params![expr, limit as i64, cluster_id.unwrap_or(0)])?;
            collect(&mut rows)
        } else {
            let mut rows = stmt.query(params![expr, limit as i64])?;
            collect(&mut rows)
        }
    }

    fn exact_identifier(
        &self,
        identifier: &str,
        limit: usize,
    ) -> Result<Vec<Chunk>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let sql = format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks WHERE identifier = ?1 LIMIT ?2"
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let mut rows = stmt.query(params![identifier, limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_chunk(row)?);
        }
        Ok(out)
    }

    fn chunks_by_id(&self, ids: &[String]) -> Result<Vec<Chunk>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().expect("store lock poisoned");
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql =
            format!("SELECT {CHUNK_COLUMNS} FROM chunks WHERE id IN ({placeholders})");
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(params_from_iter(ids.iter()))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_chunk(row)?);
        }
        Ok(out)
    }

    fn neighbors(
        &self,
        id: &str,
        edge: Edge,
        limit: usize,
    ) -> Result<Vec<Chunk>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        // ! The `IS NOT NULL` guard on same_identifier matters: without it every
        // chunk that ingest recorded no identifier for becomes a neighbour of
        // every other such chunk, which is a corpus-sized join presented as a
        // relationship.
        let sql = match edge {
            Edge::SameDocument => format!(
                "SELECT {CHUNK_COLUMNS} FROM chunks \
                 WHERE source_url = (SELECT source_url FROM chunks WHERE id = ?1) \
                   AND id != ?1 \
                 ORDER BY id LIMIT ?2"
            ),
            Edge::SameIdentifier => format!(
                "SELECT {CHUNK_COLUMNS} FROM chunks \
                 WHERE identifier IS NOT NULL \
                   AND identifier = (SELECT identifier FROM chunks WHERE id = ?1) \
                   AND id != ?1 \
                 ORDER BY id LIMIT ?2"
            ),
        };
        let mut stmt = conn.prepare_cached(&sql)?;
        let mut rows = stmt.query(params![id, limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_chunk(row)?);
        }
        Ok(out)
    }

    fn meta(&self, key: &str) -> Result<Option<String>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let mut stmt = conn.prepare_cached("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    }

    fn largest_cluster_rows(&self) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let n: i64 = conn.query_row(
            "SELECT COALESCE(MAX(n), 0) FROM (SELECT COUNT(*) AS n FROM chunks GROUP BY cluster_id)",
            [],
            |r| r.get(0),
        )?;
        Ok(usize::try_from(n).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode_vector;

    /// A 4-dim corpus: small enough to reason about by hand, real enough to
    /// exercise every query path.
    fn fixture() -> (tempfile::TempDir, SqliteStore) {
        let dir = tempfile::tempdir().unwrap();
        let space = EmbeddingSpace {
            model_id: "test/tiny".into(),
            dim: 4,
            normalized: false,
            query_instruction: String::new(),
            validated_providers: Vec::new(),
        };
        let store = SqliteStore::create(dir.path().join("c.db"), &space).unwrap();
        store.with_connection(|conn| {
            conn.execute(
                "INSERT INTO domains (id, description, anchor, row_count) VALUES (?1,?2,?3,?4)",
                params!["reg", "Indonesian regulations", encode_vector(&[1.0, 0.0, 0.0, 0.0]), 3],
            )
            .unwrap();
            for (id, centroid, n) in [
                (1, [1.0, 0.0, 0.0, 0.0], 2),
                (2, [0.0, 1.0, 0.0, 0.0], 1),
            ] {
                conn.execute(
                    "INSERT INTO clusters (id, domain_id, centroid, row_count) VALUES (?1,?2,?3,?4)",
                    params![id, "reg", encode_vector(&centroid), n],
                )
                .unwrap();
            }
            for (rowid, id, cluster, body, ident, vec) in [
                (1, "c1", 1, "wajib pajak dikenai sanksi administrasi", Some("UU 28/2007"), [1.0, 0.0, 0.0, 0.0]),
                (2, "c2", 1, "tarif pajak penghasilan orang pribadi", None, [0.9, 0.1, 0.0, 0.0]),
                (3, "c3", 2, "ketentuan umum perpajakan daerah", Some("PP 74/2011"), [0.0, 1.0, 0.0, 0.0]),
            ] {
                conn.execute(
                    "INSERT INTO chunks (rowid, id, domain_id, cluster_id, body, embedding, \
                     source_title, source_url, locator_page, locator_section, identifier) \
                     VALUES (?1,?2,'reg',?3,?4,?5,'Title','https://example/doc.pdf',14,'Pasal 9',?6)",
                    params![rowid, id, cluster, body, encode_vector(&vec), ident],
                )
                .unwrap();
            }
            // External-content FTS5 needs an explicit rebuild after bulk insert.
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')", [])
                .unwrap();
        });
        (dir, store)
    }

    #[test]
    fn a_corpus_reports_the_space_it_was_built_with() {
        let (_d, store) = fixture();
        let space = store.corpus_space().unwrap();
        assert_eq!(space.dim, 4);
        assert_eq!(space.model_id, "test/tiny");
    }

    #[test]
    fn a_half_written_corpus_refuses_to_open() {
        // ! The state batched commits make reachable: valid schema, working FTS,
        // some of the rows. Nothing about it looks wrong from the outside, which
        // is exactly why the marker has to be checked rather than the row count
        // eyeballed.
        let (dir, store) = fixture();
        store.with_connection(|conn| {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('build_state', 'in_progress')",
                [],
            )
            .unwrap();
        });
        drop(store);
        let err = SqliteStore::open(dir.path().join("c.db")).unwrap_err();
        match err {
            StoreError::IncompleteBuild { state, .. } => assert_eq!(state, "in_progress"),
            other => panic!("{other}"),
        }
    }

    #[test]
    fn a_corpus_predating_the_marker_still_opens() {
        // ! Absent means complete. Those builds used one transaction and are
        // atomic by construction, so rejecting them would break every existing
        // index to catch a state none of them can be in.
        let (dir, store) = fixture();
        drop(store);
        assert!(SqliteStore::open(dir.path().join("c.db")).is_ok());
    }

    #[test]
    fn a_corpus_with_no_recorded_space_refuses_to_open() {
        // ! Fails closed. Defaulting to 4096 here would let a 1024-dim corpus
        // load and produce confident nonsense.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bare.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(SCHEMA).unwrap();
        drop(conn);
        assert!(matches!(
            SqliteStore::open(&path).unwrap_err(),
            StoreError::NoRecordedSpace
        ));
    }

    #[test]
    fn layer_one_and_two_load_their_hot_structures() {
        let (_d, store) = fixture();
        let domains = store.domains().unwrap();
        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].anchor, vec![1.0, 0.0, 0.0, 0.0]);
        let centroids = store.centroids("reg").unwrap();
        assert_eq!(centroids.len(), 2);
        assert!(store.centroids("nonexistent").unwrap().is_empty());
    }

    #[test]
    fn scanning_a_cluster_visits_exactly_its_own_rows() {
        let (_d, store) = fixture();
        let mut seen = Vec::new();
        let n = store
            .scan_cluster(1, &mut |row| seen.push(row.id.to_owned()))
            .unwrap();
        assert_eq!(n, 2);
        seen.sort();
        assert_eq!(seen, ["c1", "c2"]);
    }

    #[test]
    fn a_scan_hands_the_visitor_a_borrowed_buffer_that_is_reused() {
        // ! The structural half of the OOM guarantee: a caller keeping every
        // vector has to copy it, and that copy is visible in the caller's code.
        let (_d, store) = fixture();
        let mut addresses = Vec::new();
        store
            .scan_cluster(1, &mut |row| {
                addresses.push(row.vector.as_ptr());
                assert_eq!(row.vector.len(), 4);
            })
            .unwrap();
        assert_eq!(addresses.len(), 2);
        assert_eq!(addresses[0], addresses[1], "buffer was not reused");
    }

    #[test]
    fn a_new_cluster_generation_completely_replaces_the_old_one() {
        // ! CLUSTER_MAINTENANCE.md §3. A Tier-3 re-cluster writes generation 2
        // alongside the live generation 1. Without the generation filter the
        // engine probes a mix of both — the half-updated index the atomic swap
        // exists to prevent — and the mix is silent: routing simply gets worse.
        let (_d, store) = fixture();
        store.with_connection(|conn| {
            conn.execute(
                "INSERT INTO clusters (id, domain_id, centroid, row_count, generation) \
                 VALUES (?1,?2,?3,?4,2)",
                params![99, "reg", encode_vector(&[0.0, 0.0, 1.0, 0.0]), 3],
            )
            .unwrap();
        });
        let centroids = store.centroids("reg").unwrap();
        assert_eq!(
            centroids.len(),
            1,
            "expected only generation 2, got {:?}",
            centroids.iter().map(|c| c.id).collect::<Vec<_>>()
        );
        assert_eq!(centroids[0].id, 99);
    }

    #[test]
    fn keyword_search_scoped_to_a_cluster_never_leaves_it() {
        let (_d, store) = fixture();
        let hits = store.keyword_search("pajak", Some(1), 10).unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.id.as_str()).collect();
        assert!(ids.contains(&"c1") && ids.contains(&"c2"), "{ids:?}");
        assert!(!ids.contains(&"c3"), "leaked a row from cluster 2");
    }

    #[test]
    fn keyword_search_without_a_cluster_spans_the_whole_corpus() {
        // ! This is the routing bypass. If it ever silently became
        // cluster-scoped, a known regulation could go missing with no error.
        let (_d, store) = fixture();
        let hits = store.keyword_search("perpajakan", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "c3", "global search missed cluster 2");
    }

    #[test]
    fn bm25_scores_are_returned_higher_is_better() {
        let (_d, store) = fixture();
        let hits = store.keyword_search("pajak", None, 10).unwrap();
        assert!(hits.len() >= 2);
        assert!(
            hits[0].score >= hits[1].score,
            "scores must descend: {hits:?}"
        );
        assert!(hits[0].score > 0.0, "sign was not flipped: {hits:?}");
    }

    #[test]
    fn a_query_of_pure_punctuation_returns_nothing_rather_than_erroring() {
        // ! FTS5 would raise a syntax error on the raw string.
        let (_d, store) = fixture();
        assert!(store.keyword_search("--- \"\" ***", None, 10).unwrap().is_empty());
    }

    #[test]
    fn a_regulation_number_is_searchable_despite_its_punctuation() {
        // "UU 28/2007" is a MATCH syntax error unquoted · the exact case
        // LOOPHOLES.md §1 says must never silently fail.
        let (_d, store) = fixture();
        let hits = store.keyword_search("sanksi UU 28/2007", None, 10).unwrap();
        assert!(hits.iter().any(|h| h.id == "c1"), "{hits:?}");
    }

    #[test]
    fn exact_identifier_lookup_is_corpus_wide() {
        let (_d, store) = fixture();
        let hits = store.exact_identifier("PP 74/2011", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "c3");
        assert!(store.exact_identifier("UU 99/9999", 10).unwrap().is_empty());
    }

    #[test]
    fn a_source_hash_round_trips_and_its_absence_reads_as_unknown() {
        // ! LOOPHOLES.md §8. A link is a promise the source can break silently:
        // revised, renumbered, moved. The hash is what makes the citation
        // falsifiable. `None` must stay distinguishable from "verified intact",
        // or a corpus ingested without hashes would look fully verified.
        let (_d, store) = fixture();
        store.with_connection(|conn| {
            conn.execute(
                "UPDATE chunks SET source_hash = 'sha256:abc' WHERE id = 'c1'",
                [],
            )
            .unwrap();
        });
        let chunks = store.chunks_by_id(&["c1".to_owned(), "c2".to_owned()]).unwrap();
        let c1 = chunks.iter().find(|c| c.id == "c1").unwrap();
        let c2 = chunks.iter().find(|c| c.id == "c2").unwrap();
        assert_eq!(c1.source_hash.as_deref(), Some("sha256:abc"));
        assert_eq!(c2.source_hash, None, "unrecorded is unknown, not intact");
    }

    #[test]
    fn provenance_comes_back_per_id_from_stored_fields() {
        let (_d, store) = fixture();
        let prov = store.provenance(&["c1".to_owned(), "c3".to_owned()]).unwrap();
        assert_eq!(prov.len(), 2);
        let (_, source) = prov.iter().find(|(id, _)| id == "c1").unwrap();
        assert_eq!(source.url, "https://example/doc.pdf");
        assert_eq!(source.locator.page, Some(14));
        assert_eq!(source.locator.section.as_deref(), Some("Pasal 9"));
    }

    #[test]
    fn asking_for_no_ids_costs_no_query() {
        let (_d, store) = fixture();
        assert!(store.chunks_by_id(&[]).unwrap().is_empty());
    }

    #[test]
    fn traversing_same_document_returns_the_other_chunks_of_that_document() {
        let (_d, store) = fixture();
        let n = store.neighbors("c1", Edge::SameDocument, 10).unwrap();
        // c1, c2, c3 all share the fixture's single source_url.
        let ids: Vec<&str> = n.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["c2", "c3"], "must exclude the chunk itself");
    }

    #[test]
    fn traversing_same_identifier_matches_only_the_same_regulation() {
        let (_d, store) = fixture();
        // c1 is "UU 28/2007", c3 is "PP 74/2011", c2 has none.
        assert!(store.neighbors("c1", Edge::SameIdentifier, 10).unwrap().is_empty());
    }

    #[test]
    fn chunks_without_an_identifier_are_not_neighbours_of_each_other() {
        // ! Without the NOT NULL guard this is a corpus-sized join dressed up
        // as a relationship: every unlabelled chunk related to every other.
        let (_d, store) = fixture();
        assert!(store.neighbors("c2", Edge::SameIdentifier, 10).unwrap().is_empty());
    }

    #[test]
    fn traversal_respects_its_limit() {
        let (_d, store) = fixture();
        assert_eq!(store.neighbors("c1", Edge::SameDocument, 1).unwrap().len(), 1);
    }

    #[test]
    fn traversing_from_an_unknown_id_yields_nothing_rather_than_erroring() {
        let (_d, store) = fixture();
        assert!(store.neighbors("nope", Edge::SameDocument, 10).unwrap().is_empty());
    }

    #[test]
    fn edge_names_round_trip_and_the_vocabulary_is_closed() {
        for e in Edge::all() {
            assert_eq!(Edge::parse(e.name()), Some(*e));
        }
        // ! Edges the design anticipates but the schema cannot answer must not
        // parse · describe publishes only what traverse can actually do.
        for absent in ["parent", "children", "cites", "cited_by", "versions"] {
            assert_eq!(Edge::parse(absent), None, "{absent} must not be accepted yet");
        }
    }

    #[test]
    fn the_largest_cluster_is_reported_for_the_ram_budget() {
        let (_d, store) = fixture();
        assert_eq!(store.largest_cluster_rows().unwrap(), 2);
    }

    #[test]
    fn match_expressions_quote_every_token() {
        assert_eq!(fts_match_expression("UU 28/2007"), Some("\"uu\" OR \"28\" OR \"2007\"".into()));
        assert_eq!(fts_match_expression("  "), None);
        assert_eq!(fts_match_expression("NEAR OR AND*"), Some("\"near\" OR \"or\" OR \"and\"".into()));
    }
}
