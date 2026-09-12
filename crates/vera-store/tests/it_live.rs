//! Integration tests against a live, ingested corpus.
//!
//! ! All `#[ignore]`d. They need Postgres with a loaded corpus, which CI does
//! not have — `pipeline.yaml`'s `full` stage will provide one. Ignoring keeps
//! `cargo test` hermetic on every machine while these stay runnable on demand:
//!
//!     docker compose up -d
//!     cargo test -p vera-store -- --ignored --nocapture
//!
//! ! `#[ignore]` here means "needs a database", ✗ "known flaky". A test that
//! fails intermittently gets fixed or deleted the same day.

use vera_store::{SearchOps, connect, sparse_literal};

const DEFAULT_PG: &str = "host=localhost port=5432 dbname=vera user=vera password=vera";

fn ops() -> SearchOps {
    let url = std::env::var("VERA_PG").unwrap_or_else(|_| DEFAULT_PG.to_owned());
    SearchOps::new(connect(&url, 4).expect("pool"))
}

#[tokio::test]
#[ignore = "needs a live corpus"]
async fn corpus_meta_declares_the_recipe_that_produced_the_vectors() {
    let meta = ops().corpus_meta().await.expect("corpus_meta");
    println!(
        "corpus {} · {} @ {}d · {} pooling · sparse {} @ {}d",
        meta.id,
        meta.dense_model,
        meta.dense_dim,
        meta.dense_pooling,
        meta.sparse_scheme,
        meta.sparse_dim
    );

    assert!(!meta.dense_model.is_empty(), "a corpus must name its model");
    assert!(meta.dense_dim > 0);
    assert!(
        !meta.sparse_vocab_sha256.is_empty(),
        "a sparse vocabulary that is not recorded cannot score a query"
    );

    // The engine this corpus was built for.
    meta.ensure_compatible("qwen/qwen3-embedding-0.6b", 1024)
        .expect("engine and corpus must agree");

    // ! And the pin the project will eventually move to must be REFUSED until
    // the corpus is re-embedded for it.
    assert!(
        meta.ensure_compatible("qwen/qwen3-embedding-8b", 4096)
            .is_err()
    );
}

#[tokio::test]
#[ignore = "needs a live corpus"]
async fn centroids_load_at_the_declared_width() {
    let store = ops();
    let meta = store.corpus_meta().await.expect("corpus_meta");
    let centroids = store.centroids().await.expect("centroids");

    println!("{} centroids", centroids.len());
    assert!(centroids.len() > 1, "routing needs more than one cluster");
    for (id, v) in &centroids {
        assert_eq!(
            v.len(),
            usize::try_from(meta.dense_dim).unwrap(),
            "cluster {id} centroid is the wrong width"
        );
    }
}

#[tokio::test]
#[ignore = "needs a live corpus"]
async fn a_cluster_scan_returns_ranked_hits_from_that_cluster_only() {
    let store = ops();
    let meta = store.corpus_meta().await.expect("corpus_meta");
    let centroids = store.centroids().await.expect("centroids");
    let (cluster_id, centroid) = &centroids[0];

    // Querying with a cluster's own centroid must surface its members.
    let hits = store
        .dense_in_cluster(*cluster_id, centroid, 5)
        .await
        .expect("dense scan");
    assert!(!hits.is_empty(), "cluster {cluster_id} returned nothing");

    // Scores are descending similarity.
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score, "not ranked: {hits:?}");
    }

    // ! Every hit really is in the probed cluster · this is the pruning
    // guarantee, and a leak here would mean the OOM bound is fiction.
    let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    for row in store.chunks_by_id(&ids).await.expect("fetch") {
        assert_eq!(row.cluster_id, Some(*cluster_id));
    }
    println!(
        "cluster {cluster_id} ({}d) -> {} hits",
        meta.dense_dim,
        hits.len()
    );
}

#[tokio::test]
#[ignore = "needs a live corpus"]
async fn the_text_arm_finds_indonesian_legal_phrasing() {
    let hits = ops()
        .text("sanksi administrasi berupa denda", 10)
        .await
        .expect("text search");
    assert!(!hits.is_empty(), "stemmed lexical search returned nothing");
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[tokio::test]
#[ignore = "needs a live corpus"]
async fn the_sparse_arm_accepts_a_pgvector_literal() {
    let store = ops();
    let meta = store.corpus_meta().await.expect("corpus_meta");
    let dim = u32::try_from(meta.sparse_dim).unwrap();

    // An empty query vector is legal and simply matches nothing meaningful;
    // it must not error, because a query of pure stopwords produces one.
    let empty = store.sparse(&sparse_literal(&[], dim), 5).await;
    assert!(
        empty.is_ok(),
        "empty sparse query must not error: {empty:?}"
    );

    let some = store
        .sparse(&sparse_literal(&[(10, 1.0), (500, 1.0)], dim), 5)
        .await
        .expect("sparse search");
    println!("sparse returned {} hits", some.len());
}

#[tokio::test]
#[ignore = "needs a live corpus"]
async fn exact_identifier_search_bypasses_routing_entirely() {
    // ! Invariant 4, the highest-value guard in the suite. Routing measured
    // 95.0% recall, so ~5% of the time the right document is in a cluster that
    // was never probed. This path must find it anyway.
    let store = ops();
    let hits = store
        .exact_identifier("26", Some(2009), 10)
        .await
        .expect("exact");
    assert!(
        !hits.is_empty(),
        "PP 26/2009 is in the corpus and must be findable"
    );

    for h in &hits {
        assert!(h.matched_on.contains("26"), "{h:?}");
    }

    // Never gated by cluster selection: the hits come from wherever they live.
    let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    let rows = store.chunks_by_id(&ids).await.expect("fetch");
    assert_eq!(rows.len(), ids.len());
    println!(
        "exact match -> {} chunks, e.g. {}",
        rows.len(),
        hits[0].matched_on
    );
}

#[tokio::test]
#[ignore = "needs a live corpus"]
async fn a_nonexistent_identifier_returns_empty_rather_than_erroring() {
    let hits = ops()
        .exact_identifier("99999", Some(1800), 10)
        .await
        .expect("must succeed with no results");
    assert!(hits.is_empty());
}

#[tokio::test]
#[ignore = "needs a live corpus"]
async fn this_corpus_reports_incomplete_provenance_rather_than_faking_it() {
    // ! The spike corpus genuinely has no source_url. The honest behaviour is
    // to say so (invariant 8); a synthesised link would look verifiable and
    // lead a human nowhere.
    let store = ops();
    let hits = store.text("peraturan", 3).await.expect("text");
    let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    for row in store.chunks_by_id(&ids).await.expect("fetch") {
        assert!(!row.source_title.is_empty(), "a citation needs a title");
        if !row.provenance_complete() {
            assert!(
                row.source_url.is_none(),
                "incomplete means absent, not blank"
            );
        }
    }
}
