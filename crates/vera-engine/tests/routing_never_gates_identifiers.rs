//! `CLAUDE.md` §7 rule 4 · an exact identifier is never gated by routing.
//!
//! ! This is the invariant with the worst failure mode in the whole engine.
//! When routing loses a named regulation the response is `success: true` with
//! an empty result set — indistinguishable, to the caller, from "no such
//! regulation exists". A human acting on that concludes the law does not say
//! what they were told it says.
//!
//! The bug this file pins actually shipped: the identifier lookup was written
//! after the layer-1 early return, so a query naming `UU 28/2007` returned
//! nothing whenever its vector fell below the domain-anchor threshold.

use vera_core::{Config, EmbeddingSpace};
use vera_engine::{Engine, Probe};
use vera_index::{IngestRow, KMeansConfig, Matrix, build_corpus};
use vera_store::sqlite::SqliteStore;

fn space() -> EmbeddingSpace {
    EmbeddingSpace {
        model_id: "test/tiny".into(),
        dim: 4,
        normalized: true,
        query_instruction: String::new(),
        validated_providers: Vec::new(),
    }
}

/// A corpus whose rows all sit near `+x`, with one carrying a citable id.
fn corpus(dir: &std::path::Path) -> SqliteStore {
    let mut vectors = Matrix::new(4);
    let mut rows = Vec::new();
    for i in 0..40 {
        let v = if i == 7 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [0.99, 0.14, 0.0, 0.0]
        };
        vectors.push(&v);
        rows.push(IngestRow {
            id: format!("c{i}"),
            body: format!("ketentuan umum perpajakan bagian {i}"),
            source_title: "Undang-Undang Ketentuan Umum Perpajakan".into(),
            source_url: "https://peraturan.example/uu-28-2007.pdf".into(),
            locator_page: Some(14),
            locator_section: Some("Pasal 9".into()),
            heading_path: None,
            identifier: (i == 7).then(|| "UU 28/2007".to_owned()),
            source_hash: None,
        });
    }
    let path = dir.join("corpus.db");
    build_corpus(
        &path,
        &space(),
        "regulations",
        "test",
        &rows,
        &vectors,
        &KMeansConfig {
            k: 4,
            ..Default::default()
        },
    )
    .expect("build");
    SqliteStore::open(&path).expect("open")
}

/// An engine whose layer-1 threshold is forced high enough to reject anything.
fn engine_rejecting_every_domain(store: SqliteStore) -> Engine<SqliteStore> {
    let mut config = Config {
        embedding: space(),
        ..Config::default()
    };
    // ! Explicit override, ✗ a calibrated value: the point is to simulate
    // routing failing, which is exactly when the bypass has to hold.
    config.routing.domain_threshold = Some(0.999_9);
    Engine::load(store, config).expect("load")
}

#[test]
fn a_named_regulation_is_found_even_when_layer_1_rejects_the_query() {
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_rejecting_every_domain(corpus(dir.path()));

    // Points away from the corpus · layer 1 must reject it.
    let orthogonal = [0.0, 0.0, 1.0, 0.0];
    let out = engine
        .search("apa isi UU 28/2007", &orthogonal, Probe::Nearest(5))
        .expect("search");

    assert!(out.response.success);
    assert!(
        out.response.detected_domain.is_none(),
        "routing was supposed to fail for this test to mean anything"
    );
    assert!(
        !out.response.exact_matches.is_empty(),
        "the named regulation was silently lost · this is the rule-4 failure"
    );
    assert_eq!(out.response.exact_matches[0].id, "c7");
    assert!(
        !out.response.results.is_empty(),
        "exact hits must reach `results`, not only `exact_matches` · an agent \
         reading results would otherwise see nothing"
    );
    assert_eq!(out.response.results[0].id, "c7");
    // Provenance still comes from ingest, never invented.
    assert_eq!(
        out.response.results[0].source.url,
        "https://peraturan.example/uu-28-2007.pdf"
    );
    assert!(!out.response.citation_block.is_empty());
}

#[test]
fn an_off_topic_query_with_no_identifier_still_returns_nothing() {
    // ! The bypass must not become a backdoor that resurrects the "guess a
    // domain" behaviour for ordinary queries.
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_rejecting_every_domain(corpus(dir.path()));

    let out = engine
        .search("resep rendang padang", &[0.0, 0.0, 1.0, 0.0], Probe::Nearest(5))
        .expect("search");
    assert!(out.response.success, "finding nothing is not an error");
    assert!(out.response.detected_domain.is_none());
    assert!(out.response.results.is_empty());
    assert!(out.response.exact_matches.is_empty());
    assert!(out.response.hint.is_some(), "must say what to do next");
}

#[test]
fn a_regulation_named_in_a_query_that_does_route_is_also_returned() {
    // The ordinary path: routing succeeds *and* an identifier is present.
    let dir = tempfile::tempdir().unwrap();
    let store = corpus(dir.path());
    let config = Config {
        embedding: space(),
        ..Config::default()
    };
    let engine = Engine::load(store, config).expect("load");

    let out = engine
        .search("sanksi UU 28/2007", &[1.0, 0.0, 0.0, 0.0], Probe::Nearest(5))
        .expect("search");
    assert!(out.response.detected_domain.is_some(), "should have routed");
    assert!(
        out.response.exact_matches.iter().any(|m| m.id == "c7"),
        "{:?}",
        out.response.exact_matches
    );
    assert!(out.response.results.iter().any(|r| r.id == "c7"));
}

#[test]
fn the_identifier_lookup_runs_before_routing_so_it_costs_nothing_to_fail() {
    // ! Ordering guard. If the lookup ever drifts back below the layer-1 early
    // return, `exact_path` will be zero on a rejected query because it never
    // ran — and the first test would fail too, but this one names the cause.
    let dir = tempfile::tempdir().unwrap();
    let engine = engine_rejecting_every_domain(corpus(dir.path()));
    let out = engine
        .search("UU 28/2007", &[0.0, 0.0, 1.0, 0.0], Probe::Nearest(5))
        .expect("search");
    assert!(
        out.timings.exact_path > std::time::Duration::ZERO,
        "the exact-identifier path did not execute on a routing-rejected query"
    );
    assert_eq!(out.timings.clusters_probed, 0, "no clusters should be opened");
}
