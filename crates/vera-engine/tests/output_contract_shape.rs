//! The wire shape of `search_knowledge`, asserted against `OUTPUT_CONTRACT.md` §2.
//!
//! ! Field names and shapes here are the **contract an agent parses**. A Rust
//! refactor that renames a field or changes an array's element type is a
//! breaking API change, and nothing else in the test suite would notice —
//! serde will happily serialize whatever the struct says.

use serde_json::Value;
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
    }
}

fn response(dir: &std::path::Path) -> Value {
    let mut vectors = Matrix::new(4);
    let mut rows = Vec::new();
    for i in 0..24 {
        vectors.push(&[1.0, 0.0, 0.0, 0.0]);
        rows.push(IngestRow {
            id: format!("reg::uu-28-2007::pasal-9::c{i}"),
            body: format!(
                "Wajib Pajak yang terlambat menyampaikan laporan dikenai sanksi \
                 administrasi bagian {i}"
            ),
            source_title: "UU No. 28 Tahun 2007 — Ketentuan Umum Perpajakan".into(),
            source_url: "https://peraturan.example/uu-28-2007.pdf".into(),
            locator_page: Some(14),
            locator_section: Some("Pasal 9 ayat (3)".into()),
            heading_path: None,
            identifier: (i == 3).then(|| "UU 28/2007".to_owned()),
        });
    }
    let path = dir.join("c.db");
    build_corpus(
        &path,
        &space(),
        "regulations",
        "Indonesian regulations",
        &rows,
        &vectors,
        &KMeansConfig {
            k: 3,
            ..Default::default()
        },
    )
    .expect("build");

    let engine = Engine::load(
        SqliteStore::open(&path).expect("open"),
        Config {
            embedding: space(),
            ..Config::default()
        },
    )
    .expect("load");

    let out = engine
        .search(
            "sanksi keterlambatan pelaporan pajak UU 28/2007",
            &[1.0, 0.0, 0.0, 0.0],
            Probe::Nearest(5),
        )
        .expect("search");
    assert!(
        !out.response.results.is_empty(),
        "fixture produced no results · the shape assertions below would be vacuous"
    );
    serde_json::to_value(&out.response).expect("serialize")
}

#[test]
fn the_top_level_fields_match_the_documented_contract() {
    let dir = tempfile::tempdir().unwrap();
    let v = response(dir.path());
    for field in [
        "success",
        "op",
        "query",
        "detected_domain",
        "domain_confidence",
        "clusters_probed",
        "results",
        "citation_block",
        "summary_payload",
        "exact_matches",
        "confidence",
        "progress",
        "token_estimate",
        "truncated",
    ] {
        assert!(v.get(field).is_some(), "missing documented field `{field}`");
    }
    assert_eq!(v["op"], "search_knowledge");
    assert_eq!(v["detected_domain"], "regulations");
}

#[test]
fn a_result_matches_the_documented_shape() {
    let dir = tempfile::tempdir().unwrap();
    let v = response(dir.path());
    let r = &v["results"][0];
    for field in ["id", "snippet", "score", "scores", "source"] {
        assert!(r.get(field).is_some(), "result missing `{field}`");
    }
    assert!(r["scores"]["dense"].is_number());
    assert!(r["scores"]["bm25"].is_number());
    assert!(r["source"]["title"].is_string());
    assert!(r["source"]["url"].is_string());
    assert_eq!(r["source"]["locator"]["page"], 14);
    assert_eq!(r["source"]["locator"]["section"], "Pasal 9 ayat (3)");
}

#[test]
fn citation_block_is_an_array_of_ready_to_render_strings() {
    // OUTPUT_CONTRACT.md §2:
    //   "citation_block": [
    //     "[1] UU No. 28 Tahun 2007, Pasal 9 ayat (3), p.14 — https://…"
    //   ]
    // ! "ready-to-render, ordered list the agent can drop into its answer" —
    // an array of objects forces every caller to reassemble the line, and two
    // callers will format it differently, which defeats the point.
    let dir = tempfile::tempdir().unwrap();
    let v = response(dir.path());
    let block = v["citation_block"].as_array().expect("array");
    assert!(!block.is_empty());
    assert!(
        block[0].is_string(),
        "citation_block entries must be strings, got {}",
        block[0]
    );
    let first = block[0].as_str().unwrap();
    assert!(first.starts_with("[1] "), "{first}");
    assert!(first.contains("Pasal 9 ayat (3), p.14"), "{first}");
    assert!(first.contains("https://peraturan.example/"), "{first}");
}

#[test]
fn summary_payload_sources_are_citation_references() {
    // OUTPUT_CONTRACT.md §2: "sources": ["[1]", "[2]"]
    // ! They index into citation_block. Repeating the full title+url here
    // duplicates the citation block and leaves the agent to correlate two
    // differently-formatted lists.
    let dir = tempfile::tempdir().unwrap();
    let v = response(dir.path());
    let sources = v["summary_payload"]["sources"].as_array().expect("array");
    assert!(!sources.is_empty());
    let first = sources[0].as_str().expect("string");
    assert!(
        first.starts_with('[') && first.ends_with(']'),
        "expected a citation reference like \"[1]\", got {first:?}"
    );
}

#[test]
fn summary_payload_coverage_describes_the_search() {
    // OUTPUT_CONTRACT.md §2:
    //   "coverage": "5 clusters probed, 3 distinct documents, top score 0.87"
    // ! It tells the agent how broadly the corpus was consulted, which is what
    // it needs to decide whether to widen. A bare result count says nothing.
    let dir = tempfile::tempdir().unwrap();
    let v = response(dir.path());
    let coverage = v["summary_payload"]["coverage"].as_str().expect("string");
    assert!(coverage.contains("cluster"), "coverage was {coverage:?}");
    assert!(coverage.contains("document"), "coverage was {coverage:?}");
    assert!(coverage.contains("score"), "coverage was {coverage:?}");
}

#[test]
fn confidence_reflects_result_quality_not_only_routing_certainty() {
    // OUTPUT_CONTRACT.md §4: high = "strong top scores, exact-identifier match
    // present, or tight agreement between dense and BM25".
    // ! All three are computable without a model, so deriving confidence from
    // the domain-anchor similarity alone reports something different from what
    // the contract promises — and domain similarity is nearly constant across
    // queries, so it carries almost no signal.
    let dir = tempfile::tempdir().unwrap();
    let v = response(dir.path());
    assert!(
        !v["exact_matches"].as_array().unwrap().is_empty(),
        "fixture should have produced an exact-identifier hit"
    );
    assert_eq!(
        v["confidence"], "high",
        "an exact-identifier match present must read as high confidence"
    );
}
