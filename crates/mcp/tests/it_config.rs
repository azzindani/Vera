//! The config files this repository ships must load.
//!
//! ! A config file in the repository that does not parse is worse than no file
//! at all: it is copied into a deployment, the process refuses to start, and
//! the operator's first assumption is that their environment is wrong. These
//! tests are cheap and they run in CI with no database.
//!
//! ! `config/vocabulary.id_regulation.json` is also asserted to reproduce the
//! compiled-in fallback **exactly**. That equality is the entire migration: it
//! says the file is the same data the binary carries, so moving the tables out
//! of `factors.rs` and into the corpus changes nothing that ranks. When Ravel
//! writes this into `corpus_meta`, this test is what says the two agree.

use std::path::PathBuf;

fn repo_file(rel: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{} · {e}", p.display()))
}

#[test]
fn the_shipped_algorithm_registry_loads() {
    let r = contract::Registry::parse(&repo_file("config/algorithms.json"))
        .expect("config/algorithms.json must load");
    assert!(
        r.get(contract::DEFAULT_ALGORITHM).is_some(),
        "the default must be defined"
    );
}

#[test]
fn every_shipped_algorithm_stays_inside_the_prior_bound() {
    // Redundant with `parse` validating, and deliberately so: this is the test
    // whose NAME appears when someone raises a weight because Recall@5 went up.
    let r = contract::Registry::parse(&repo_file("config/algorithms.json")).expect("loads");
    for (name, a) in &r.algorithms {
        assert!(
            a.prior_bound() <= contract::MAX_PRIOR_BOUND + f32::EPSILON,
            "`{name}` bound {:.2}x",
            a.prior_bound()
        );
    }
}

#[test]
fn only_measured_algorithms_are_offered_to_an_agent() {
    // ! An unfitted algorithm is a raw weight vector with a friendly name.
    // `balanced_expanded` ships without a `fitted` block on purpose: sibling
    // expansion was measured at +2.3 points, which is ONE case out of 44 and
    // is not separable from noise on this eval set (docs/SCORING.md §7).
    let r = contract::Registry::parse(&repo_file("config/algorithms.json")).expect("loads");
    let offered = r.offered();
    assert!(offered.contains(&"balanced"), "{offered:?}");
    assert!(offered.contains(&"literal"), "{offered:?}");
    assert!(
        !offered.contains(&"balanced_expanded"),
        "unfitted must not be offered: {offered:?}"
    );
    assert!(r.get("balanced_expanded").is_some(), "still reachable");
}

#[test]
fn a_fitted_block_names_the_harness_that_produced_it() {
    // Invariant 15. A number with no command behind it is a claim, ✗ a
    // measurement, and this is where that rule is mechanical rather than
    // cultural.
    let r = contract::Registry::parse(&repo_file("config/algorithms.json")).expect("loads");
    for (name, a) in &r.algorithms {
        if let Some(f) = &a.fitted {
            assert!(!f.harness.trim().is_empty(), "`{name}` claims without a harness");
            assert!(f.n > 0, "`{name}` claims a figure from zero cases");
        }
    }
}

#[test]
fn the_shipped_vocabulary_reproduces_the_compiled_in_fallback_exactly() {
    // ! The migration, asserted. Moving the Indonesian tables out of
    // `engine::factors` and into the corpus must be a no-op on ranking, and the
    // only way to know that is to compare the two artefacts rather than to read
    // them side by side.
    let from_file = load_vocab("config/vocabulary.id_regulation.json");
    let compiled = engine::Vocabulary::id_regulation();

    // ! `abs() < EPSILON`, not `==`. These are f32 parsed from decimal text on
    // one side and written as a literal on the other; exact equality happens to
    // hold for these values and is not the property being asserted.
    assert!((from_file.max_tier - compiled.max_tier).abs() < f32::EPSILON);
    assert!(
        (from_file.structural_default - compiled.structural_default).abs() < f32::EPSILON
    );
    assert_eq!(from_file.min_term_chars, compiled.min_term_chars);
    assert_eq!(from_file.structural.len(), compiled.structural.len());
    for (a, b) in from_file.structural.iter().zip(&compiled.structural) {
        // ! Compared by the pattern's SOURCE, not by the compiled automaton: two
        // identical patterns compile to distinct values, and the property under test
        // is "built from the same declaration".
        assert_eq!(a.pattern.as_str(), b.pattern.as_str());
        assert_eq!(a.field, b.field);
        assert!(
            (a.score - b.score).abs() < f32::EPSILON,
            "{}",
            a.pattern.as_str()
        );
    }
    assert_eq!(from_file.stopwords, compiled.stopwords);

    // Authority is order-insensitive: a map in the file, a Vec in the engine.
    let mut a = from_file.authority.clone();
    let mut b = compiled.authority.clone();
    a.sort();
    b.sort();
    assert_eq!(a, b);

    assert!(
        from_file.is_builtin(),
        "the file must be recognised as equal to the fallback"
    );
}

#[test]
fn the_shipped_vocabulary_can_evaluate_the_shipped_algorithms() {
    // ! The startup guard, run in CI. An algorithm leaning on `structural`
    // against a vocabulary with no structural ladder is two config files that
    // disagree, and the request that reveals it is whichever one names that
    // algorithm -- possibly weeks later.
    let v = load_vocab("config/vocabulary.id_regulation.json");
    let r = contract::Registry::parse(&repo_file("config/algorithms.json")).expect("loads");
    for (name, a) in &r.algorithms {
        let w = engine::Weights {
            relevance_floor: a.factors.relevance_floor,
            authority: a.factors.authority,
            structural: a.factors.structural,
            temporal: a.factors.temporal,
            completeness: a.factors.completeness,
            topical: a.factors.topical,
        };
        assert!(
            v.missing_for(&w).is_empty(),
            "`{name}` weights factors this vocabulary cannot evaluate: {:?}",
            v.missing_for(&w)
        );
    }
}

/// The loader lives in the binary crate, so an integration test cannot import
/// it. Parsing here through `serde_json` would test a different code path, so
/// the file is read and the mapping asserted against what the binary's own unit
/// tests already cover.
fn load_vocab(rel: &str) -> engine::Vocabulary {
    let body = repo_file(rel);
    let v: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
    let obj = v.as_object().expect("object");

    let authority = obj
        .get("authority")
        .and_then(serde_json::Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    v.as_u64()
                        .and_then(|n| u8::try_from(n).ok())
                        .map(|n| (k.trim().to_uppercase(), n))
                })
                .collect()
        })
        .unwrap_or_default();

    let structural = obj
        .get("structural")
        .and_then(serde_json::Value::as_array)
        .map(|rules| {
            rules
                .iter()
                .map(|r| {
                    #[allow(clippy::cast_possible_truncation)]
                    engine::StructuralRule::new(
                        &regex::escape(r["label"].as_str().unwrap_or_default().trim()),
                        match r["field"].as_str().unwrap_or("either") {
                            "article" => engine::Field::Article,
                            "chapter" => engine::Field::Chapter,
                            _ => engine::Field::Either,
                        },
                        r["score"].as_f64().unwrap_or(0.0) as f32,
                        r["prefix"].as_bool().unwrap_or(false),
                    )
                    .expect("a shipped label must compile")
                })
                .collect()
        })
        .unwrap_or_default();

    #[allow(clippy::cast_possible_truncation)]
    engine::Vocabulary {
        authority,
        max_tier: obj.get("max_tier").and_then(serde_json::Value::as_f64).unwrap_or(10.0) as f32,
        structural,
        structural_default: obj
            .get("structural_default")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.5) as f32,
        stopwords: obj
            .get("stopwords")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(|s| s.trim().to_lowercase()))
                    .collect()
            })
            .unwrap_or_default(),
        min_term_chars: obj
            .get("min_term_chars")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(3),
    }
}
