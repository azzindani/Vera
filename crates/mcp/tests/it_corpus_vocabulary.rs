//! Vera parses what Ravel actually writes.
//!
//! ! The fixtures in `tests/fixtures/vocab.*.json` are **generated**, not written:
//!
//! ```text
//! cd D:/Github/Ravel && python -c "
//! import sys, json; sys.path.insert(0,'src')
//! from spec import Registry
//! print(json.dumps(Registry.load().get('id_regulation').spec.scoring_vocabulary))"
//! ```
//!
//! A hand-written fixture would test that Vera can parse a shape somebody imagined
//! Ravel produces. The two repositories cannot import each other and there is no
//! shared schema artefact, so a generated fixture checked in beside the parser is
//! the closest thing to a contract test that exists — and when it drifts, it drifts
//! visibly in a diff rather than silently in production.
//!
//! ! `generic` is here for the reason that profile exists: to prove the engine is
//! not Indonesian-specific. It declares an English vocabulary and one genuine
//! pattern — `\d+(\.\d+)*\.?\s` for numbered sections — which is why `engine`
//! carries a regex dependency at all.

use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{} · {e}", p.display()))
}

/// The loader lives in the binary crate, so an integration test cannot call it.
/// This mirrors `vocabulary::CorpusVocabulary::load`; the binary's own unit tests
/// cover the mapping, and what this file asserts is that the real documents parse
/// and score.
fn load(name: &str) -> engine::Vocabulary {
    let v: serde_json::Value = serde_json::from_str(&fixture(name)).expect("valid JSON");
    let o = v.as_object().expect("object");
    let f = |k: &str, d: f64| o.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
    let list = |k: &str| -> Vec<String> {
        o.get(k)
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };

    let mut structural = Vec::new();
    #[allow(clippy::cast_possible_truncation)]
    for (key, score) in [
        ("annex", f("annex_score", 0.2) as f32),
        ("operative", f("operative_score", 1.0) as f32),
    ] {
        for pattern in list(key) {
            structural.push(
                engine::StructuralRule::new(&pattern, engine::Field::Either, score, true)
                    .unwrap_or_else(|e| panic!("{pattern} · {e}")),
            );
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    engine::Vocabulary {
        authority: o
            .get("authority")
            .and_then(serde_json::Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        v.as_u64()
                            .and_then(|n| u8::try_from(n).ok())
                            .map(|n| (k.to_uppercase(), n))
                    })
                    .collect()
            })
            .unwrap_or_default(),
        max_tier: f("authority_scale", 10.0) as f32,
        structural,
        structural_default: f("labelled_score", 0.6) as f32,
        stopwords: list("stopwords").iter().map(|s| s.to_lowercase()).collect(),
        min_term_chars: o
            .get("min_term_chars")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(3),
    }
}

fn facets(reg: &'static str, article: &'static str) -> engine::Facets<'static> {
    engine::Facets {
        regulation_type: Some(reg),
        article: Some(article),
        ..engine::Facets::default()
    }
}

#[test]
fn the_indonesian_profile_ranks_the_hierarchy_it_declares() {
    let v = load("vocab.id_regulation.json");

    assert_eq!(v.tier("UNDANG-UNDANG"), Some(8));
    assert_eq!(v.tier("PERATURAN BUPATI"), Some(2));
    assert!(
        engine::factors::authority(&facets("UNDANG-UNDANG", "Pasal 1"), &v)
            > engine::factors::authority(&facets("PERATURAN BUPATI", "Pasal 1"), &v)
    );
}

#[test]
fn the_corpus_declares_more_of_the_ladder_than_the_engine_ever_carried() {
    // ! The drift, measured from both sides. `engine::Vocabulary::id_regulation`
    // carries ten regulation types — the ones present in the live corpus — while
    // the profile declares the full ladder. Two copies of one table in two
    // repositories that must agree had ALREADY disagreed, which is the whole
    // argument for there being one.
    let corpus = load("vocab.id_regulation.json");
    let compiled = engine::Vocabulary::id_regulation();

    assert!(
        corpus.authority.len() > compiled.authority.len(),
        "corpus {} vs compiled {}",
        corpus.authority.len(),
        compiled.authority.len()
    );
    // Everything the engine knew, the corpus still knows — the extra entries are
    // additions, not a different table.
    for (label, tier) in &compiled.authority {
        assert_eq!(corpus.tier(label), Some(*tier), "{label}");
    }
}

#[test]
fn an_annex_still_ranks_below_an_operative_clause_under_the_corpus_ladder() {
    let v = load("vocab.id_regulation.json");

    let pasal = engine::factors::structural(&facets("UNDANG-UNDANG", "Pasal 5"), &v);
    let annex = engine::factors::structural(&facets("UNDANG-UNDANG", "LAMPIRAN I"), &v);
    assert!(annex < pasal, "annex {annex} · pasal {pasal}");
}

#[test]
fn a_marker_inside_prose_is_not_a_heading() {
    // ! Every corpus rule is anchored, matching `enrich/factors.py::_structural`.
    // "Ketentuan Pasal 9 dihapus" is an amending clause that CONTAINS an operative
    // marker without being one.
    let v = load("vocab.id_regulation.json");

    let mention = engine::factors::structural(&facets("UNDANG-UNDANG", "Ketentuan Pasal 9"), &v);
    let real = engine::factors::structural(&facets("UNDANG-UNDANG", "Pasal 9"), &v);
    assert!(mention < real, "mention {mention} · heading {real}");
}

#[test]
fn the_generic_profile_scores_an_english_corpus() {
    // The reason `generic@1.0.yaml` exists, and the reason the engine has a regex
    // dependency: `\d+(\.\d+)*\.?\s` is a real pattern, not a label.
    let v = load("vocab.generic.json");

    assert!(v.authority.is_empty(), "generic declares no hierarchy");
    let section = engine::factors::structural(&facets("", "Section 4.2"), &v);
    let numbered = engine::factors::structural(&facets("", "4.2 Retention of records"), &v);
    let appendix = engine::factors::structural(&facets("", "Appendix B"), &v);

    assert!((section - 1.0).abs() < f32::EPSILON, "{section}");
    assert!(
        (numbered - 1.0).abs() < f32::EPSILON,
        "a numbered section is operative · {numbered}"
    );
    assert!(
        appendix < section,
        "appendix {appendix} · section {section}"
    );
}

#[test]
fn no_indonesian_reaches_the_generic_vocabulary() {
    // Two profiles, two vocabularies. A table that leaked across would make the
    // factor a property of the engine again, just less visibly.
    let v = load("vocab.generic.json");

    assert_eq!(v.tier("UNDANG-UNDANG"), None);
    assert!(!v.stopwords.iter().any(|w| w == "yang"));
    assert!(v.stopwords.iter().any(|w| w == "the"));
}

#[test]
fn an_english_corpus_can_evaluate_the_shipped_algorithms_except_where_it_declares_nothing() {
    // ! The honest half. `generic` declares no hierarchy, so `authority` is inert —
    // and the engine REFUSES to start rather than serving a weight that cannot act.
    // That refusal is the feature: the alternative is an operator setting
    // FACTOR_AUTHORITY and never learning it did nothing.
    let v = load("vocab.generic.json");
    let missing = v.missing_for(&engine::Weights::FITTED);

    assert_eq!(missing, vec!["authority"]);
    // And its structural ladder IS declared, so that factor is usable.
    assert!(!v.structural.is_empty());
}
