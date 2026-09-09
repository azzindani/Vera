//! Acceptance tests for feature `embed-provider` · scaffolded by
//! pipeline_test.ac_to_test, then filled in.
//!
//! Each test began #[ignore]d. Deleting that attribute is the act of claiming
//! the criterion — from here the gate enforces it.
//!
//! ! AC-01 was originally worded "returns 4096 dimensions". The *criterion*
//! behind it is that the provider returns the width the corpus was built at,
//! and 4096 was that width only because Qwen3-8B was assumed. Pinning the
//! literal made a 1024-dim corpus unservable while catching no bug the space
//! check does not catch, so the assertion now tests the width **the configured
//! space declares** — with 4096 still exercised as one of the cases.

use vera_core::EmbeddingSpace;
use vera_embed::{DEFAULT_MODEL_ID, EmbedError, EmbeddingProvider, StubProvider};

/// trait exposes embed_query returning the configured number of dimensions
#[tokio::test]
async fn ac_01_trait_exposes_embed_query_returning_the_configured_dimensions() {
    // Exercised through the trait object, not the concrete type: the contract is
    // what routing depends on.
    for space in [EmbeddingSpace::qwen3_8b(), EmbeddingSpace::qwen3_06b()] {
        let declared = space.dim;
        let provider: Box<dyn EmbeddingProvider> = Box::new(StubProvider::new(space));
        let v = provider.embed_query("ketentuan sanksi pajak").await.unwrap();
        assert_eq!(v.len(), declared);
        assert_eq!(v.len(), provider.space().dim);
    }
}

/// stub provider is deterministic for a given input string
#[tokio::test]
async fn ac_02_stub_provider_is_deterministic_for_a_given_input_string() {
    let p = StubProvider::new(EmbeddingSpace::qwen3_06b());
    let first = p.embed_query("tarif pajak penghasilan").await.unwrap();
    // A fresh instance must agree with the first · determinism spans instances,
    // not just repeat calls on one object.
    let second = StubProvider::new(EmbeddingSpace::qwen3_06b())
        .embed_query("tarif pajak penghasilan")
        .await
        .unwrap();
    assert_eq!(first, second);
    // …and different inputs must not collide.
    let other = p.embed_query("sanksi administrasi").await.unwrap();
    assert_ne!(first, other);
}

/// real client pins the model id and fails closed on mismatch
#[test]
fn ac_03_real_client_pins_the_model_id_and_fails_closed_on_mismatch() {
    assert_eq!(DEFAULT_MODEL_ID, "qwen/qwen3-embedding-8b");
    let space = EmbeddingSpace::qwen3_8b();
    assert_eq!(space.model_id, DEFAULT_MODEL_ID);

    // Right model, right width → accepted.
    assert!(vera_embed::validate_response(&space, &space.model_id, vec![0.1; space.dim]).is_ok());

    // ! Wrong model → refuse. A different model produces vectors in a different
    // space, so ranking against this corpus would be meaningless but plausible.
    let mismatch = vera_embed::validate_response(
        &space,
        "openai/text-embedding-3-large",
        vec![0.1; space.dim],
    )
    .unwrap_err();
    assert!(matches!(mismatch, EmbedError::ModelMismatch { .. }));

    // Right model, truncated vector → also refuse.
    let truncated =
        vera_embed::validate_response(&space, &space.model_id, vec![0.1; 1024]).unwrap_err();
    assert!(matches!(
        truncated,
        EmbedError::DimensionMismatch {
            expected: 4096,
            got: 1024
        }
    ));

    // ! And the mirror image, which is the mistake this project actually made:
    // a 1024-dim corpus must reject a 4096-dim vector just as firmly. Width is
    // meaningful only relative to the configured space.
    let small = EmbeddingSpace::qwen3_06b();
    let oversized =
        vera_embed::validate_response(&small, &small.model_id, vec![0.1; 4096]).unwrap_err();
    assert!(matches!(
        oversized,
        EmbedError::DimensionMismatch {
            expected: 1024,
            got: 4096
        }
    ));
}

/// no LLM completion call exists anywhere on the query path
#[test]
fn ac_04_no_llm_completion_call_exists_anywhere_on_the_query_path() {
    // ! Structural check, not a runtime one. The invariant is "the engine never
    // calls an LLM" (OUTPUT_CONTRACT.md §1) — a unit test cannot observe the
    // absence of a call, so assert against the source of the query-path crates.
    let roots = [
        "crates/vera-embed/src",
        "crates/vera-engine/src",
        "crates/vera-store/src",
        "crates/vera-core/src",
    ];
    // Completion-shaped API surfaces. Substrings, so `/v1/chat/completions`,
    // `messages.create`, and `generate_content` are all caught.
    let banned = [
        "chat/completions",
        "v1/messages",
        "generate_content",
        "completions.create",
    ];

    let mut scanned = 0usize;
    for root in roots {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(root);
        if !dir.exists() {
            continue;
        }
        // ! Recursive: the original walked one level, so a completion call in
        // any submodule (retrieval/, provider/, routing/ — all planned) would
        // have passed unseen.
        let mut stack = vec![dir];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("read crate src") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().is_none_or(|e| e != "rs") {
                    continue;
                }
                scanned += 1;
                let text = std::fs::read_to_string(&path).expect("read source");
                for needle in banned {
                    assert!(
                        !text.contains(needle),
                        "the query path must never call a completion endpoint: \
                         {} contains {needle}",
                        path.display()
                    );
                }
            }
        }
    }
    assert!(scanned > 0, "scanned no sources · the check was vacuous");
}
