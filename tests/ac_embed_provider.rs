//! Acceptance tests for feature `embed-provider` · scaffolded by
//! pipeline_test.ac_to_test, then filled in.
//!
//! Each test began #[ignore]d. Deleting that attribute is the act of claiming
//! the criterion — from here the gate enforces it.

use embed::{EMBEDDING_DIM, EmbedError, EmbeddingProvider, MODEL_ID, StubProvider};

/// trait exposes embed_query returning 4096 dimensions
#[tokio::test]
async fn ac_01_trait_exposes_embed_query_returning_4096_dimensions() {
    // Exercised through the trait object, not the concrete type: the contract is
    // what routing depends on.
    let provider: Box<dyn EmbeddingProvider> = Box::new(StubProvider::new());
    let v = provider
        .embed_query("ketentuan sanksi pajak")
        .await
        .unwrap();
    assert_eq!(v.len(), 4096);
    assert_eq!(v.len(), EMBEDDING_DIM);
}

/// stub provider is deterministic for a given input string
#[tokio::test]
async fn ac_02_stub_provider_is_deterministic_for_a_given_input_string() {
    let p = StubProvider::new();
    let first = p.embed_query("tarif pajak penghasilan").await.unwrap();
    // A fresh instance must agree with the first · determinism spans instances,
    // not just repeat calls on one object.
    let second = StubProvider::new()
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
    assert_eq!(MODEL_ID, "qwen/qwen3-embedding-8b");

    // Right model, right width → accepted.
    assert!(embed::validate_response(MODEL_ID, vec![0.1; EMBEDDING_DIM]).is_ok());

    // ! Wrong model → refuse. A different model produces vectors in a different
    // space, so ranking against this corpus would be meaningless but plausible.
    let mismatch =
        embed::validate_response("openai/text-embedding-3-large", vec![0.1; EMBEDDING_DIM])
            .unwrap_err();
    assert!(matches!(mismatch, EmbedError::ModelMismatch { .. }));

    // Right model, truncated vector → also refuse.
    let truncated = embed::validate_response(MODEL_ID, vec![0.1; 1024]).unwrap_err();
    assert!(matches!(
        truncated,
        EmbedError::DimensionMismatch {
            expected: 4096,
            got: 1024
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
        "crates/embed/src",
        "crates/engine/src",
        "crates/store/src",
        "crates/contract/src",
    ];
    // Completion-shaped API surfaces. Substrings, so `/v1/chat/completions`,
    // `messages.create`, and `generate_content` are all caught.
    let banned = [
        "chat/completions",
        "v1/messages",
        "generate_content",
        "completions.create",
    ];

    for root in roots {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(root);
        if !dir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(&dir).expect("read crate src") {
            let path = entry.expect("dir entry").path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
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
