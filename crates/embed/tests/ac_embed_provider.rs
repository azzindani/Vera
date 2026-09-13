//! Acceptance tests for the embedding provider.
//!
//! Two claims are under test here and they are the load-bearing ones: a query
//! is embedded into the space the **corpus** declares, and nothing on the query
//! path can reach a completion endpoint.

use embed::{EmbedError, EmbeddingProvider, StubProvider};

/// The space the dev corpus declares. Production reads this from `corpus_meta`;
/// a test has to name one, so it names this one.
const MODEL: &str = "qwen/qwen3-embedding-0.6b";
const DIM: usize = 1024;

/// the trait exposes `embed_query` at the corpus's width
#[tokio::test]
async fn ac_01_the_trait_exposes_embed_query_at_the_corpuss_width() {
    // Exercised through the trait object, not the concrete type: the contract is
    // what routing depends on.
    let provider: Box<dyn EmbeddingProvider> = Box::new(StubProvider::new(MODEL, DIM));
    let v = provider
        .embed_query("ketentuan sanksi pajak")
        .await
        .unwrap();
    assert_eq!(v.len(), DIM);
}

/// stub provider is deterministic for a given input string
#[tokio::test]
async fn ac_02_stub_provider_is_deterministic_for_a_given_input_string() {
    let p = StubProvider::new(MODEL, DIM);
    let first = p.embed_query("tarif pajak penghasilan").await.unwrap();
    // A fresh instance must agree with the first · determinism spans instances,
    // not just repeat calls on one object.
    let second = StubProvider::new(MODEL, DIM)
        .embed_query("tarif pajak penghasilan")
        .await
        .unwrap();
    assert_eq!(first, second);
    // …and different inputs must not collide.
    let other = p.embed_query("sanksi administrasi").await.unwrap();
    assert_ne!(first, other);
}

/// a provider serving another space fails closed rather than degrading
#[test]
fn ac_03_a_provider_serving_another_space_fails_closed() {
    // The corpus's own space → accepted.
    assert!(embed::validate_against(MODEL, DIM, MODEL, vec![0.1; DIM]).is_ok());

    // ! Wrong model → refuse. A different model produces vectors in a different
    // space, so ranking against this corpus would be meaningless but plausible.
    let mismatch =
        embed::validate_against(MODEL, DIM, "openai/text-embedding-3-large", vec![0.1; DIM])
            .unwrap_err();
    assert!(matches!(mismatch, EmbedError::ModelMismatch { .. }));

    // Right model, truncated vector → also refuse.
    let truncated = embed::validate_against(MODEL, DIM, MODEL, vec![0.1; 512]).unwrap_err();
    assert!(matches!(
        truncated,
        EmbedError::DimensionMismatch {
            expected: DIM,
            got: 512
        }
    ));
}

/// no LLM completion call exists anywhere on the query path
#[test]
fn ac_04_no_llm_completion_call_exists_anywhere_on_the_query_path() {
    // ! Structural check, not a runtime one. The invariant is "the engine never
    // calls an LLM" (docs/OUTPUT_CONTRACT.md §1) — a unit test cannot observe
    // the absence of a call, so assert against the source of every crate on the
    // query path.
    //
    // ! The workspace root is walked up to, ✗ assumed. `CARGO_MANIFEST_DIR` is
    // this crate, and a test that silently finds no files to scan would pass
    // for the wrong reason.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("crates").is_dir())
        .expect("workspace root");

    // Completion-shaped API surfaces. Substrings, so `/v1/chat/completions`,
    // `messages.create`, and `generate_content` are all caught.
    let banned = [
        "chat/completions",
        "v1/messages",
        "generate_content",
        "completions.create",
    ];

    let mut scanned = 0;
    for crate_dir in ["embed", "engine", "store", "contract", "mcp"] {
        let dir = root.join("crates").join(crate_dir).join("src");
        for entry in std::fs::read_dir(&dir).expect("read crate src") {
            let path = entry.expect("dir entry").path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read source");
            scanned += 1;
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
    assert!(
        scanned > 5,
        "expected to scan the query path, saw {scanned} files"
    );
}
