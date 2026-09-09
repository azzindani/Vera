//! `vera-embed`
//!
//! Embedding providers. One trait, two implementations: a pinned remote client
//! and a deterministic stub.
//!
//! ! The **only** outbound model call Vera ever makes is an *embedding* call.
//! There is no completion path here and there must never be one — an LLM in the
//! engine breaks statelessness and makes every query cost a model round-trip
//! (`OUTPUT_CONTRACT.md` §1, `LOOPHOLES.md` §2).
//!
//! ! Corpus and query must land in the **same vector space**: same model, same
//! version, same instruction, same pooling, same normalization. That pin is
//! carried by [`vera_core::EmbeddingSpace`] and checked against what the corpus
//! recorded, so a mismatch fails closed rather than degrading quietly
//! (`EMBEDDING.md` §2).
//!
//! The space is **configuration**, ✗ a compile-time constant: the production
//! target is Qwen3-8B at 4096, but a corpus embedded with Qwen3-0.6B at 1024 is
//! equally valid and must not require a recompile to serve.

use async_trait::async_trait;
use vera_core::EmbeddingSpace;

/// The production model pin. A *default*, ✗ the only permitted value.
pub const DEFAULT_MODEL_ID: &str = "qwen/qwen3-embedding-8b";

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("embedding provider transport: {0}")]
    Transport(String),

    #[error(
        "provider returned model '{returned}' but this corpus was embedded with \
         '{expected}' · vectors would not share a space · refusing"
    )]
    ModelMismatch { expected: String, returned: String },

    #[error("provider returned {got} dimensions, expected {expected}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("provider returned no embedding for the query")]
    Empty,

    #[error(
        "startup canary: cosine {cosine:.6} against the stored reference is below \
         {threshold:.6} · the provider has drifted out of the corpus space · refusing to serve"
    )]
    CanaryDrift { cosine: f32, threshold: f32 },
}

/// A source of query embeddings.
///
/// Deliberately narrow: `embed_query` and nothing else. Document embedding
/// happens offline on GPU (`PRE_EMBEDDING.md`), never on the query path.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a query into the corpus's vector space.
    ///
    /// # Errors
    /// Transport failure, or a response that fails [`validate_response`].
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError>;

    /// The space this provider produces vectors in.
    fn space(&self) -> &EmbeddingSpace;

    /// Identifier for logs and the `search(dry_run)` routing plan.
    fn describe(&self) -> String {
        let s = self.space();
        format!("{}(dim={})", s.model_id, s.dim)
    }
}

/// Deterministic, offline provider for tests, fixtures and benchmarks.
///
/// ! Deterministic by construction, ✗ random: the same query must yield the same
/// vector across runs or no retrieval test can assert a stable ranking. This is
/// a hash expanded to the right width — it carries **no semantics** and must
/// never be pointed at a real corpus expecting meaningful ranking.
#[derive(Debug, Clone)]
pub struct StubProvider {
    space: EmbeddingSpace,
}

impl StubProvider {
    #[must_use]
    pub fn new(space: EmbeddingSpace) -> Self {
        Self { space }
    }

    /// FNV-1a · small, dependency-free, and stable across platforms and runs.
    fn hash(seed: u64, bytes: &[u8]) -> u64 {
        let mut h = seed ^ 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// The vector a given text maps to · exposed so tests can predict it.
    #[must_use]
    pub fn vector_for(text: &str, dim: usize) -> Vec<f32> {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            let h = Self::hash(i as u64, bytes);
            // Map into [-1, 1] · a plausible embedding range.
            #[allow(clippy::cast_precision_loss)]
            let unit = (h >> 11) as f64 / (1u64 << 53) as f64;
            #[allow(clippy::cast_possible_truncation)]
            out.push(((unit * 2.0) - 1.0) as f32);
        }
        out
    }
}

#[async_trait]
impl EmbeddingProvider for StubProvider {
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        // The instruction is part of the input on the real path, so the stub
        // applies it too · otherwise the two providers key on different strings.
        let text = format!("{}{query}", self.space.query_instruction);
        let mut v = Self::vector_for(&text, self.space.dim);
        if self.space.normalized {
            l2_normalize(&mut v);
        }
        Ok(v)
    }

    fn space(&self) -> &EmbeddingSpace {
        &self.space
    }

    fn describe(&self) -> String {
        format!("stub(deterministic, dim={})", self.space.dim)
    }
}

/// Scale a vector to unit length, in place. A zero vector is left alone.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity. Assumes nothing about normalization — divides by both
/// norms, so it is correct for raw and unit vectors alike.
///
/// Returns 0.0 for mismatched lengths or a zero vector: callers on the query
/// path have already validated width, and a zero-norm centroid is degenerate
/// rather than an error worth unwinding for.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= f32::EPSILON { 0.0 } else { dot / denom }
}

/// Validate a provider response before it reaches the routing layers.
///
/// Shared by every real provider: a wrong-width or wrong-model vector is a
/// silent-corruption bug, so it is rejected at the boundary (`error_handling`
/// standard: validate at the edge, ✗ deep inside).
///
/// # Errors
/// [`EmbedError::ModelMismatch`] if the model is not the configured one,
/// [`EmbedError::DimensionMismatch`] on the wrong width, [`EmbedError::Empty`]
/// if no vector was returned.
pub fn validate_response(
    space: &EmbeddingSpace,
    model: &str,
    vector: Vec<f32>,
) -> Result<Vec<f32>, EmbedError> {
    if model != space.model_id {
        return Err(EmbedError::ModelMismatch {
            expected: space.model_id.clone(),
            returned: model.to_owned(),
        });
    }
    if vector.is_empty() {
        return Err(EmbedError::Empty);
    }
    if vector.len() != space.dim {
        return Err(EmbedError::DimensionMismatch {
            expected: space.dim,
            got: vector.len(),
        });
    }
    Ok(vector)
}

/// Startup canary · `MCP_ENGINE.md` §6.3.
///
/// Embeds a known string and compares against the vector the corpus recorded
/// for it. Catches silent provider drift — a model retrained under the same id,
/// a provider swapped behind a router — *before* it corrupts a single result.
///
/// ! Runs at startup and fails closed. Serving on a drifted provider is worse
/// than not serving: results stay plausible while being wrong.
///
/// # Errors
/// [`EmbedError::CanaryDrift`] below threshold, or whatever the provider raised.
pub async fn canary_check(
    provider: &dyn EmbeddingProvider,
    canary_text: &str,
    reference: &[f32],
    threshold: f32,
) -> Result<f32, EmbedError> {
    let fresh = provider.embed_query(canary_text).await?;
    if fresh.len() != reference.len() {
        return Err(EmbedError::DimensionMismatch {
            expected: reference.len(),
            got: fresh.len(),
        });
    }
    let cos = cosine(&fresh, reference);
    if cos < threshold {
        return Err(EmbedError::CanaryDrift {
            cosine: cos,
            threshold,
        });
    }
    Ok(cos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space_1024() -> EmbeddingSpace {
        EmbeddingSpace::qwen3_06b()
    }

    #[tokio::test]
    async fn stub_is_deterministic_across_calls_and_instances() {
        let a = StubProvider::new(space_1024())
            .embed_query("ketentuan sanksi")
            .await
            .unwrap();
        let b = StubProvider::new(space_1024())
            .embed_query("ketentuan sanksi")
            .await
            .unwrap();
        assert_eq!(a, b, "same query must yield the same vector");
    }

    #[tokio::test]
    async fn stub_separates_different_queries() {
        let p = StubProvider::new(space_1024());
        let a = p.embed_query("ketentuan sanksi").await.unwrap();
        let b = p.embed_query("tarif pajak").await.unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn the_provider_returns_its_configured_width_whatever_that_is() {
        // ! The point of making the space configurable: 1024 and 4096 are both
        // valid, and neither is baked in.
        for space in [EmbeddingSpace::qwen3_06b(), EmbeddingSpace::qwen3_8b()] {
            let dim = space.dim;
            let v = StubProvider::new(space).embed_query("x").await.unwrap();
            assert_eq!(v.len(), dim);
        }
    }

    #[tokio::test]
    async fn a_normalized_space_yields_unit_vectors() {
        let v = StubProvider::new(space_1024()).embed_query("x").await.unwrap();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[tokio::test]
    async fn the_query_instruction_is_part_of_the_embedded_input() {
        // ! Queries carry the instruction, documents do not. If this stopped
        // being true the query would land in a different space than the corpus.
        let space = space_1024();
        let instruction = space.query_instruction.clone();
        let dim = space.dim;
        let via_trait = StubProvider::new(space).embed_query("tarif").await.unwrap();

        let mut expected = StubProvider::vector_for(&format!("{instruction}tarif"), dim);
        l2_normalize(&mut expected);
        assert_eq!(via_trait, expected);

        let mut bare = StubProvider::vector_for("tarif", dim);
        l2_normalize(&mut bare);
        assert_ne!(via_trait, bare);
    }

    #[test]
    fn a_different_model_fails_closed_rather_than_degrading() {
        let space = space_1024();
        let e =
            validate_response(&space, "openai/text-embedding-3-large", vec![0.0; space.dim])
                .unwrap_err();
        assert!(matches!(e, EmbedError::ModelMismatch { .. }), "{e}");
        assert!(e.to_string().contains("would not share a space"));
    }

    #[test]
    fn a_wrong_width_vector_is_rejected_against_the_configured_space() {
        // 4096 is *wrong* here — the configured corpus is 1024. Width is only
        // meaningful relative to the space, which is why it is not a constant.
        let space = space_1024();
        let e = validate_response(&space, &space.model_id, vec![0.0; 4096]).unwrap_err();
        assert!(matches!(
            e,
            EmbedError::DimensionMismatch {
                expected: 1024,
                got: 4096
            }
        ));
    }

    #[test]
    fn an_empty_response_is_rejected() {
        let space = space_1024();
        assert!(matches!(
            validate_response(&space, &space.model_id, vec![]).unwrap_err(),
            EmbedError::Empty
        ));
    }

    #[test]
    fn the_configured_model_at_the_configured_width_passes() {
        let space = space_1024();
        assert!(validate_response(&space, &space.model_id, vec![0.5; 1024]).is_ok());
    }

    #[test]
    fn cosine_is_one_for_identical_and_zero_for_orthogonal() {
        assert!((cosine(&[1.0, 0.0], &[3.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_degrades_to_zero_rather_than_panicking_on_bad_input() {
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0, "length mismatch");
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0, "zero vector");
    }

    #[tokio::test]
    async fn the_canary_passes_against_a_provider_that_has_not_drifted() {
        let p = StubProvider::new(space_1024());
        let reference = p.embed_query("canary").await.unwrap();
        let cos = canary_check(&p, "canary", &reference, 0.999).await.unwrap();
        assert!(cos > 0.999);
    }

    #[tokio::test]
    async fn the_canary_refuses_to_serve_a_drifted_provider() {
        // ! The failure this exists to catch: the provider still answers, still
        // returns the right width — it just answers from a different space.
        let p = StubProvider::new(space_1024());
        let wrong_reference = p.embed_query("a completely different string").await.unwrap();
        let e = canary_check(&p, "canary", &wrong_reference, 0.999)
            .await
            .unwrap_err();
        assert!(matches!(e, EmbedError::CanaryDrift { .. }), "{e}");
        assert!(e.to_string().contains("refusing to serve"));
    }
}
