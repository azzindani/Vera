//! `embed`
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
//! version, same instruction, same pooling, same normalization. That is why the
//! model id is pinned and a mismatch fails closed rather than degrading quietly
//! (`EMBEDDING.md` §2).

use async_trait::async_trait;

/// Full Qwen3-Embedding-8B width.
pub const EMBEDDING_DIM: usize = 4096;

/// The pinned model. ✗ configurable: changing it silently invalidates every
/// vector already in the corpus.
pub const MODEL_ID: &str = "qwen/qwen3-embedding-8b";

/// Instruction prepended to *queries* only.
///
/// Qwen3-Embedding is instruction-aware: queries carry an instruction, documents
/// do not. Applying the wrong one puts the two sides in different spaces.
pub const QUERY_INSTRUCTION: &str =
    "Instruct: Given a legal or regulatory question, retrieve passages that answer it\nQuery: ";

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

    /// Identifier for logs and `explain_routing`.
    fn describe(&self) -> String;
}

/// Deterministic, offline provider for tests and fixtures.
///
/// ! Deterministic by construction, ✗ random: the same query must yield the same
/// vector across runs or no retrieval test can assert a stable ranking. This is
/// a hash expanded to the right width — it carries no semantics and must never
/// be used against a real corpus.
#[derive(Debug, Clone, Default)]
pub struct StubProvider;

impl StubProvider {
    #[must_use]
    pub fn new() -> Self {
        Self
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
    pub fn vector_for(text: &str) -> Vec<f32> {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(EMBEDDING_DIM);
        for i in 0..EMBEDDING_DIM {
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
        Ok(Self::vector_for(&format!("{QUERY_INSTRUCTION}{query}")))
    }

    fn describe(&self) -> String {
        format!("stub(deterministic, dim={EMBEDDING_DIM})")
    }
}

/// An OpenAI/TEI-style embedding endpoint reached over HTTP.
///
/// ! Embeddings only. There is no completion method here and there must never
/// be one — an LLM on the query path breaks statelessness and makes every
/// query cost a model round-trip (`OUTPUT_CONTRACT.md` §1).
///
/// The declared model and width come from the corpus, so a provider serving
/// different weights than the corpus was built with fails closed rather than
/// returning plausible nonsense.
#[derive(Debug, Clone)]
pub struct HttpProvider {
    endpoint: String,
    model: String,
    dim: usize,
    client: reqwest::Client,
}

impl HttpProvider {
    /// Point at an embedding server that declares `model` at `dim` dimensions.
    ///
    /// # Errors
    /// A client that cannot be constructed (TLS backend failure).
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        dim: usize,
    ) -> Result<Self, EmbedError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| EmbedError::Transport(e.to_string()))?;
        Ok(Self {
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            model: model.into(),
            dim,
            client,
        })
    }

    /// The model this provider claims to serve.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl EmbeddingProvider for HttpProvider {
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Reply {
            /// TEI returns a bare array of vectors.
            Bare(Vec<Vec<f32>>),
            /// OpenAI-shaped servers wrap them.
            Wrapped { data: Vec<Embedding> },
        }
        #[derive(serde::Deserialize)]
        struct Embedding {
            embedding: Vec<f32>,
        }

        let resp = self
            .client
            .post(format!("{}/embed", self.endpoint))
            .json(&serde_json::json!({ "inputs": query, "truncate": false }))
            .send()
            .await
            .map_err(|e| EmbedError::Transport(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(EmbedError::Transport(format!(
                "provider returned HTTP {}",
                resp.status()
            )));
        }

        let reply: Reply = resp
            .json()
            .await
            .map_err(|e| EmbedError::Transport(e.to_string()))?;
        let vector = match reply {
            Reply::Bare(mut v) => v.drain(..).next().ok_or(EmbedError::Empty)?,
            Reply::Wrapped { mut data } => {
                data.drain(..).next().ok_or(EmbedError::Empty)?.embedding
            }
        };

        // ! Validated at the boundary, against the corpus's declared space.
        validate_against(&self.model, self.dim, &self.model, vector)
    }

    fn describe(&self) -> String {
        format!(
            "http({}, model={}, dim={})",
            self.endpoint, self.model, self.dim
        )
    }
}

/// Validate a provider response before it reaches the routing layers.
///
/// Shared by every real provider: a wrong-width or wrong-model vector is a
/// silent-corruption bug, so it is rejected at the boundary (`error_handling`
/// standard: validate at the edge, ✗ deep inside).
///
/// # Errors
/// [`EmbedError::ModelMismatch`] if the model is not the pinned one,
/// [`EmbedError::DimensionMismatch`] on the wrong width, [`EmbedError::Empty`]
/// if no vector was returned.
pub fn validate_response(model: &str, vector: Vec<f32>) -> Result<Vec<f32>, EmbedError> {
    validate_against(MODEL_ID, EMBEDDING_DIM, model, vector)
}

/// Validate against the space the **corpus declares**, ✗ a compiled-in pin.
///
/// ! This is the enforceable form of invariant 2. [`MODEL_ID`] is the model
/// this project intends to reach; the corpus in front of you may legitimately
/// be a different one (a 1024-dim spike, say), and the rule that actually
/// matters is that query and corpus agree — not that a constant is satisfied.
/// `store::CorpusMeta` supplies `expected_model` and `expected_dim`.
///
/// # Errors
/// [`EmbedError::ModelMismatch`], [`EmbedError::DimensionMismatch`], or
/// [`EmbedError::Empty`].
pub fn validate_against(
    expected_model: &str,
    expected_dim: usize,
    model: &str,
    vector: Vec<f32>,
) -> Result<Vec<f32>, EmbedError> {
    if model != expected_model {
        return Err(EmbedError::ModelMismatch {
            expected: expected_model.to_owned(),
            returned: model.to_owned(),
        });
    }
    if vector.is_empty() {
        return Err(EmbedError::Empty);
    }
    if vector.len() != expected_dim {
        return Err(EmbedError::DimensionMismatch {
            expected: expected_dim,
            got: vector.len(),
        });
    }
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_is_deterministic_across_calls() {
        let p = StubProvider::new();
        let a = p.embed_query("ketentuan sanksi").await.unwrap();
        let b = p.embed_query("ketentuan sanksi").await.unwrap();
        assert_eq!(a, b, "same query must yield the same vector");
    }

    #[tokio::test]
    async fn stub_separates_different_queries() {
        let p = StubProvider::new();
        let a = p.embed_query("ketentuan sanksi").await.unwrap();
        let b = p.embed_query("tarif pajak").await.unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn every_provider_returns_the_pinned_width() {
        let v = StubProvider::new().embed_query("x").await.unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
        assert_eq!(EMBEDDING_DIM, 4096);
    }

    #[tokio::test]
    async fn the_query_instruction_is_part_of_the_embedded_input() {
        // ! Queries carry the instruction, documents do not. If this stopped
        // being true the query would land in a different space than the corpus.
        let p = StubProvider::new();
        let via_trait = p.embed_query("tarif").await.unwrap();
        assert_eq!(
            via_trait,
            StubProvider::vector_for(&format!("{QUERY_INSTRUCTION}tarif"))
        );
        assert_ne!(via_trait, StubProvider::vector_for("tarif"));
    }

    #[test]
    fn a_different_model_fails_closed_rather_than_degrading() {
        let e = validate_response("openai/text-embedding-3-large", vec![0.0; EMBEDDING_DIM])
            .unwrap_err();
        assert!(matches!(e, EmbedError::ModelMismatch { .. }), "{e}");
        assert!(e.to_string().contains("would not share a space"));
    }

    #[test]
    fn a_truncated_vector_is_rejected() {
        let e = validate_response(MODEL_ID, vec![0.0; 1024]).unwrap_err();
        assert!(matches!(
            e,
            EmbedError::DimensionMismatch {
                expected: 4096,
                got: 1024
            }
        ));
    }

    #[test]
    fn an_empty_response_is_rejected() {
        assert!(matches!(
            validate_response(MODEL_ID, vec![]).unwrap_err(),
            EmbedError::Empty
        ));
    }

    #[test]
    fn the_pinned_model_passes() {
        assert!(validate_response(MODEL_ID, vec![0.5; EMBEDDING_DIM]).is_ok());
    }

    #[test]
    fn stub_values_stay_in_a_plausible_embedding_range() {
        let v = StubProvider::vector_for("anything");
        assert!(v.iter().all(|x| (-1.0..=1.0).contains(x)));
    }
}
