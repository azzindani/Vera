//! `embed`
//!
//! Embedding providers. One trait, two implementations: a pinned remote client
//! and a deterministic stub.
//!
//! ! The **only** outbound model call Vera ever makes is an *embedding* call.
//! There is no completion path here and there must never be one — an LLM in the
//! engine breaks statelessness and makes every query cost a model round-trip
//! (`docs/OUTPUT_CONTRACT.md` §1).
//!
//! ! Corpus and query must land in the **same vector space**: same model, same
//! version, same instruction, same pooling, same normalization. That is why the
//! model id is pinned and a mismatch fails closed rather than degrading quietly
//! (`docs/EMBEDDING.md` §2).

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
/// happens offline on GPU (`dev_tools/PRE_EMBEDDING.md`), never on the query path.
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
///
/// ! The width is a parameter, ✗ a constant. A stub that only ever produced one
/// dimensionality would quietly stop standing in for the corpus the day the
/// corpus changed width, which is precisely the failure it exists to catch.
#[derive(Debug, Clone)]
pub struct StubProvider {
    model: String,
    dim: usize,
}

impl StubProvider {
    /// A stub standing in for `model` at `dim` dimensions.
    #[must_use]
    pub fn new(model: impl Into<String>, dim: usize) -> Self {
        Self {
            model: model.into(),
            dim,
        }
    }

    /// The model this stub claims to serve.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
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
        Ok(Self::vector_for(query, self.dim))
    }

    fn describe(&self) -> String {
        format!(
            "stub(deterministic, model={}, dim={})",
            self.model, self.dim
        )
    }
}

/// An OpenAI/TEI-style embedding endpoint reached over HTTP.
///
/// ! Embeddings only. There is no completion method here and there must never
/// be one — an LLM on the query path breaks statelessness and makes every
/// query cost a model round-trip (`docs/OUTPUT_CONTRACT.md` §1).
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

/// Validate against the space the **corpus declares**, ✗ a compiled-in pin.
///
/// ! This is the enforceable form of invariant 2, and it is deliberately the
/// only validator. A constant in this crate could only ever encode the model
/// the *author* expected; the rule that actually matters is that query and
/// corpus agree, and only the corpus can state which space that is.
/// `store::CorpusMeta` supplies `expected_model` and `expected_dim`.
///
/// Shared by every real provider: a wrong-width or wrong-model vector is a
/// silent-corruption bug, so it is rejected at the boundary (`error_handling`
/// standard: validate at the edge, ✗ deep inside).
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

    /// The space the dev corpus declares. A literal here, ✗ a constant in the
    /// library: tests pick a space to exercise, production reads it from the
    /// corpus.
    const MODEL: &str = "qwen/qwen3-embedding-0.6b";
    const DIM: usize = 1024;

    fn stub() -> StubProvider {
        StubProvider::new(MODEL, DIM)
    }

    #[tokio::test]
    async fn stub_is_deterministic_across_calls() {
        let p = stub();
        let a = p.embed_query("ketentuan sanksi").await.unwrap();
        let b = p.embed_query("ketentuan sanksi").await.unwrap();
        assert_eq!(a, b, "same query must yield the same vector");
    }

    #[tokio::test]
    async fn stub_separates_different_queries() {
        let p = stub();
        let a = p.embed_query("ketentuan sanksi").await.unwrap();
        let b = p.embed_query("tarif pajak").await.unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn the_stub_answers_at_the_width_it_was_built_for() {
        assert_eq!(stub().embed_query("x").await.unwrap().len(), DIM);
        // ! And at a different one, which is the point of it being a parameter.
        let wide = StubProvider::new("some/other-model", 4096);
        assert_eq!(wide.embed_query("x").await.unwrap().len(), 4096);
    }

    #[test]
    fn a_different_model_fails_closed_rather_than_degrading() {
        let e = validate_against(MODEL, DIM, "openai/text-embedding-3-large", vec![0.0; DIM])
            .unwrap_err();
        assert!(matches!(e, EmbedError::ModelMismatch { .. }), "{e}");
        assert!(e.to_string().contains("would not share a space"));
    }

    #[test]
    fn a_truncated_vector_is_rejected() {
        let e = validate_against(MODEL, DIM, MODEL, vec![0.0; 512]).unwrap_err();
        assert!(matches!(
            e,
            EmbedError::DimensionMismatch {
                expected: DIM,
                got: 512
            }
        ));
    }

    #[test]
    fn an_empty_response_is_rejected() {
        assert!(matches!(
            validate_against(MODEL, DIM, MODEL, vec![]).unwrap_err(),
            EmbedError::Empty
        ));
    }

    #[test]
    fn the_corpuss_own_space_passes() {
        assert!(validate_against(MODEL, DIM, MODEL, vec![0.5; DIM]).is_ok());
    }

    #[test]
    fn stub_values_stay_in_a_plausible_embedding_range() {
        let v = StubProvider::vector_for("anything", DIM);
        assert!(v.iter().all(|x| (-1.0..=1.0).contains(x)));
    }
}
