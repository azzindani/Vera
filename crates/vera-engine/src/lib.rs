//! `vera-engine`
//!
//! Routing, leaf scan, and fusion. The retrieval core, with zero MCP imports —
//! the tool surface wraps this, never the other way round.
//!
//! Reading order: [`routing`] (which clusters), [`topk`] (bounded candidate
//! keeping), [`identifier`] (the routing bypass), [`fusion`] (combining the two
//! halves), then [`search`] which sequences all of it and times each stage.

pub mod fusion;
pub mod identifier;
pub mod routing;
pub mod search;
pub mod topk;

pub use fusion::{Fused, RankedList, reciprocal_rank_fusion};
pub use identifier::Identifier;
pub use routing::{DetectedDomain, ProbedCluster, detect_domain, nearest_clusters};
pub use search::{Candidates, Engine, EngineError, Probe, SearchOutcome, StageTimings};
pub use topk::{Scored, TopK};
