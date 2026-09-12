//! `vera-engine`
//!
//! Routing layers and fusion · the pure domain logic of retrieval.
//!
//! ! Zero I/O and zero MCP imports. Everything here is a function of numbers
//! and ranked lists, which is what makes it testable without a database and
//! what keeps the query path replicable (architecture/STANDARDS §2).
//!
//! The numbers in these modules' docs are measured, ✗ assumed — they come from
//! the 182K-row spike corpus, and `EVAL.md` is what keeps them honest.

pub mod fusion;
pub mod routing;

pub use fusion::{Arm, DEFAULT_K, Fused, reciprocal_rank_fusion};
pub use routing::{Centroid, DomainAnchor, Route, cosine, detect_domain, route, select_clusters};

/// How much the engine trusts a result set · mirrors `vera_core::Confidence`.
///
/// Kept as a separate mapping rather than a constructor on the contract type so
/// the thresholds live with the routing logic that produces them, ✗ with the
/// wire format that merely reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    High,
    Medium,
    Low,
    None,
}

/// Derive trust from what routing and retrieval actually found.
///
/// ! An empty result set is `None`, never `Low`. "I found nothing" and "I found
/// something weak" are different answers and the agent must be able to tell
/// them apart (`OUTPUT_CONTRACT.md` §4).
#[must_use]
pub fn trust_from(domain_matched: bool, results: usize, top_score: f32) -> Trust {
    if !domain_matched || results == 0 {
        return Trust::None;
    }
    if top_score >= 0.030 {
        Trust::High
    } else if top_score >= 0.020 {
        Trust::Medium
    } else {
        Trust::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_domain_means_no_confidence_regardless_of_score() {
        assert_eq!(trust_from(false, 10, 0.9), Trust::None);
    }

    #[test]
    fn an_empty_result_set_is_none_not_low() {
        assert_eq!(trust_from(true, 0, 0.0), Trust::None);
    }

    #[test]
    fn trust_tracks_the_top_fused_score() {
        assert_eq!(trust_from(true, 5, 0.040), Trust::High);
        assert_eq!(trust_from(true, 5, 0.025), Trust::Medium);
        assert_eq!(trust_from(true, 5, 0.005), Trust::Low);
    }
}
