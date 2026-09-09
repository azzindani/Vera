//! Layer-1 calibration · the geometry a corpus records about itself.
//!
//! ! Lives in `vera-core`, ✗ in `vera-index`, because **both sides need it**:
//! the offline indexer measures it, and the online engine reads it back to
//! decide a threshold. Putting it in the indexer would force the query path to
//! depend on the build path.
//!
//! ! The stats are stored, and the **threshold is derived at load time** rather
//! than baked in at build time. Retuning the policy must not require
//! re-ingesting the corpus — at 100M rows that is the difference between a
//! config change and a day of GPU time.

use serde::{Deserialize, Serialize};

/// How tightly a corpus sits around its own layer-1 anchor.
///
/// ! This measurement is what makes layer 1 usable at all. Transformer
/// embeddings are anisotropic — they occupy a cone whose width depends on the
/// model — so there is no universal "close enough to the domain" cosine.
/// Measuring it per corpus turns a guess into a calibration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnchorStats {
    pub min: f32,
    pub p1: f32,
    pub p5: f32,
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
    pub mean: f32,
    /// The threshold these stats imply under the default margin, recorded at
    /// build time for reference. The engine recomputes it — see the module note.
    pub threshold: f32,
}

impl AnchorStats {
    /// The layer-1 threshold these stats imply, given a tail `margin`.
    ///
    /// ```text
    /// threshold = p1 − margin · (p50 − p1)
    /// ```
    ///
    /// ! p1 alone is **not** safe, and the benchmark proved it: calibrating on
    /// corpus rows and applying the result to queries rejected **5%** of
    /// legitimate queries outright — five times the 1% the percentile implies.
    /// A query is not a row. It is *near* a row, and that displacement moves it
    /// further from the anchor than any document sits, so the row distribution
    /// systematically understates how far a real query can fall. The tail is
    /// extrapolated one more `(p50 − p1)` step down to absorb the drift.
    ///
    /// The asymmetry justifies erring low. A false reject loses the query
    /// entirely and answers `success: true` with nothing, which no caller can
    /// distinguish from "this corpus has no answer". A false accept merely
    /// returns weak results carrying a low `confidence` the agent can act on.
    #[must_use]
    pub fn threshold_at(&self, margin: f32) -> f32 {
        self.p1 - margin * (self.p50 - self.p1).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(p1: f32, p50: f32) -> AnchorStats {
        AnchorStats {
            min: p1 - 0.05,
            p1,
            p5: p1 + 0.01,
            p50,
            p95: p50 + 0.02,
            max: p50 + 0.05,
            mean: p50,
            threshold: 0.0,
        }
    }

    #[test]
    fn the_margin_extrapolates_one_step_below_p1() {
        // p1 0.71, p50 0.73 → spread 0.02 → threshold 0.69
        let s = stats(0.71, 0.73);
        assert!((s.threshold_at(1.0) - 0.69).abs() < 1e-5, "{}", s.threshold_at(1.0));
    }

    #[test]
    fn a_zero_margin_is_the_bare_percentile() {
        let s = stats(0.71, 0.73);
        assert!((s.threshold_at(0.0) - 0.71).abs() < 1e-6);
    }

    #[test]
    fn a_wider_corpus_gets_a_wider_safety_margin() {
        // ! The point of deriving the margin from the spread rather than fixing
        // it: a diffuse corpus needs more room than a tight one, and neither
        // number is knowable in advance.
        let tight = stats(0.71, 0.73);
        let loose = stats(0.30, 0.60);
        assert!(
            loose.threshold_at(1.0) < tight.threshold_at(1.0),
            "loose {} vs tight {}",
            loose.threshold_at(1.0),
            tight.threshold_at(1.0)
        );
    }

    #[test]
    fn an_inverted_distribution_does_not_raise_the_threshold() {
        // Degenerate input (p50 below p1) must not produce a *stricter* cutoff
        // than p1 — that would reject more, which is the unsafe direction.
        let weird = stats(0.80, 0.40);
        assert!((weird.threshold_at(1.0) - 0.80).abs() < 1e-6);
    }

    #[test]
    fn stats_survive_a_json_round_trip() {
        // They cross the process boundary as corpus metadata.
        let s = stats(0.71, 0.73);
        let back: AnchorStats = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }
}
