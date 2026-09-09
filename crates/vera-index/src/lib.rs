//! `vera-index`
//!
//! Offline index construction: the layer-2 k-means clustering and the layer-1
//! anchors. This is the `cluster_maint` pipeline's core
//! (`CLUSTER_MAINTENANCE.md`), and it is deliberately **not** on the query path
//! — nothing here is called while serving.
//!
//! ! The engine's whole performance argument rests on the assumption that
//! clusters are semantically coherent. If clustering is poor, routing prunes the
//! wrong 99% and recall collapses while latency still looks excellent. So this
//! module is measured by the eval harness, ✗ trusted.

pub mod build;
pub mod kmeans;
pub mod matrix;
pub mod preflight;
pub mod split;
pub mod stream;

pub use build::{BuildError, BuildReport, IngestRow, build_corpus, measure_anchor};
pub use kmeans::{KMeans, KMeansConfig, kmeans};
pub use matrix::Matrix;
pub use preflight::{PreflightError, SpaceCheck, estimated_bytes, require_free_space};
pub use split::{SplitReport, split_oversized};
pub use stream::{BUILD_STATE_KEY, RowSource, SliceSource, build_corpus_streaming};

/// Deterministic, seedable PRNG (PCG-XSH-RR 64/32).
///
/// ! Hand-rolled and seeded so index builds are **reproducible**: rebuilding an
/// index from the same corpus must produce the same clusters, or an eval result
/// cannot be attributed to a code change rather than to a reshuffle.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }

    pub fn next_u32(&mut self) -> u32 {
        let state = self.0;
        self.0 = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
        let rot = (state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, n)`. Returns 0 for `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u32() as usize) % n
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        {
            self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_replays_the_same_sequence() {
        // ! Reproducible index builds depend on this.
        let a: Vec<u32> = (0..50).map(|_| Rng::new(42).next_u32()).take(1).collect();
        let mut r1 = Rng::new(7);
        let mut r2 = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(r1.next_u32(), r2.next_u32());
        }
        assert!(!a.is_empty());
    }

    #[test]
    fn different_seeds_diverge() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        let a: Vec<u32> = (0..8).map(|_| r1.next_u32()).collect();
        let b: Vec<u32> = (0..8).map(|_| r2.next_u32()).collect();
        assert_ne!(a, b);
    }

    #[test]
    fn below_stays_in_range_and_handles_zero() {
        let mut r = Rng::new(3);
        for _ in 0..1_000 {
            assert!(r.below(10) < 10);
        }
        assert_eq!(r.below(0), 0);
    }

    #[test]
    fn unit_stays_in_the_half_open_interval() {
        let mut r = Rng::new(5);
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u), "{u}");
        }
    }
}
