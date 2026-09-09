//! Layer-1 domain detection and layer-2 cluster selection.
//!
//! Both structures are small and both are held **hot** in engine memory: layer 1
//! is a handful of anchors, layer 2 is ~10K centroids (~40 MB at 1024 dims,
//! ~160 MB at 4096). Neither is re-read per query — they are the index, and the
//! whole point of the design is that the index is small enough to keep resident
//! while the corpus is not (`ARCHITECTURE.md` §2).

use vera_embed::cosine;
use vera_store::{Centroid, DomainAnchor};

/// A layer-2 cluster chosen for probing, with the score that chose it.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbedCluster {
    pub cluster_id: i32,
    pub similarity: f32,
    pub row_count: i64,
}

/// The outcome of layer 1.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedDomain {
    pub id: String,
    pub similarity: f32,
}

/// Match a query vector against the layer-1 anchors.
///
/// Returns `None` when nothing clears `threshold`.
///
/// ! `None` is a **result**, ✗ an error. Forcing an off-topic query into the
/// nearest domain returns confident, well-formed, wrong evidence — which a
/// downstream agent has no way to detect. Returning nothing is honest and is
/// what `CLAUDE.md` §7 rule 13 requires.
#[must_use]
pub fn detect_domain(
    query: &[f32],
    anchors: &[DomainAnchor],
    threshold: f32,
) -> Option<DetectedDomain> {
    detect_domain_with(query, anchors, |_| threshold)
}

/// [`detect_domain`] with a **per-domain** threshold.
///
/// ! Each domain gets its own threshold because each is calibrated against its
/// own corpus. Two knowledge bases embedded with the same model still occupy
/// differently shaped regions of the space, so one shared cutoff would be too
/// tight for one and too loose for the other — and "too tight" fails silently.
#[must_use]
pub fn detect_domain_with(
    query: &[f32],
    anchors: &[DomainAnchor],
    threshold_for: impl Fn(&str) -> f32,
) -> Option<DetectedDomain> {
    anchors
        .iter()
        .map(|a| (a, cosine(query, &a.anchor)))
        .filter(|(a, sim)| *sim >= threshold_for(&a.id))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(a, sim)| DetectedDomain {
            id: a.id.clone(),
            similarity: sim,
        })
}

/// Rank layer-2 centroids and take the `n` nearest.
///
/// This is the recall/latency dial. Raising `n` costs latency linearly and RAM
/// not at all, because the clusters it selects are loaded one at a time.
#[must_use]
pub fn nearest_clusters(query: &[f32], centroids: &[Centroid], n: usize) -> Vec<ProbedCluster> {
    let mut ranked: Vec<ProbedCluster> = centroids
        .iter()
        .map(|c| ProbedCluster {
            cluster_id: c.id,
            similarity: cosine(query, &c.centroid),
            row_count: c.row_count,
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Deterministic tie-break · an unstable probe set would make the
            // eval harness report different recall for identical inputs.
            .then_with(|| a.cluster_id.cmp(&b.cluster_id))
    });
    ranked.truncate(n);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(id: &str, v: &[f32]) -> DomainAnchor {
        DomainAnchor {
            id: id.to_owned(),
            description: String::new(),
            anchor: v.to_vec(),
            row_count: 0,
        }
    }

    fn centroid(id: i32, v: &[f32]) -> Centroid {
        Centroid {
            id,
            domain_id: "d".to_owned(),
            centroid: v.to_vec(),
            row_count: 100,
        }
    }

    #[test]
    fn the_closest_anchor_above_threshold_wins() {
        let anchors = [
            anchor("regulations", &[1.0, 0.0]),
            anchor("contracts", &[0.0, 1.0]),
        ];
        let d = detect_domain(&[0.9, 0.1], &anchors, 0.25).unwrap();
        assert_eq!(d.id, "regulations");
        assert!(d.similarity > 0.9);
    }

    #[test]
    fn an_off_topic_query_detects_nothing_rather_than_the_nearest_domain() {
        // ! The load-bearing case. [0,1] is orthogonal to the only anchor, so
        // "nearest" would still be `regulations` — and would be wrong.
        let anchors = [anchor("regulations", &[1.0, 0.0])];
        assert!(detect_domain(&[0.0, 1.0], &anchors, 0.25).is_none());
    }

    #[test]
    fn a_query_exactly_at_the_threshold_is_accepted() {
        let anchors = [anchor("d", &[1.0, 0.0])];
        // cosine([1,1],[1,0]) is exactly 1/√2 · the comparison is `>=`, so a
        // threshold at the value itself admits and a hair above rejects.
        let exact = std::f32::consts::FRAC_1_SQRT_2;
        assert!(detect_domain(&[1.0, 1.0], &anchors, exact).is_some());
        assert!(detect_domain(&[1.0, 1.0], &anchors, exact + 1e-4).is_none());
    }

    #[test]
    fn an_empty_corpus_detects_no_domain() {
        assert!(detect_domain(&[1.0, 0.0], &[], 0.0).is_none());
    }

    #[test]
    fn clusters_come_back_nearest_first_and_capped() {
        let centroids = [
            centroid(1, &[1.0, 0.0]),
            centroid(2, &[0.7, 0.7]),
            centroid(3, &[0.0, 1.0]),
        ];
        let probed = nearest_clusters(&[1.0, 0.0], &centroids, 2);
        assert_eq!(probed.len(), 2);
        assert_eq!(probed[0].cluster_id, 1);
        assert_eq!(probed[1].cluster_id, 2);
        assert!(probed[0].similarity > probed[1].similarity);
    }

    #[test]
    fn asking_for_more_clusters_than_exist_returns_what_there_is() {
        let centroids = [centroid(1, &[1.0, 0.0])];
        assert_eq!(nearest_clusters(&[1.0, 0.0], &centroids, 50).len(), 1);
    }

    #[test]
    fn probing_zero_clusters_yields_none() {
        let centroids = [centroid(1, &[1.0, 0.0])];
        assert!(nearest_clusters(&[1.0, 0.0], &centroids, 0).is_empty());
    }

    #[test]
    fn tied_centroids_are_ordered_deterministically() {
        // ! Identical centroids · without the id tie-break the probe set would
        // vary run to run and so would measured recall.
        let centroids = [
            centroid(7, &[1.0, 0.0]),
            centroid(3, &[1.0, 0.0]),
            centroid(5, &[1.0, 0.0]),
        ];
        for _ in 0..20 {
            let ids: Vec<i32> = nearest_clusters(&[1.0, 0.0], &centroids, 2)
                .iter()
                .map(|p| p.cluster_id)
                .collect();
            assert_eq!(ids, [3, 5]);
        }
    }

    #[test]
    fn probing_more_clusters_never_drops_one_it_already_chose() {
        // ! The dial must be monotone: raising clusters_probed can only add
        // candidates. If it could reorder, recall would not be monotone in the
        // dial and tuning it would be meaningless.
        let centroids: Vec<Centroid> = (0i16..20)
            .map(|i| {
                centroid(
                    i32::from(i),
                    &[1.0 - f32::from(i) * 0.01, f32::from(i) * 0.01],
                )
            })
            .collect();
        let query = [1.0, 0.0];
        let small: Vec<i32> = nearest_clusters(&query, &centroids, 3)
            .iter()
            .map(|p| p.cluster_id)
            .collect();
        let large: Vec<i32> = nearest_clusters(&query, &centroids, 10)
            .iter()
            .map(|p| p.cluster_id)
            .collect();
        assert_eq!(large[..3], small[..], "the first 3 must be unchanged");
    }
}
