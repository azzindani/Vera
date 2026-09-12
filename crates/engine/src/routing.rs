//! Layer-1 domain detection and layer-2 cluster selection.
//!
//! ! The agent never passes a domain (`CLAUDE.md` §7.13). It is detected here
//! by anchor match on the query vector, and if nothing clears the threshold the
//! honest answer is *no domain* — ✗ the closest guess. A confidently wrong
//! domain is worse than an empty result, because the agent cannot tell it
//! happened.
//!
//! Measured on the 182K-row spike: probing 5 of 85 clusters touches 5.9% of the
//! corpus and retains 95.0% routing recall. The 5% that misses is exactly what
//! the global exact-identifier path exists to catch (`LOOPHOLES.md` §1) —
//! routing accelerates, the keyword net guarantees.

/// Cosine similarity between two vectors of equal length.
///
/// Corpus vectors are L2-normalised at ingest, so this is usually a plain dot
/// product — but normalising defensively costs little and stops an
/// unnormalised centroid from silently skewing every comparison.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
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
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// A knowledge base and the pre-embedded anchor that identifies it.
#[derive(Debug, Clone)]
pub struct DomainAnchor {
    pub id: String,
    pub anchor: Vec<f32>,
}

/// A layer-2 centroid.
#[derive(Debug, Clone)]
pub struct Centroid {
    pub id: i32,
    pub vector: Vec<f32>,
}

/// What routing decided · surfaced verbatim by `explain_routing`.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub domain: Option<String>,
    pub domain_confidence: f32,
    /// Cluster ids to scan, nearest first. Empty when no domain matched.
    pub clusters: Vec<i32>,
    /// `(cluster id, similarity)` for every probed cluster, for transparency.
    pub cluster_scores: Vec<(i32, f32)>,
}

/// Pick the domain whose anchor the query is nearest, if any clears `threshold`.
///
/// ! Returns `None` rather than the argmax when nothing clears the bar.
#[must_use]
pub fn detect_domain(
    query: &[f32],
    anchors: &[DomainAnchor],
    threshold: f32,
) -> Option<(String, f32)> {
    anchors
        .iter()
        .map(|d| (d.id.clone(), cosine(query, &d.anchor)))
        .filter(|(_, s)| *s >= threshold)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// Select the `probe` nearest centroids.
///
/// Ties break on cluster id so a query always probes the same clusters — an
/// unstable choice here would make routing recall unreproducible.
#[must_use]
pub fn select_clusters(query: &[f32], centroids: &[Centroid], probe: usize) -> Vec<(i32, f32)> {
    let mut scored: Vec<(i32, f32)> = centroids
        .iter()
        .map(|c| (c.id, cosine(query, &c.vector)))
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(probe);
    scored
}

/// Full routing decision for one query.
#[must_use]
pub fn route(
    query: &[f32],
    anchors: &[DomainAnchor],
    centroids: &[Centroid],
    threshold: f32,
    probe: usize,
) -> Route {
    match detect_domain(query, anchors, threshold) {
        None => Route {
            domain: None,
            domain_confidence: 0.0,
            clusters: Vec::new(),
            cluster_scores: Vec::new(),
        },
        Some((domain, confidence)) => {
            let scores = select_clusters(query, centroids, probe);
            Route {
                domain: Some(domain),
                domain_confidence: confidence,
                clusters: scores.iter().map(|(id, _)| *id).collect(),
                cluster_scores: scores,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn anchors() -> Vec<DomainAnchor> {
        vec![
            DomainAnchor {
                id: "regulations".into(),
                anchor: unit(&[1.0, 0.0, 0.0]),
            },
            DomainAnchor {
                id: "medical".into(),
                anchor: unit(&[0.0, 1.0, 0.0]),
            },
        ]
    }

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let v = unit(&[0.3, 0.4, 0.5]);
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_mismatched_lengths_is_zero_rather_than_panicking() {
        assert!(cosine(&[1.0, 0.0], &[1.0]).abs() < f32::EPSILON);
        assert!(cosine(&[], &[]).abs() < f32::EPSILON);
    }

    #[test]
    fn the_nearest_anchor_above_threshold_wins() {
        let q = unit(&[0.9, 0.1, 0.0]);
        let (id, score) = detect_domain(&q, &anchors(), 0.5).unwrap();
        assert_eq!(id, "regulations");
        assert!(score > 0.9);
    }

    #[test]
    fn nothing_above_threshold_returns_none_rather_than_the_closest() {
        // ! The core of invariant 13. This query is nearest to "regulations",
        // but not near enough — so the answer is None, ✗ "regulations".
        let q = unit(&[0.0, 0.0, 1.0]);
        assert!(detect_domain(&q, &anchors(), 0.5).is_none());
        let r = route(&q, &anchors(), &[], 0.5, 5);
        assert!(r.domain.is_none());
        assert!(r.domain_confidence.abs() < f32::EPSILON);
        assert!(r.clusters.is_empty(), "no domain means nothing to probe");
    }

    #[test]
    fn probing_selects_the_nearest_centroids_in_order() {
        let centroids = vec![
            Centroid {
                id: 1,
                vector: unit(&[1.0, 0.0, 0.0]),
            },
            Centroid {
                id: 2,
                vector: unit(&[0.0, 1.0, 0.0]),
            },
            Centroid {
                id: 3,
                vector: unit(&[0.9, 0.1, 0.0]),
            },
        ];
        let q = unit(&[1.0, 0.0, 0.0]);
        let picked = select_clusters(&q, &centroids, 2);
        assert_eq!(picked.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [1, 3]);
        assert!(
            picked[0].1 >= picked[1].1,
            "sorted by similarity, best first"
        );
    }

    #[test]
    fn probing_more_clusters_than_exist_is_not_an_error() {
        let centroids = vec![Centroid {
            id: 7,
            vector: unit(&[1.0, 0.0, 0.0]),
        }];
        assert_eq!(
            select_clusters(&unit(&[1.0, 0.0, 0.0]), &centroids, 50).len(),
            1
        );
    }

    #[test]
    fn cluster_selection_is_deterministic_when_similarities_tie() {
        // Two identical centroids · the lower id must always win, or routing
        // recall stops being reproducible between runs.
        let centroids = vec![
            Centroid {
                id: 9,
                vector: unit(&[1.0, 0.0, 0.0]),
            },
            Centroid {
                id: 4,
                vector: unit(&[1.0, 0.0, 0.0]),
            },
        ];
        let q = unit(&[1.0, 0.0, 0.0]);
        assert_eq!(select_clusters(&q, &centroids, 1)[0].0, 4);
        assert_eq!(
            select_clusters(&q, &centroids, 1),
            select_clusters(&q, &centroids, 1)
        );
    }

    #[test]
    fn a_matched_domain_yields_probed_clusters_and_their_scores() {
        let centroids = vec![
            Centroid {
                id: 1,
                vector: unit(&[1.0, 0.0, 0.0]),
            },
            Centroid {
                id: 2,
                vector: unit(&[0.0, 0.0, 1.0]),
            },
        ];
        let r = route(&unit(&[1.0, 0.0, 0.0]), &anchors(), &centroids, 0.5, 2);
        assert_eq!(r.domain.as_deref(), Some("regulations"));
        assert_eq!(r.clusters, [1, 2]);
        assert_eq!(r.cluster_scores.len(), 2);
    }
}
