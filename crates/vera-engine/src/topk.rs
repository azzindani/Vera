//! A bounded top-k collector.
//!
//! ! The leaf scan streams millions of rows through this, so it must be O(k) in
//! memory and never hold the input. Sorting the whole cluster and truncating
//! would be simpler and would also defeat the entire OOM argument
//! (`ARCHITECTURE.md` §4).
//!
//! A binary min-heap keyed on score: the cheapest element is always at the top,
//! so admitting a candidate is one comparison and eviction is O(log k).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One scored candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub id: String,
    pub score: f32,
}

/// Min-heap ordering wrapper · `BinaryHeap` is a max-heap, so the comparison is
/// reversed to keep the *worst* candidate at the top for eviction.
#[derive(Debug)]
struct Worst(Scored);

impl PartialEq for Worst {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Worst {}

impl Ord for Worst {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: greater = worse score. NaN sorts as worst so a degenerate
        // score is evicted first rather than poisoning the ordering.
        other
            .0
            .score
            .partial_cmp(&self.0.score)
            .unwrap_or_else(|| match (other.0.score.is_nan(), self.0.score.is_nan()) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => Ordering::Equal,
            })
    }
}

impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Keeps the best `k` candidates seen, in O(k) memory.
#[derive(Debug)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Worst>,
}

impl TopK {
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Offer a candidate. Cheap to reject: one peek when the heap is full.
    ///
    /// The `id` is only allocated when the candidate is actually admitted, so a
    /// scan that rejects most rows does almost no allocation.
    ///
    /// ! A NaN score is refused outright rather than admitted and later evicted.
    /// Admitting one poisons the eviction test: `score > worst` is *false* for
    /// every finite score once `worst` is NaN, so a single corrupt row would
    /// wedge the heap and silently reject every genuinely better candidate
    /// after it. Keeping NaN out entirely also guarantees `peek()` is always
    /// comparable, which is what makes the plain `>` below correct.
    pub fn offer(&mut self, id: &str, score: f32) {
        if self.k == 0 || score.is_nan() {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(Worst(Scored {
                id: id.to_owned(),
                score,
            }));
            return;
        }
        // ! Compare before allocating the id · this is the hot path.
        if let Some(worst) = self.heap.peek()
            && score > worst.0.score
        {
            self.heap.pop();
            self.heap.push(Worst(Scored {
                id: id.to_owned(),
                score,
            }));
        }
    }

    /// Drain into a best-first ranked list.
    #[must_use]
    pub fn into_ranked(self) -> Vec<Scored> {
        let mut out: Vec<Scored> = self.heap.into_iter().map(|w| w.0).collect();
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        out
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_keeps_the_best_k_and_orders_them_best_first() {
        let mut t = TopK::new(3);
        for (id, score) in [("a", 0.1), ("b", 0.9), ("c", 0.5), ("d", 0.7), ("e", 0.2)] {
            t.offer(id, score);
        }
        let ranked = t.into_ranked();
        let ids: Vec<_> = ranked.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["b", "d", "c"]);
    }

    #[test]
    fn memory_stays_bounded_no_matter_how_many_rows_stream_through() {
        // ! The property the leaf scan depends on. 100K offers, capacity k.
        let mut t = TopK::new(50);
        for i in 0..100_000 {
            t.offer(&format!("row-{i}"), (i % 977) as f32);
        }
        assert_eq!(t.len(), 50);
        assert!(t.heap.capacity() < 200, "heap grew to {}", t.heap.capacity());
    }

    #[test]
    fn fewer_candidates_than_k_are_all_kept() {
        let mut t = TopK::new(10);
        t.offer("a", 1.0);
        t.offer("b", 2.0);
        assert_eq!(t.into_ranked().len(), 2);
    }

    #[test]
    fn a_zero_width_collector_keeps_nothing_rather_than_panicking() {
        let mut t = TopK::new(0);
        t.offer("a", 1.0);
        assert!(t.is_empty());
    }

    #[test]
    fn a_nan_score_is_refused_rather_than_corrupting_the_ranking() {
        // ! Regression, and the reason NaN is rejected at the door: admitted
        // first, a NaN sits at the top of the min-heap, and `0.9 > NaN` is
        // false — so every better candidate that follows is silently dropped.
        // The heap wedges on one corrupt row.
        let mut t = TopK::new(2);
        t.offer("nan", f32::NAN);
        t.offer("good", 0.5);
        t.offer("better", 0.9);
        let ids: Vec<String> = t.into_ranked().into_iter().map(|s| s.id).collect();
        assert_eq!(ids, ["better", "good"]);
    }

    #[test]
    fn a_nan_arriving_after_the_heap_is_full_changes_nothing() {
        let mut t = TopK::new(2);
        t.offer("a", 0.5);
        t.offer("b", 0.9);
        t.offer("nan", f32::NAN);
        let ids: Vec<String> = t.into_ranked().into_iter().map(|s| s.id).collect();
        assert_eq!(ids, ["b", "a"]);
    }

    #[test]
    fn an_all_nan_scan_yields_nothing_rather_than_garbage() {
        let mut t = TopK::new(5);
        for i in 0..10 {
            t.offer(&format!("r{i}"), f32::NAN);
        }
        assert!(t.is_empty());
    }

    #[test]
    fn ties_are_kept_rather_than_collapsed() {
        let mut t = TopK::new(3);
        for id in ["a", "b", "c", "d"] {
            t.offer(id, 0.5);
        }
        assert_eq!(t.into_ranked().len(), 3);
    }
}
