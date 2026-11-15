//! An explanation must not outrank the article it explains.
//!
//! Ravel stores the *penjelasan* of `Pasal 6` under `article = 'Pasal 6'`, in chapter
//! `PENJELASAN - ...`. Both are the same article of the same document by their labels, and
//! the explanation often wins on text: it repeats the question's terms ("Yang dimaksud
//! dengan ..."). Cited as `Pasal 6`, it is the wrong clause with the right number.
//!
//! ! An **ordering rule between two members of the pool**, like currency, ✗ a prior. The
//! structural ladder already scores annex text lower, but a bounded prior cannot keep a
//! strong explanation below a weak article. This moves nothing out of the pool and needs no
//! fetch: it only acts when both texts were retrieved.
//!
//! Pure: keys and structural scores in, an order out. No database, no I/O.

/// Reorder so no candidate sits above a candidate of the same `key` (same document, same
/// article) that scores higher on the structural ladder. The higher one moves to just above
/// the first lower one; everything else keeps its order. Returns how many moved.
///
/// `key` is `None` for a candidate with no article label, which is never regrouped.
pub fn operative_above_annex<T, K: PartialEq>(
    items: &mut Vec<T>,
    key: impl Fn(&T) -> Option<K>,
    structural: impl Fn(&T) -> f32,
) -> usize {
    let n = items.len();
    let keys: Vec<Option<K>> = items.iter().map(&key).collect();
    let scores: Vec<f32> = items.iter().map(&structural).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut placed = vec![false; n];
    let mut moved = 0;
    for i in 0..n {
        if placed[i] {
            continue;
        }
        if let Some(k) = &keys[i] {
            // Later candidates of the same article that the ladder ranks above this one.
            for j in (i + 1)..n {
                if !placed[j] && keys[j].as_ref() == Some(k) && scores[j] > scores[i] {
                    order.push(j);
                    placed[j] = true;
                    moved += 1;
                }
            }
        }
        order.push(i);
        placed[i] = true;
    }
    if moved > 0 {
        let mut slots: Vec<Option<T>> = items.drain(..).map(Some).collect();
        items.extend(order.into_iter().filter_map(|i| slots[i].take()));
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    // (name, document, article, structural)
    type C = (&'static str, &'static str, Option<&'static str>, f32);

    fn run(mut v: Vec<C>) -> (Vec<&'static str>, usize) {
        let moved = operative_above_annex(&mut v, |c| c.2.map(|a| (c.1, a)), |c| c.3);
        (v.into_iter().map(|c| c.0).collect(), moved)
    }

    #[test]
    fn the_article_moves_above_its_own_explanation() {
        let (order, moved) = run(vec![
            ("expl6", "d", Some("Pasal 6"), 0.2),
            ("other", "e", Some("Pasal 1"), 1.0),
            ("art6", "d", Some("Pasal 6"), 1.0),
        ]);
        assert_eq!(order, ["art6", "expl6", "other"]);
        assert_eq!(moved, 1);
    }

    #[test]
    fn another_documents_article_is_not_pulled_up() {
        let (order, moved) = run(vec![
            ("expl6", "d", Some("Pasal 6"), 0.2),
            ("art6_elsewhere", "e", Some("Pasal 6"), 1.0),
        ]);
        assert_eq!(order, ["expl6", "art6_elsewhere"]);
        assert_eq!(moved, 0);
    }

    #[test]
    fn an_article_already_above_stays_where_it_is() {
        let (order, moved) = run(vec![
            ("art6", "d", Some("Pasal 6"), 1.0),
            ("x", "e", None, 0.6),
            ("expl6", "d", Some("Pasal 6"), 0.2),
        ]);
        assert_eq!(order, ["art6", "x", "expl6"]);
        assert_eq!(moved, 0);
    }

    #[test]
    fn unlabelled_candidates_are_never_regrouped() {
        let (order, moved) = run(vec![("a", "d", None, 0.2), ("b", "d", None, 1.0)]);
        assert_eq!(order, ["a", "b"]);
        assert_eq!(moved, 0);
    }
}
