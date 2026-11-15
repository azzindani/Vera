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

/// Within each `key` group (one regulation), refill the group's slots with its operative
/// members first, then the rest, each in their existing order. Other groups and ungrouped
/// items keep their slots. Returns how many slots changed hands.
///
/// ! Why (`ROADMAP.md` Stage 3, `dev_tools/eval/article_sim.py`): of 91 right-law,
/// wrong-article misses, the same law's explanation outranked the answer in 37, its preamble
/// in 35. The structural ladder scores both low, but a bounded prior moves a score by
/// ~20% while fusion keeps the pool within ~13%, so a strong explanation still wins. This
/// keeps the law's place in the ranking and changes only which of its chunks fills it.
pub fn operative_first_within<T, K: PartialEq>(
    items: &mut [T],
    key: impl Fn(&T) -> Option<K>,
    operative: impl Fn(&T) -> bool,
) -> usize {
    let keys: Vec<Option<K>> = items.iter().map(&key).collect();
    let ops: Vec<bool> = items.iter().map(&operative).collect();
    let mut target: Vec<usize> = (0..items.len()).collect();
    let mut done = vec![false; items.len()];
    for i in 0..items.len() {
        let Some(k) = &keys[i] else { continue };
        if done[i] {
            continue;
        }
        let slots: Vec<usize> = (i..items.len())
            .filter(|&j| keys[j].as_ref() == Some(k))
            .collect();
        let order: Vec<usize> = slots
            .iter()
            .copied()
            .filter(|&j| ops[j])
            .chain(slots.iter().copied().filter(|&j| !ops[j]))
            .collect();
        for (&slot, &from) in slots.iter().zip(&order) {
            target[slot] = from;
            done[slot] = true;
        }
    }
    let moved = target.iter().enumerate().filter(|(s, f)| s != *f).count();
    if moved > 0 {
        permute(items, &target);
    }
    moved
}

/// Put `items[target[s]]` at slot `s`.
fn permute<T>(items: &mut [T], target: &[usize]) {
    let mut seen = vec![false; items.len()];
    for start in 0..items.len() {
        if seen[start] || target[start] == start {
            continue;
        }
        // Follow the cycle: slot s takes from target[s].
        let mut s = start;
        loop {
            seen[s] = true;
            let from = target[s];
            if from == start {
                break;
            }
            items.swap(s, from);
            s = from;
        }
    }
}

/// Place each anchor's `additions` directly above it, in order; an item already present is
/// moved up rather than duplicated. The list keeps its length, so what falls off the end is
/// the lowest-ranked. Returns how many items were placed.
///
/// ! For an explanation in the pool, the additions are its own article's operative text
/// (`Pipeline::pull_in_explained_articles`): the explanation says which clause the question
/// is about, and the clause is the answer. Measured offline: 6 wins, 0 losses alone.
pub fn place_above<T, I: PartialEq>(
    items: &mut Vec<T>,
    additions: Vec<(I, Vec<T>)>,
    id: impl Fn(&T) -> I,
) -> usize {
    if additions.is_empty() {
        return 0;
    }
    let n = items.len();
    let mut pending: Vec<(I, Vec<T>)> = additions;
    let mut out: Vec<T> = Vec::with_capacity(n);
    let mut placed = 0;
    for item in std::mem::take(items) {
        let me = id(&item);
        if let Some(at) = pending.iter().position(|(a, _)| *a == me) {
            let (_, adds) = pending.swap_remove(at);
            for add in adds {
                let aid = id(&add);
                if !out.iter().any(|o| id(o) == aid) {
                    out.push(add);
                    placed += 1;
                }
            }
        }
        if !out.iter().any(|o| id(o) == me) {
            out.push(item);
        }
    }
    out.truncate(n);
    *items = out;
    placed
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

    #[test]
    fn a_laws_operative_chunks_take_its_slots_first() {
        // (name, document, operative)
        let mut v = vec![
            ("expl", Some("d"), false),
            ("x", Some("e"), true),
            ("pre", Some("d"), false),
            ("art", Some("d"), true),
            ("none", None, false),
        ];
        let moved = operative_first_within(&mut v, |c| c.1, |c| c.2);
        let order: Vec<_> = v.iter().map(|c| c.0).collect();
        assert_eq!(order, ["art", "x", "expl", "pre", "none"]);
        assert_eq!(moved, 3);
    }

    #[test]
    fn a_law_with_no_operative_chunk_or_already_in_order_is_untouched() {
        let mut v = vec![
            ("a", Some("d"), true),
            ("b", Some("d"), false),
            ("c", Some("e"), false),
        ];
        assert_eq!(operative_first_within(&mut v, |c| c.1, |c| c.2), 0);
        assert_eq!(v.iter().map(|c| c.0).collect::<Vec<_>>(), ["a", "b", "c"]);
    }

    #[test]
    fn additions_go_above_their_anchor_and_existing_ones_move_up() {
        let mut v = vec!["a", "expl", "b", "art1"];
        let placed = place_above(&mut v, vec![("expl", vec!["art1", "art2"])], |s| *s);
        assert_eq!(v, ["a", "art1", "art2", "expl"]);
        assert_eq!(placed, 2);
    }

    #[test]
    fn nothing_to_place_changes_nothing() {
        let mut v = vec!["a", "b"];
        assert_eq!(
            place_above(&mut v, Vec::<(&str, Vec<&str>)>::new(), |s| *s),
            0
        );
        assert_eq!(v, ["a", "b"]);
    }
}
