//! A revoked regulation must not outrank the one that revoked it.
//!
//! Ravel reads, from the verb beside each citation, which regulation revokes or replaces
//! which, and records it per document (`doc_status`). A search for "tarif retribusi" then
//! finds both the 2008 regulation and the 2015 one that ended it, and relevance alone can
//! put the dead one first, because it was written in the words the query uses.
//!
//! ! This is an **ordering rule between two members of the pool**, ✗ a prior. It never
//! multiplies a score by a number nobody measured, never removes a result, and does
//! nothing for a revoked regulation whose revoker did not make the pool: that one keeps its
//! place and is labelled instead. `docs/SCORING.md` §11 records why it is not a weight.
//!
//! Pure: documents and revokers in, an order out. No database, no I/O.

/// Reorder so no revoked document sits above a candidate from a document that revokes it.
/// Returns how many candidates were held back to make that true.
///
/// `doc` is a candidate's document id; `revoked_by(item, other)` says whether the document
/// `other` revokes or replaces the item's document.
///
/// ! A **stable topological order**, ✗ a series of moves: at each step the highest-ranked
/// candidate that is not waiting on a revoker present in the pool comes next. Everything
/// else keeps the order relevance and the factors gave it, and a revoked candidate lands as
/// soon as its last revoker has. Moving candidates one at a time instead leaves a chain
/// (1 revoked by 2, 2 by 3) with 1 back above 2 after 2 is moved below 3.
///
/// ! Terminates and never loops. A pair that revoke each other (one exists in the real
/// corpus, a bad link between a regulation and its own amendments) is treated as no
/// revocation at all: ranking one below the other would be arbitrary, and the data does not
/// say which is right. A longer cycle is broken at its highest-ranked member.
pub fn rank_below_revokers<'d, T>(
    items: &mut Vec<T>,
    doc: impl Fn(&T) -> &'d str,
    revoked_by: impl Fn(&T, &str) -> bool,
) -> usize {
    let n = items.len();
    // waiting_on[i]: positions of candidates that must come before i.
    let waiting_on: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let me = doc(&items[i]);
            if me.is_empty() {
                return Vec::new();
            }
            (0..n)
                .filter(|&j| {
                    let theirs = doc(&items[j]);
                    j != i
                        && !theirs.is_empty()
                        && theirs != me
                        && revoked_by(&items[i], theirs)
                        && !revoked_by(&items[j], me)
                })
                .collect()
        })
        .collect();

    let mut emitted = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut held_back = vec![false; n];
    while order.len() < n {
        let ready = (0..n)
            .find(|&i| !emitted[i] && waiting_on[i].iter().all(|&j| emitted[j]))
            // A cycle: nothing is ready, so release the highest-ranked one left.
            .or_else(|| (0..n).find(|&i| !emitted[i]));
        let Some(next) = ready else { break };
        for i in 0..next {
            if !emitted[i] {
                held_back[i] = true;
            }
        }
        emitted[next] = true;
        order.push(next);
    }

    let moved = held_back.iter().filter(|&&h| h).count();
    if moved > 0 {
        let mut taken: Vec<Option<T>> = items.drain(..).map(Some).collect();
        items.extend(order.into_iter().filter_map(|i| taken[i].take()));
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (document id, ids of the documents that revoke it)
    type C = (&'static str, Vec<String>);

    fn c(doc: &'static str, by: &[&str]) -> C {
        (doc, by.iter().map(|s| (*s).to_owned()).collect())
    }

    fn run(items: &mut Vec<C>) -> usize {
        rank_below_revokers(items, |x| x.0, |x, d| x.1.iter().any(|r| r == d))
    }

    fn order(items: &[C]) -> Vec<&'static str> {
        items.iter().map(|x| x.0).collect()
    }

    #[test]
    fn a_revoked_regulation_drops_below_the_one_that_revoked_it() {
        let mut v = vec![c("old", &["new"]), c("other", &[]), c("new", &[])];
        assert_eq!(run(&mut v), 1);
        assert_eq!(order(&v), ["other", "new", "old"]);
        // ! `other` was never waiting on anything, so it is not "held back".
    }

    #[test]
    fn nothing_moves_when_the_revoker_is_not_in_the_pool() {
        // ! The revoked regulation keeps its place: its revoker may simply not be in the
        // corpus, and a demotion on that evidence would be a guess.
        let mut v = vec![c("old", &["new"]), c("other", &[])];
        assert_eq!(run(&mut v), 0);
        assert_eq!(order(&v), ["old", "other"]);
    }

    #[test]
    fn nothing_moves_when_the_revoker_already_ranks_first() {
        let mut v = vec![c("new", &[]), c("old", &["new"])];
        assert_eq!(run(&mut v), 0);
        assert_eq!(order(&v), ["new", "old"]);
    }

    #[test]
    fn it_lands_directly_after_the_last_revoker() {
        let mut v = vec![
            c("old", &["a", "b"]),
            c("x", &[]),
            c("a", &[]),
            c("y", &[]),
            c("b", &[]),
            c("z", &[]),
        ];
        run(&mut v);
        assert_eq!(order(&v), ["x", "a", "y", "b", "old", "z"]);
    }

    #[test]
    fn two_chunks_of_one_regulation_both_follow_its_revoker() {
        let mut v = vec![c("old", &["new"]), c("old", &["new"]), c("new", &[])];
        run(&mut v);
        assert_eq!(order(&v), ["new", "old", "old"]);
    }

    #[test]
    fn a_pair_that_revoke_each_other_is_left_alone() {
        // ! The corpus holds one such pair: a bad link, not a fact.
        let mut v = vec![c("a", &["b"]), c("b", &["a"])];
        assert_eq!(run(&mut v), 0);
        assert_eq!(order(&v), ["a", "b"]);
    }

    #[test]
    fn a_chain_settles_oldest_last() {
        // 1 revoked by 2, 2 revoked by 3.
        let mut v = vec![c("1", &["2"]), c("2", &["3"]), c("3", &[])];
        assert_eq!(run(&mut v), 2);
        assert_eq!(order(&v), ["3", "2", "1"]);
    }

    #[test]
    fn a_candidate_without_a_document_id_is_never_moved() {
        let mut v = vec![c("", &["new"]), c("new", &[])];
        assert_eq!(run(&mut v), 0);
    }

    #[test]
    fn an_empty_pool_is_fine() {
        let mut v: Vec<C> = vec![];
        assert_eq!(run(&mut v), 0);
    }
}
