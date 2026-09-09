//! A flat row-major f32 matrix.
//!
//! ! Flat, ✗ `Vec<Vec<f32>>`. At 750K × 1024 the nested form costs an extra
//! 24 bytes of `Vec` header per row (~18 MB) and, far worse, scatters rows
//! across the heap — k-means is a bandwidth-bound scan over every row on every
//! iteration, so locality is the difference between minutes and tens of minutes.

/// Row-major `rows × dim` matrix of f32.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f32>,
    dim: usize,
}

impl Matrix {
    /// Empty matrix with a declared width.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
        }
    }

    /// Pre-allocate for `rows` rows · one allocation for the whole corpus.
    #[must_use]
    pub fn with_capacity(dim: usize, rows: usize) -> Self {
        Self {
            data: Vec::with_capacity(dim * rows),
            dim,
        }
    }

    /// Append a row.
    ///
    /// # Panics
    /// If `row.len() != dim` — a width mismatch here would silently shear every
    /// subsequent row by one position, which is unrecoverable and invisible.
    pub fn push(&mut self, row: &[f32]) {
        assert_eq!(
            row.len(),
            self.dim,
            "row width {} does not match matrix width {}",
            row.len(),
            self.dim
        );
        self.data.extend_from_slice(row);
    }

    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrow row `i`.
    ///
    /// # Panics
    /// If `i` is out of range.
    #[must_use]
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Overwrite row `i` in place.
    ///
    /// ! Exists for split-on-size, which **replaces** an oversized centroid with
    /// the first half and appends the second (`crate::split`). Reusing the slot
    /// is what keeps every untouched row's assignment valid, so a split costs
    /// O(members) rather than a corpus-wide renumbering.
    ///
    /// # Panics
    /// If `i` is out of range, or `row.len() != dim` — same reasoning as
    /// [`push`](Self::push).
    pub fn set_row(&mut self, i: usize, row: &[f32]) {
        assert_eq!(
            row.len(),
            self.dim,
            "row width {} does not match matrix width {}",
            row.len(),
            self.dim
        );
        let start = i * self.dim;
        self.data[start..start + self.dim].copy_from_slice(row);
    }

    /// Iterate rows.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.dim)
    }

    /// Bytes held · the term in the offline RAM budget.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_go_in_and_come_back_intact() {
        let mut m = Matrix::new(3);
        m.push(&[1.0, 2.0, 3.0]);
        m.push(&[4.0, 5.0, 6.0]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.row(0), [1.0, 2.0, 3.0]);
        assert_eq!(m.row(1), [4.0, 5.0, 6.0]);
        assert_eq!(m.iter_rows().count(), 2);
    }

    #[test]
    #[should_panic(expected = "does not match matrix width")]
    fn a_wrong_width_row_panics_rather_than_shearing_the_matrix() {
        // ! Silent acceptance would offset every later row by one float, and
        // nothing downstream could detect it.
        let mut m = Matrix::new(3);
        m.push(&[1.0, 2.0]);
    }

    #[test]
    fn an_empty_matrix_reports_zero_rows() {
        let m = Matrix::new(128);
        assert_eq!(m.rows(), 0);
        assert!(m.is_empty());
        assert_eq!(m.iter_rows().count(), 0);
    }

    #[test]
    fn capacity_is_taken_once_for_the_whole_corpus() {
        let mut m = Matrix::with_capacity(4, 1_000);
        let before = m.data.capacity();
        for _ in 0..1_000 {
            m.push(&[0.0; 4]);
        }
        assert_eq!(m.data.capacity(), before, "matrix reallocated mid-build");
        assert_eq!(m.bytes(), 1_000 * 4 * 4);
    }
}
