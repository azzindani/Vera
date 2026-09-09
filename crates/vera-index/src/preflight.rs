//! Free-space preflight · `LOOPHOLES.md` §10.
//!
//! ! Runs **before** the first write, never during. A bulk load that fills the
//! disk halfway through leaves a corpus that opens, reports a row count, answers
//! queries and is missing the second half — the failure class this whole
//! document set is organised around, because nothing about it looks like an
//! error. At the target scale (~1.3 TB, `HARDWARE.md` §3) a rebuild is hours,
//! so failing in the first second is worth a syscall.
//!
//! ! An **estimate**, and the estimate is deliberately conservative. Predicting
//! SQLite's exact on-disk size is not possible — page overhead, the FTS5 index,
//! the WAL and free-list reuse all move it — so the check multiplies the part
//! that is knowable (vectors and bodies, which dominate) by a headroom factor
//! and refuses when the margin is thin. Refusing a build that would just barely
//! have fit costs a flag; discovering the shortfall at 80% costs the run.

use std::path::Path;

/// What a build is about to need, and what the filesystem has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceCheck {
    pub required_bytes: u64,
    pub available_bytes: u64,
}

impl SpaceCheck {
    #[must_use]
    pub const fn fits(&self) -> bool {
        self.available_bytes >= self.required_bytes
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PreflightError {
    #[error(
        "not enough free space: the build needs about {required_gb:.1} GB and \
         {available_gb:.1} GB is free at {path} · refusing to start rather than \
         fail partway and leave a corpus that opens, answers queries and is \
         missing rows"
    )]
    Insufficient {
        path: String,
        required_gb: f64,
        available_gb: f64,
    },

    #[error("could not determine free space at {path}: {reason}")]
    Unknown { path: String, reason: String },
}

/// Bytes a corpus of this shape is expected to occupy.
///
/// ! Counts the vectors and the bodies twice over, once for the table and once
/// as headroom for the FTS5 index, the WAL and page overhead. FTS5 here is an
/// **external-content** table, so it stores an index over the bodies rather than
/// a second copy — but the index is not free and the WAL can transiently reach
/// the size of the transaction, which for a bulk load is the whole corpus.
#[must_use]
pub fn estimated_bytes(rows: usize, dim: usize, mean_body_bytes: usize) -> u64 {
    let vectors = rows as u64 * dim as u64 * 4;
    let bodies = rows as u64 * mean_body_bytes as u64;
    // Provenance columns: url, title, locator, identifier, hash.
    let metadata = rows as u64 * 256;
    let base = vectors + bodies + metadata;
    // ×2 headroom for the FTS5 index, WAL, and page overhead.
    base * 2
}

/// Refuse a build that will not fit.
///
/// `path` is the file the corpus is being written to; its **parent directory**
/// is what gets measured, since the file itself does not exist yet.
///
/// # Errors
/// [`PreflightError::Insufficient`] when the estimate exceeds free space, or
/// [`PreflightError::Unknown`] when free space cannot be read at all — which is
/// also a refusal, because proceeding would mean claiming a check ran that did
/// not.
pub fn require_free_space(path: &Path, required_bytes: u64) -> Result<SpaceCheck, PreflightError> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let available = available_bytes(dir).ok_or_else(|| PreflightError::Unknown {
        path: dir.display().to_string(),
        reason: "statvfs is unavailable on this platform or the path is unreadable".to_owned(),
    })?;

    let check = SpaceCheck {
        required_bytes,
        available_bytes: available,
    };
    if check.fits() {
        return Ok(check);
    }
    #[allow(clippy::cast_precision_loss)]
    Err(PreflightError::Insufficient {
        path: dir.display().to_string(),
        required_gb: required_bytes as f64 / 1e9,
        available_gb: available as f64 / 1e9,
    })
}

/// Free bytes on the filesystem holding `dir`.
///
/// ! Shelled out to `df`, ✗ `libc::statvfs`. The workspace forbids `unsafe`
/// (`Cargo.toml`), and `statvfs` is an FFI call — pulling in a crate that wraps
/// it would move the `unsafe` rather than remove it, for a check that runs once
/// per build and is allowed to be slow. `df -kP` is POSIX-specified output.
///
/// Returns `None` rather than a default: an unknown free-space figure must not
/// be mistaken for a large one.
fn available_bytes(dir: &Path) -> Option<u64> {
    let out = std::process::Command::new("df")
        .arg("-kP")
        .arg(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // POSIX `df -P` guarantees one line per filesystem after the header, with
    // available 1K-blocks in the fourth field.
    let line = text.lines().nth(1)?;
    let available_kb: u64 = line.split_whitespace().nth(3)?.parse().ok()?;
    Some(available_kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_estimate_is_dominated_by_the_vectors_at_production_width() {
        // 100M × 4096 f32 · HARDWARE.md §3 puts the vectors at ~1.6 TB in f32.
        let with_bodies = estimated_bytes(1_000_000, 4096, 1_000);
        let vectors_only = estimated_bytes(1_000_000, 4096, 0);
        assert!(
            with_bodies < vectors_only * 2,
            "bodies must not dominate at 4096 dims"
        );
        assert!(vectors_only >= 1_000_000 * 4096 * 4, "vectors are counted");
    }

    #[test]
    fn the_estimate_leaves_headroom_rather_than_predicting_exactly() {
        // ! Deliberately over-estimates. SQLite's real footprint depends on page
        // overhead, the FTS5 index and the WAL, none of which are predictable;
        // refusing a build that would just barely have fit costs a flag, and
        // discovering the shortfall at 80% costs the run.
        let raw = 1_000u64 * 1024 * 4;
        assert!(estimated_bytes(1_000, 1024, 0) > raw, "no headroom was added");
    }

    #[test]
    fn a_build_that_fits_is_permitted() {
        let dir = tempfile::tempdir().unwrap();
        let check = require_free_space(&dir.path().join("corpus.db"), 1024).unwrap();
        assert!(check.fits());
        assert!(check.available_bytes > 0, "free space was actually read");
    }

    #[test]
    fn a_build_that_cannot_fit_is_refused_before_any_write() {
        // ! The point of the check. Half a corpus is worse than none: it opens,
        // reports a row count, answers queries, and is silently incomplete.
        let dir = tempfile::tempdir().unwrap();
        let err = require_free_space(&dir.path().join("corpus.db"), u64::MAX).unwrap_err();
        assert!(matches!(err, PreflightError::Insufficient { .. }), "{err}");
        assert!(err.to_string().contains("missing rows"), "{err}");
    }

    #[test]
    fn the_check_measures_the_parent_directory_not_the_absent_file() {
        // The corpus file does not exist yet, so statting it would fail.
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist-yet.db");
        assert!(!missing.exists());
        assert!(require_free_space(&missing, 1).is_ok());
    }

    #[test]
    fn a_bare_filename_falls_back_to_the_working_directory() {
        assert!(require_free_space(Path::new("corpus.db"), 1).is_ok());
    }
}
