//! `contract`
//!
//! Domain types and the agent-facing output contract.
//!
//! ! Innermost layer (architecture/STANDARDS §2): zero I/O, zero sibling deps.
//! No database, no HTTP, no MCP — data plus the pure logic that derives one
//! shape from another.
//!
//! Field names here are the **wire format** an agent parses, fixed by
//! `OUTPUT_CONTRACT.md`. Renaming one is a breaking change, ✗ a refactor.

pub mod contract;

pub use contract::{
    Citation, ComponentScores, Confidence, ExactMatch, Locator, SearchResponse, SearchResult,
    Source, SummaryPayload,
};

/// Embedding width · Qwen3-Embedding-8B at full dimensionality.
///
/// ! Load-bearing. 4096 is why there is no global ANN index (pgvector cannot
/// index it) and therefore why routing exists at all (`EMBEDDING.md` §3).
pub const EMBEDDING_DIM: usize = 4096;

/// A stored chunk: body plus the provenance captured at ingest.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub id: String,
    pub domain_id: String,
    pub cluster_id: i32,
    pub body: String,
    pub source_title: String,
    pub source_url: String,
    pub locator_page: Option<i32>,
    pub locator_section: Option<String>,
    pub heading_path: Option<String>,
    pub identifier: Option<String>,
}

impl Chunk {
    /// Provenance for this chunk, always derived from stored fields.
    ///
    /// ! Never synthesizes. If ingest recorded no locator, the locator is empty
    /// rather than guessed (`LOOPHOLES.md` §8).
    #[must_use]
    pub fn source(&self) -> Source {
        Source {
            title: self.source_title.clone(),
            url: self.source_url.clone(),
            locator: Locator {
                page: self.locator_page,
                section: self.locator_section.clone(),
            },
        }
    }

    /// Bounded preview for `search_knowledge` · never the full body.
    ///
    /// Cuts on a char boundary and prefers the last word break, so a snippet
    /// does not end mid-token. Multibyte-safe.
    #[must_use]
    pub fn snippet(&self, max_chars: usize) -> String {
        let body = self.body.trim();
        if body.chars().count() <= max_chars {
            return body.to_owned();
        }
        let cut: String = body.chars().take(max_chars).collect();
        // Only honour a word break that keeps most of the budget.
        let keep = match cut.rfind(char::is_whitespace) {
            Some(i) if i >= max_chars.saturating_mul(3) / 4 => &cut[..i],
            _ => cut.as_str(),
        };
        format!("{}…", keep.trim_end())
    }
}

/// A knowledge base, as advertised by `list_domains`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Domain {
    pub id: String,
    pub description: String,
    pub row_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(body: &str) -> Chunk {
        Chunk {
            id: "reg::uu-28-2007::pasal-9::c3".into(),
            domain_id: "regulations".into(),
            cluster_id: 7,
            body: body.into(),
            source_title: "UU No. 28 Tahun 2007".into(),
            source_url: "https://peraturan.example/uu-28-2007.pdf".into(),
            locator_page: Some(14),
            locator_section: Some("Pasal 9 ayat (3)".into()),
            heading_path: None,
            identifier: Some("UU 28/2007".into()),
        }
    }

    #[test]
    fn short_bodies_are_returned_whole_without_an_ellipsis() {
        let c = chunk("Wajib Pajak");
        assert_eq!(c.snippet(64), "Wajib Pajak");
    }

    #[test]
    fn long_bodies_are_cut_on_a_word_break() {
        let c = chunk("Wajib Pajak yang terlambat menyampaikan laporan dikenai sanksi");
        let s = c.snippet(20);
        assert!(s.ends_with('…'), "{s}");
        assert!(s.starts_with("Wajib Pajak"), "{s}");
    }

    #[test]
    fn snippet_never_splits_a_multibyte_char() {
        // A cut at an arbitrary byte offset would panic or corrupt; chars() cannot.
        let c = chunk("pératuran émbedding ünicode ✓ 日本語のテキストもここにある");
        for n in 1..40 {
            let s = c.snippet(n);
            assert!(s.is_char_boundary(s.len()));
        }
    }

    #[test]
    fn provenance_comes_from_stored_fields_and_is_never_invented() {
        let source = chunk("x").source();
        assert_eq!(source.url, "https://peraturan.example/uu-28-2007.pdf");
        assert_eq!(source.locator.page, Some(14));
        assert_eq!(source.locator.section.as_deref(), Some("Pasal 9 ayat (3)"));
    }

    #[test]
    fn a_chunk_ingested_without_a_locator_reports_an_empty_one() {
        let mut c = chunk("x");
        c.locator_page = None;
        c.locator_section = None;
        let locator = c.source().locator;
        assert!(locator.page.is_none() && locator.section.is_none());
    }
}
