//! Several domains, one engine (`docs/operations/SCALING.md` §4a).
//!
//! ! A domain is `(database, schema)`. **Separated** is a database per corpus, for
//! production, where corpora live and are replaced apart; **unified** is one database
//! with a schema per corpus, for one box. One list, `DOMAINS_FILE`, configures both, and
//! a mix. Without it the engine serves the one database `DATABASE_URL` names, exactly as
//! before: one domain, no selection.
//!
//! ! The agent still never names a domain (`CLAUDE.md` §7.13). Each domain's own gate
//! judges the query, and the best domain that would serve it answers.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Deserialize;

use crate::pipeline::Pipeline;

/// One `DOMAINS_FILE` entry.
///
/// ! It names the variable that holds the connection string, ✗ the string: a DSN carries
/// a password (`CLAUDE.md` §7.14), and this file is meant to live in git.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Entry {
    /// What `list_domains` calls it: `id_reg`, `ca_laws`.
    pub(crate) name: String,
    /// The schema the corpus lives in, in a database that holds several. Absent: `public`.
    #[serde(default)]
    pub(crate) schema: Option<String>,
    /// The corpus's BM25 vocabulary, the file Ravel wrote beside its bundle.
    pub(crate) vocab: PathBuf,
    /// The algorithms file this corpus is ranked by. Absent: `ALGORITHMS_PATH`'s.
    #[serde(default)]
    pub(crate) algorithms: Option<String>,
    /// The environment variable holding the connection string.
    #[serde(default = "default_dsn_env")]
    pub(crate) dsn_env: String,
}

fn default_dsn_env() -> String {
    "DATABASE_URL".to_owned()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    /// The file's own notes, ignored.
    #[serde(default, rename = "_doc")]
    _doc: Option<serde_json::Value>,
    domains: Vec<Entry>,
}

/// Parse and check a `DOMAINS_FILE`.
///
/// # Errors
/// Malformed JSON, an unknown key, no domain, a domain without a name, or a name twice.
pub(crate) fn parse(body: &str) -> Result<Vec<Entry>, String> {
    let file: File = serde_json::from_str(body).map_err(|e| e.to_string())?;
    if file.domains.is_empty() {
        return Err("no domains listed".to_owned());
    }
    let mut seen = BTreeSet::new();
    for e in &file.domains {
        if e.name.trim().is_empty() {
            return Err("a domain with no name".to_owned());
        }
        if !seen.insert(e.name.as_str()) {
            return Err(format!("domain {:?} listed twice", e.name));
        }
    }
    Ok(file.domains)
}

/// A domain being served: its name and the pipeline over its corpus.
pub(crate) struct Domain {
    pub(crate) name: String,
    pub(crate) pipe: Arc<Pipeline>,
}

/// One domain's gate on one query.
pub(crate) struct Verdict<'a> {
    pub(crate) domain: &'a Domain,
    pub(crate) serves: bool,
    pub(crate) similarity: f32,
    /// The gate could not be asked (the embedder or the database failed).
    pub(crate) failed: bool,
}

/// Every domain's gate on `query`, best first: the domains that would serve, then by
/// centroid similarity.
///
/// ! One after another, ✗ concurrently: each asks the embedder for the same text, and
/// after the first the embedder's query cache answers (`ROADMAP.md` Stage 4,
/// `QUERY_CACHE`), so what is left per domain is its centroids and one lexical probe.
pub(crate) async fn gates<'a>(domains: &'a [Domain], query: &str) -> Vec<Verdict<'a>> {
    let mut out = Vec::with_capacity(domains.len());
    for domain in domains {
        let verdict = match domain.pipe.explain(query).await {
            Ok(r) => Verdict {
                domain,
                serves: !r.would_refuse,
                similarity: r.centroid_similarity,
                failed: false,
            },
            Err(_) => Verdict {
                domain,
                serves: false,
                similarity: f32::NEG_INFINITY,
                failed: true,
            },
        };
        out.push(verdict);
    }
    rank(&mut out);
    out
}

fn rank(verdicts: &mut [Verdict<'_>]) {
    verdicts.sort_by(|a, b| {
        b.serves
            .cmp(&a.serves)
            .then(b.similarity.total_cmp(&a.similarity))
    });
}

/// The progress line that says which domain answered and why the others did not.
pub(crate) fn describe(verdicts: &[Verdict<'_>]) -> String {
    let parts: Vec<String> = verdicts
        .iter()
        .map(|v| {
            if v.failed {
                format!("{} gate failed", v.domain.name)
            } else {
                let verdict = if v.serves { "serves" } else { "refuses" };
                format!("{} {:.3} {verdict}", v.domain.name, v.similarity)
            }
        })
        .collect();
    format!("domains \u{b7} {}", parts.join(" \u{b7} "))
}

/// The algorithms every domain offers, merged by name: what `tools/list` advertises.
///
/// ! The first domain's definition of a name wins, and the domain a search lands in
/// validates the request again, so a name only one corpus can serve is refused there
/// with that corpus's reason, ✗ silently replaced.
pub(crate) fn merged_registry(domains: &[Domain]) -> contract::Registry {
    let mut merged = domains[0].pipe.algorithms().clone();
    for d in &domains[1..] {
        for (name, a) in &d.pipe.algorithms().algorithms {
            merged
                .algorithms
                .entry(name.clone())
                .or_insert_with(|| a.clone());
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_lists_domains_by_the_variable_that_holds_their_dsn() {
        let d = parse(
            r#"{"_doc": "x", "domains": [
                {"name": "id_reg", "schema": "id_reg", "vocab": "/vocab/id_reg.bm25.json",
                 "algorithms": "/cfg/algorithms.json"},
                {"name": "ca_laws", "vocab": "/vocab/ca.bm25.json", "dsn_env": "CA_DATABASE_URL"}
            ]}"#,
        )
        .expect("parses");
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].dsn_env, "DATABASE_URL");
        assert_eq!(d[0].schema.as_deref(), Some("id_reg"));
        assert_eq!(d[1].dsn_env, "CA_DATABASE_URL");
        assert!(d[1].schema.is_none() && d[1].algorithms.is_none());
    }

    #[test]
    fn a_connection_string_in_the_file_is_refused() {
        // ! `deny_unknown_fields`: a `dsn` key would put a password in git.
        let e = parse(r#"{"domains": [{"name": "a", "vocab": "v", "dsn": "postgres://u:p@h/d"}]}"#)
            .expect_err("refused");
        assert!(e.contains("dsn"), "{e}");
    }

    #[test]
    fn no_domain_and_a_name_twice_are_refused() {
        assert!(parse(r#"{"domains": []}"#).is_err());
        let e = parse(r#"{"domains": [{"name": "a", "vocab": "v"}, {"name": "a", "vocab": "w"}]}"#)
            .expect_err("refused");
        assert!(e.contains("twice"), "{e}");
        assert!(parse(r#"{"domains": [{"name": " ", "vocab": "v"}]}"#).is_err());
    }
}
