//! `vera-ingest` · turn an existing embedded corpus into a Vera corpus.
//!
//! The corpus under test already exists: rows and vectors sitting in someone
//! else's SQLite schema. This tool reads that schema — column names and vector
//! encoding are all flags — clusters the vectors, and writes a Vera corpus.
//!
//! ! Column mapping is configuration, ✗ a fixed schema. The source is not ours
//! and will not match, and a hardcoded expectation would mean editing Rust to
//! ingest each new dataset.
//!
//! Two commands: `inspect` reads a source and reports what is in it (run this
//! first — it prints the exact `import` flags it inferred), `import` builds.

use std::collections::HashMap;

use rusqlite::{Connection, OpenFlags, types::ValueRef};
use vera_core::EmbeddingSpace;
use vera_index::{IngestRow, KMeansConfig, Matrix, build_corpus};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => cmd_inspect(&args[1..]),
        Some("import") => cmd_import(&args[1..]),
        _ => {
            eprintln!(
                "usage:
  vera-ingest inspect --source <db> [--table <name>]
  vera-ingest import  --source <db> --table <name> --out <db>
                      --id-col <c> --body-col <c> --vector-col <c>
                      [--vector-format f32le|f64le|json]  (default f32le)
                      [--model <id>] [--dim N] [--no-normalize]
                      [--title-col <c>] [--url-col <c>] [--page-col <c>]
                      [--section-col <c>] [--heading-col <c>] [--identifier-col <c>]
                      [--domain <id>] [--description <text>]
                      [--per-cluster N] [--iters N] [--limit N]

  inspect  report tables, columns, row counts and the detected vector encoding
  import   cluster the vectors and write a Vera corpus

Run `inspect` first: it prints the import flags it inferred from the source."
            );
            std::process::exit(2);
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn required(args: &[String], name: &str) -> Result<String, String> {
    flag(args, name).ok_or_else(|| format!("missing required flag {name}"))
}

fn parsed<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    match flag(args, name) {
        None => Ok(default),
        Some(v) => v.parse().map_err(|e| format!("{name}: {e}")),
    }
}

// ── vector decoding ──────────────────────────────────────────────────────────

/// How the source stores a vector.
///
/// ! Three encodings because all three are common in the wild and guessing
/// wrong is silent: a float64 blob read as float32 yields twice the dimensions
/// and pure garbage that still clusters and still ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorFormat {
    F32Le,
    F64Le,
    Json,
}

impl std::str::FromStr for VectorFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32le" | "float32" => Ok(Self::F32Le),
            "f64le" | "float64" => Ok(Self::F64Le),
            "json" => Ok(Self::Json),
            other => Err(format!("unknown vector format '{other}' · use f32le, f64le or json")),
        }
    }
}

fn decode(value: &ValueRef<'_>, format: VectorFormat) -> Result<Vec<f32>, String> {
    match format {
        VectorFormat::F32Le => {
            let b = value.as_blob().map_err(|e| e.to_string())?;
            if b.len() % 4 != 0 {
                return Err(format!("{} bytes is not a whole number of f32", b.len()));
            }
            Ok(b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        VectorFormat::F64Le => {
            let b = value.as_blob().map_err(|e| e.to_string())?;
            if b.len() % 8 != 0 {
                return Err(format!("{} bytes is not a whole number of f64", b.len()));
            }
            #[allow(clippy::cast_possible_truncation)]
            Ok(b.chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect())
        }
        VectorFormat::Json => {
            let s = value.as_str().map_err(|e| e.to_string())?;
            serde_json::from_str::<Vec<f32>>(s).map_err(|e| e.to_string())
        }
    }
}

/// Guess an encoding from one sample value and a plausible dimension.
///
/// Reported by `inspect` as a suggestion, never applied silently by `import`.
fn guess_format(value: &ValueRef<'_>) -> Option<(VectorFormat, usize)> {
    if let Ok(s) = value.as_str()
        && let Ok(v) = serde_json::from_str::<Vec<f32>>(s)
    {
        return Some((VectorFormat::Json, v.len()));
    }
    let b = value.as_blob().ok()?;
    // ! Prefer f32 when both divide evenly, and say so: f32 is overwhelmingly
    // the convention for stored embeddings, and `inspect` prints the dimension
    // so a wrong guess is visible before anything is built.
    for (fmt, width) in [(VectorFormat::F32Le, 4usize), (VectorFormat::F64Le, 8)] {
        if b.len() % width == 0 {
            let dim = b.len() / width;
            if (64..=8192).contains(&dim) {
                return Some((fmt, dim));
            }
        }
    }
    None
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ── inspect ──────────────────────────────────────────────────────────────────

fn cmd_inspect(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let source = required(args, "--source")?;
    let conn = Connection::open_with_flags(&source, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    let tables: Vec<String> = {
        let mut stmt = conn.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' \
             ORDER BY name",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect::<Result<_, _>>()?
    };

    println!("source: {source}");
    println!("tables: {}", tables.join(", "));

    let wanted = flag(args, "--table");
    for table in &tables {
        if wanted.as_ref().is_some_and(|w| w != table) {
            continue;
        }
        let count: i64 =
            conn.query_row(&format!("SELECT COUNT(*) FROM \"{table}\""), [], |r| r.get(0))?;
        println!();
        println!("── {table} · {count} rows");

        let mut stmt = conn.prepare(&format!("SELECT * FROM \"{table}\" LIMIT 1"))?;
        let names: Vec<String> = stmt.column_names().iter().map(|s| (*s).to_owned()).collect();
        let mut rows = stmt.query([])?;
        let Some(row) = rows.next()? else {
            println!("   (empty)");
            continue;
        };

        let mut vector_candidates: Vec<(String, VectorFormat, usize)> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let v = row.get_ref(i)?;
            let described = match v {
                ValueRef::Null => "NULL".to_owned(),
                ValueRef::Integer(n) => format!("INTEGER  e.g. {n}"),
                ValueRef::Real(f) => format!("REAL     e.g. {f}"),
                ValueRef::Text(t) => {
                    let s = String::from_utf8_lossy(t);
                    let preview: String = s.chars().take(48).collect();
                    format!("TEXT({})  e.g. {preview}…", s.chars().count())
                }
                ValueRef::Blob(b) => format!("BLOB({} bytes)", b.len()),
            };
            print!("   {name:<24} {described}");
            if let Some((fmt, dim)) = guess_format(&v) {
                print!("   → looks like {fmt:?} × {dim} dims");
                vector_candidates.push((name.clone(), fmt, dim));
            }
            println!();
        }

        if let Some((col, fmt, dim)) = vector_candidates.first() {
            let fmt_flag = match fmt {
                VectorFormat::F32Le => "f32le",
                VectorFormat::F64Le => "f64le",
                VectorFormat::Json => "json",
            };
            println!();
            println!("   inferred import flags:");
            println!(
                "     vera-ingest import --source {source} --table {table} --out corpus.db \\\n\
                 \x20      --id-col <id> --body-col <text> \\\n\
                 \x20      --vector-col {col} --vector-format {fmt_flag} --dim {dim}"
            );
            println!(
                "   ! verify the dimension against the model that produced it · \
                 a f64 blob read as f32 gives {} dims of garbage that still clusters",
                dim * 2
            );
        }
    }
    Ok(())
}

// ── import ───────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn cmd_import(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let source = required(args, "--source")?;
    let table = required(args, "--table")?;
    let out = required(args, "--out")?;
    let id_col = required(args, "--id-col")?;
    let body_col = required(args, "--body-col")?;
    let vector_col = required(args, "--vector-col")?;
    let format: VectorFormat = parsed(args, "--vector-format", VectorFormat::F32Le)?;
    let should_normalize = !has(args, "--no-normalize");
    let limit: Option<usize> = flag(args, "--limit").map(|s| s.parse()).transpose()?;

    let optional: HashMap<&str, Option<String>> = [
        ("title", flag(args, "--title-col")),
        ("url", flag(args, "--url-col")),
        ("page", flag(args, "--page-col")),
        ("section", flag(args, "--section-col")),
        ("heading", flag(args, "--heading-col")),
        ("identifier", flag(args, "--identifier-col")),
    ]
    .into_iter()
    .collect();

    let conn = Connection::open_with_flags(&source, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    // Build the projection: required columns first, then whichever optional
    // ones the caller mapped.
    let mut columns = vec![id_col.clone(), body_col.clone(), vector_col.clone()];
    let mut slot: HashMap<&str, usize> = HashMap::new();
    for key in ["title", "url", "page", "section", "heading", "identifier"] {
        if let Some(Some(col)) = optional.get(key) {
            slot.insert(key, columns.len());
            columns.push(col.clone());
        }
    }
    let projection = columns
        .iter()
        .map(|c| format!("\"{c}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = match limit {
        Some(n) => format!("SELECT {projection} FROM \"{table}\" LIMIT {n}"),
        None => format!("SELECT {projection} FROM \"{table}\""),
    };

    eprintln!("reading {source}::{table}…");
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;

    let mut ingest: Vec<IngestRow> = Vec::new();
    let mut vectors: Option<Matrix> = None;
    let mut dim = 0usize;
    let mut skipped = 0usize;

    while let Some(row) = rows.next()? {
        let vector_ref = row.get_ref(2)?;
        let mut vector = match decode(&vector_ref, format) {
            Ok(v) => v,
            Err(e) => {
                // ! Skip and count, ✗ abort. Real corpora carry a few broken
                // rows, and losing an entire multi-hour import to one of them
                // is worse than losing the row — but a silent skip is worse
                // still, so the total is reported at the end.
                if skipped < 5 {
                    eprintln!("  skipping row: {e}");
                }
                skipped += 1;
                continue;
            }
        };
        if should_normalize {
            normalize(&mut vector);
        }

        if vectors.is_none() {
            dim = vector.len();
            eprintln!("  detected {dim} dimensions");
            vectors = Some(Matrix::new(dim));
        }
        if vector.len() != dim {
            if skipped < 5 {
                eprintln!("  skipping row: {} dims, expected {dim}", vector.len());
            }
            skipped += 1;
            continue;
        }

        let text = |i: Option<&usize>| -> Option<String> {
            i.and_then(|i| row.get::<_, Option<String>>(*i).ok().flatten())
        };

        let id: String = match row.get_ref(0)? {
            ValueRef::Text(t) => String::from_utf8_lossy(t).into_owned(),
            ValueRef::Integer(n) => n.to_string(),
            _ => {
                skipped += 1;
                continue;
            }
        };

        ingest.push(IngestRow {
            id,
            body: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            source_title: text(slot.get("title")).unwrap_or_else(|| "Untitled".to_owned()),
            // ! Empty rather than fabricated. A synthesized URL would look like
            // provenance a human could click, and it is the one thing the
            // engine must never invent (CLAUDE.md §7 rule 8).
            source_url: text(slot.get("url")).unwrap_or_default(),
            locator_page: slot
                .get("page")
                .and_then(|i| row.get::<_, Option<i32>>(*i).ok().flatten()),
            locator_section: text(slot.get("section")),
            heading_path: text(slot.get("heading")),
            identifier: text(slot.get("identifier")),
        });
        vectors.as_mut().expect("initialized above").push(&vector);
    }

    let vectors = vectors.ok_or("source produced no usable rows")?;
    eprintln!(
        "  {} rows, {} skipped · {:.2} GB of vectors",
        ingest.len(),
        skipped,
        vectors.bytes() as f64 / 1e9
    );

    let space = EmbeddingSpace {
        model_id: flag(args, "--model").unwrap_or_else(|| "unknown/unspecified".to_owned()),
        dim: parsed(args, "--dim", dim)?,
        normalized: should_normalize,
        query_instruction: vera_core::DEFAULT_QUERY_INSTRUCTION.to_owned(),
    };
    if space.dim != dim {
        return Err(format!(
            "--dim {} contradicts the {dim} dimensions actually read · \
             one of them is wrong and guessing would corrupt every vector",
            space.dim
        )
        .into());
    }
    if space.model_id == "unknown/unspecified" {
        eprintln!(
            "WARNING: no --model given · the corpus will record 'unknown/unspecified' and the \
             engine cannot verify a query is embedded in the same space"
        );
    }

    let per_cluster = parsed(args, "--per-cluster", 10_000usize)?;
    let k = KMeansConfig::clusters_for(ingest.len(), per_cluster);
    eprintln!("clustering {} rows into {k} clusters…", ingest.len());

    let report = build_corpus(
        &out,
        &space,
        &flag(args, "--domain").unwrap_or_else(|| "corpus".to_owned()),
        &flag(args, "--description").unwrap_or_else(|| "Imported corpus".to_owned()),
        &ingest,
        &vectors,
        &KMeansConfig {
            k,
            max_iters: parsed(args, "--iters", 20usize)?,
            ..Default::default()
        },
    )?;

    println!("corpus:            {out}");
    println!("rows:              {} ({skipped} skipped)", report.rows);
    println!("model:             {}", space.model_id);
    println!("dimensions:        {}", space.dim);
    println!("clusters:          {}", report.clusters);
    println!("cluster tightness: {:.4}", report.mean_similarity);
    println!(
        "cluster sizes:     {} min / {} max",
        report.smallest_cluster, report.largest_cluster
    );
    println!("build time:        {:.1}s", report.build_seconds);
    println!(
        "anchor p1/p50/p95: {:.4} / {:.4} / {:.4}",
        report.anchor.p1, report.anchor.p50, report.anchor.p95
    );
    println!("domain_threshold:  {:.4} (calibrated)", report.anchor.threshold);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn f64_blob(v: &[f64]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn f32_blobs_round_trip() {
        let v = vec![0.25f32, -1.0, 3.5, 0.0];
        let blob = f32_blob(&v);
        assert_eq!(decode(&ValueRef::Blob(&blob), VectorFormat::F32Le).unwrap(), v);
    }

    #[test]
    fn f64_blobs_are_narrowed_to_f32() {
        let blob = f64_blob(&[0.25, -1.0, 3.5]);
        let got = decode(&ValueRef::Blob(&blob), VectorFormat::F64Le).unwrap();
        assert_eq!(got, vec![0.25f32, -1.0, 3.5]);
    }

    #[test]
    fn json_arrays_are_accepted() {
        let got = decode(&ValueRef::Text(b"[0.5, -0.25, 1.0]"), VectorFormat::Json).unwrap();
        assert_eq!(got, vec![0.5f32, -0.25, 1.0]);
    }

    #[test]
    fn a_blob_that_is_not_a_whole_number_of_floats_is_refused() {
        // ! Refused, ✗ truncated. A short read would shift every value and the
        // result would still cluster and still rank — silently wrong.
        let blob = vec![0u8; 10];
        assert!(decode(&ValueRef::Blob(&blob), VectorFormat::F32Le).is_err());
        assert!(decode(&ValueRef::Blob(&blob), VectorFormat::F64Le).is_err());
    }

    #[test]
    fn reading_f64_data_as_f32_yields_the_wrong_width_which_is_why_it_is_a_flag() {
        // ! The mistake `inspect` exists to prevent: the bytes decode happily,
        // there is no error, and the corpus is garbage. 1024 f64s read as f32
        // give 2048 dimensions of interleaved mantissa halves.
        let truth: Vec<f64> = (0..1024).map(|i| f64::from(i) / 1024.0).collect();
        let blob = f64_blob(&truth);
        let wrong = decode(&ValueRef::Blob(&blob), VectorFormat::F32Le).unwrap();
        assert_eq!(wrong.len(), 2048, "no error is raised · only the width betrays it");
        let right = decode(&ValueRef::Blob(&blob), VectorFormat::F64Le).unwrap();
        assert_eq!(right.len(), 1024);
    }

    #[test]
    fn the_format_guess_prefers_f32_and_reports_the_dimension() {
        let blob = f32_blob(&vec![0.1; 1024]);
        assert_eq!(
            guess_format(&ValueRef::Blob(&blob)),
            Some((VectorFormat::F32Le, 1024))
        );
    }

    #[test]
    fn the_format_guess_recognizes_json() {
        let text = b"[0.1, 0.2, 0.3]";
        // Below the 64-dim floor for blobs, but JSON is unambiguous.
        assert_eq!(
            guess_format(&ValueRef::Text(text)),
            Some((VectorFormat::Json, 3))
        );
    }

    #[test]
    fn the_format_guess_declines_values_that_are_not_vectors() {
        assert_eq!(guess_format(&ValueRef::Integer(42)), None);
        assert_eq!(guess_format(&ValueRef::Text(b"hello world")), None);
        // Too short to be an embedding.
        let tiny = f32_blob(&[1.0, 2.0]);
        assert_eq!(guess_format(&ValueRef::Blob(&tiny)), None);
    }

    #[test]
    fn format_names_parse_with_common_aliases() {
        assert_eq!("f32le".parse::<VectorFormat>().unwrap(), VectorFormat::F32Le);
        assert_eq!("float32".parse::<VectorFormat>().unwrap(), VectorFormat::F32Le);
        assert_eq!("json".parse::<VectorFormat>().unwrap(), VectorFormat::Json);
        assert!("f16".parse::<VectorFormat>().is_err());
    }

    #[test]
    fn normalizing_makes_a_unit_vector_and_leaves_zero_alone() {
        let mut v = vec![3.0f32, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut zero = vec![0.0f32, 0.0];
        normalize(&mut zero);
        assert_eq!(zero, vec![0.0, 0.0], "must not divide by zero");
    }
}
