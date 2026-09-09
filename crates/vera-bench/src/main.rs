//! `vera-bench` · builds corpora and measures what routing actually buys.
//!
//! Three commands: `synth` makes a corpus with known cluster structure, `info`
//! describes one, and `run` sweeps `clusters_probed` reporting latency **and**
//! recall against an exhaustive baseline.
//!
//! ! Latency and recall are reported together, always. A routing configuration
//! that is 40× faster and finds half the answers is not a win, and either
//! number alone hides that. The sweep exists to find where the curve turns.

mod metrics;
mod synth;

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use metrics::{Latencies, recall_at_k, reciprocal_rank, routing_recall, top1_hit};
use vera_core::{Config, EmbeddingSpace};
use vera_engine::{Engine, Probe, TopK};
use vera_index::{KMeansConfig, Rng, build_corpus};
use vera_store::{ChunkStore, sqlite::SqliteStore};
use synth::SynthConfig;

fn main() {
    // All diagnostics to stderr · stdout stays clean for piping the report,
    // and the same discipline the MCP stdio transport requires (CLAUDE.md §7
    // rule 10) applies here so the two never diverge.
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("synth") => cmd_synth(&args[1..]),
        Some("info") => cmd_info(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        _ => {
            eprintln!(
                "usage:
  vera-bench synth --out <db> [--rows N] [--dim N] [--topics N] [--spread F]
                   [--anisotropy F] [--iters N] [--seed N]
                   [--per-cluster N]   override sqrt(N) cluster sizing
  vera-bench info  --corpus <db>
  vera-bench run   --corpus <db> [--queries N] [--probe 1,2,5,10] [--k N] [--jitter F]

  synth   build a synthetic corpus with known cluster structure
  info    describe a corpus and its RAM budget
  run     sweep clusters_probed, reporting latency and recall vs a full scan"
            );
            std::process::exit(2);
        }
    }
}

// ── arg helpers ──────────────────────────────────────────────────────────────

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
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

// ── synth ────────────────────────────────────────────────────────────────────

fn cmd_synth(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let out = required(args, "--out")?;
    let cfg = SynthConfig {
        rows: parsed(args, "--rows", 200_000usize)?,
        dim: parsed(args, "--dim", 1024usize)?,
        topics: parsed(args, "--topics", 200usize)?,
        spread: parsed(args, "--spread", 0.35f32)?,
        anisotropy: parsed(args, "--anisotropy", 0.7f32)?,
        seed: parsed(args, "--seed", 0xC0FFEEu64)?,
    };
    // ! Default is sqrt(N), ✗ a fixed rows-per-cluster target. See
    // KMeansConfig::sqrt_n: "10K rows per cluster" is the 100M design point, and
    // applying it to a 200K corpus gives 20 clusters, so probing 5 scans a
    // quarter of the corpus and the measured speedup is meaningless.
    let k = match flag(args, "--per-cluster") {
        Some(v) => KMeansConfig::clusters_for(cfg.rows, v.parse::<usize>()?),
        None => KMeansConfig::sqrt_n(cfg.rows),
    };

    eprintln!(
        "generating {} rows × {} dims across {} topics (spread {})…",
        cfg.rows, cfg.dim, cfg.topics, cfg.spread
    );
    let t = Instant::now();
    let (rows, vectors) = synth::generate(&cfg);
    eprintln!(
        "  generated in {:.1}s · {:.2} GB of vectors in memory",
        t.elapsed().as_secs_f64(),
        vectors.bytes() as f64 / 1e9
    );

    let space = EmbeddingSpace {
        model_id: "synthetic/bench".into(),
        dim: cfg.dim,
        normalized: true,
        query_instruction: String::new(),
    };
    eprintln!(
        "clustering into {k} clusters (~{} rows each)…",
        cfg.rows / k.max(1)
    );

    let report = build_corpus(
        &out,
        &space,
        "synthetic",
        "Synthetic benchmark corpus",
        &rows,
        &vectors,
        &KMeansConfig {
            k,
            max_iters: parsed(args, "--iters", 15usize)?,
            ..Default::default()
        },
    )?;

    println!("corpus:            {out}");
    println!("rows:              {}", report.rows);
    println!("clusters:          {}", report.clusters);
    println!("kmeans iterations: {}", report.iterations);
    println!(
        "cluster tightness: {:.4}  (mean cosine to own centroid)",
        report.mean_similarity
    );
    println!(
        "cluster sizes:     {} min / {} max",
        report.smallest_cluster, report.largest_cluster
    );
    println!("build time:        {:.1}s", report.build_seconds);
    println!();
    println!("layer-1 anchor (cosine of a row to the domain anchor):");
    println!(
        "  min {:.4}  p1 {:.4}  p5 {:.4}  p50 {:.4}  p95 {:.4}  max {:.4}",
        report.anchor.min,
        report.anchor.p1,
        report.anchor.p5,
        report.anchor.p50,
        report.anchor.p95,
        report.anchor.max
    );
    println!(
        "  calibrated domain_threshold = {:.4}  (p1 · ~99% of corpus clears it)",
        report.anchor.threshold
    );
    Ok(())
}

// ── info ─────────────────────────────────────────────────────────────────────

fn cmd_info(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "--corpus")?;
    let store = SqliteStore::open(&path)?;
    let space = store.corpus_space()?;
    let domains = store.domains()?;
    let largest = store.largest_cluster_rows()?;

    let config = Config {
        embedding: space.clone(),
        ..Config::default()
    };

    println!("corpus:      {path}");
    println!("model:       {}", space.model_id);
    println!("dimensions:  {}", space.dim);
    println!("normalized:  {}", space.normalized);
    for d in &domains {
        let centroids = store.centroids(&d.id)?;
        println!(
            "domain:      {} · {} rows · {} clusters",
            d.id,
            d.row_count,
            centroids.len()
        );
    }
    println!("largest cluster: {largest} rows");
    println!();
    println!("RAM budget (MCP_ENGINE.md §5):");
    println!(
        "  per-request ceiling: {:.1} MB  (one cluster at f32)",
        config.per_request_ceiling_bytes(largest) as f64 / 1e6
    );
    println!(
        "  × {} concurrent:     {:.1} MB",
        config.concurrency.max_concurrent,
        config.concurrency.max_concurrent as f64
            * config.per_request_ceiling_bytes(largest) as f64
            / 1e6
    );
    Ok(())
}

// ── run ──────────────────────────────────────────────────────────────────────

/// One query: the text BM25 sees and the vector routing sees.
struct Query {
    text: String,
    vector: Vec<f32>,
}

/// Sample queries from the corpus itself.
///
/// ! Queries are drawn from the corpus and then *perturbed*, so a query is near
/// a real document without being identical to one. Using rows verbatim would
/// make every query a trivial exact hit and report a recall that no real
/// workload will ever see.
fn sample_queries(
    store: &SqliteStore,
    domain: &str,
    count: usize,
    jitter: f32,
    seed: u64,
) -> Result<Vec<Query>, Box<dyn std::error::Error>> {
    let centroids = store.centroids(domain)?;
    if centroids.is_empty() {
        return Err("corpus has no clusters".into());
    }
    let mut rng = Rng::new(seed);
    let mut queries = Vec::with_capacity(count);

    while queries.len() < count {
        let cluster = &centroids[rng.below(centroids.len())];
        // Take the first row of a randomly chosen cluster; cheap and spreads
        // queries across the corpus's topics.
        let mut picked: Option<(String, Vec<f32>)> = None;
        let want = rng.below(cluster.row_count.max(1) as usize);
        let mut seen = 0usize;
        store.scan_cluster(cluster.id, &mut |row| {
            if seen == want && picked.is_none() {
                picked = Some((row.id.to_owned(), row.vector.to_vec()));
            }
            seen += 1;
        })?;
        let Some((id, vector)) = picked else { continue };
        let Some(chunk) = store.chunks_by_id(&[id])?.into_iter().next() else {
            continue;
        };
        queries.push(Query {
            // Keep a slice of the body · a realistic query is shorter than the
            // document it should find.
            text: chunk.body.split_whitespace().take(8).collect::<Vec<_>>().join(" "),
            vector: synth::query_near(&vector, jitter, &mut rng),
        });
    }
    Ok(queries)
}

/// The true dense-only top-k, by exhaustive scan.
///
/// ! Routing recall must be measured against **this**, ✗ against the fused
/// exhaustive result. Part of the fused baseline is by construction not
/// dense-reachable: a document BM25 found on an exact term match may sit
/// nowhere near the query vector, so no amount of probing would ever reach it.
/// Scoring routing against a target it cannot hit by design understates it, and
/// the resulting number says more about the corpus's lexical overlap than about
/// the clustering.
fn dense_baseline(
    store: &SqliteStore,
    domain: &str,
    query_vector: &[f32],
    k: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut top = TopK::new(k);
    for c in store.centroids(domain)? {
        store.scan_cluster(c.id, &mut |row| {
            top.offer(row.id, vera_embed::cosine(query_vector, row.vector));
        })?;
    }
    Ok(top.into_ranked().into_iter().map(|s| s.id).collect())
}

#[allow(clippy::too_many_lines)]
fn cmd_run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = required(args, "--corpus")?;
    let n_queries = parsed(args, "--queries", 100usize)?;
    let k = parsed(args, "--k", 10usize)?;
    let jitter = parsed(args, "--jitter", 0.15f32)?;
    let seed = parsed(args, "--seed", 12345u64)?;
    let probes: Vec<usize> = flag(args, "--probe")
        .unwrap_or_else(|| "1,2,3,5,10,20".to_owned())
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;

    let store = SqliteStore::open(&path)?;
    // Second read-only handle: the engine takes ownership of the first, and the
    // dense baseline needs raw scan access the tool surface deliberately lacks.
    let raw = SqliteStore::open(&path)?;
    let space = store.corpus_space()?;
    let domains = store.domains()?;
    let domain = domains.first().ok_or("corpus has no domain")?.id.clone();
    let total_clusters = store.centroids(&domain)?.len();
    let corpus_rows: i64 = domains.iter().map(|d| d.row_count).sum();

    eprintln!("sampling {n_queries} queries (jitter {jitter})…");
    let queries = sample_queries(&store, &domain, n_queries, jitter, seed)?;

    // ! Chunk → cluster, for routing recall. Built once from the store before
    // the engine takes ownership: `EVAL.md` §3 needs to know whether a true
    // result was *reachable*, which is a fact about the index, not the query.
    eprintln!("mapping chunks to clusters for routing recall…");
    let mut cluster_of: HashMap<String, i32> = HashMap::new();
    for c in store.centroids(&domain)? {
        store.scan_cluster(c.id, &mut |row| {
            cluster_of.insert(row.id.to_owned(), c.id);
        })?;
    }

    let config = Config {
        embedding: space.clone(),
        search: vera_core::SearchConfig {
            max_results: k,
            ..Default::default()
        },
        ..Config::default()
    };
    let engine = Engine::load(store, config)?;

    // ── Baseline: exhaustive scan ────────────────────────────────────────────
    // ! Ground truth for recall. Same scoring, same fusion — the only variable
    // is how many clusters were opened, so the gap is attributable to routing.
    eprintln!("running exhaustive baseline over all {total_clusters} clusters…");
    let mut baselines = Vec::with_capacity(queries.len());
    let mut dense_truth = Vec::with_capacity(queries.len());
    let mut baseline_times = Vec::with_capacity(queries.len());
    let mut baseline_rows = 0usize;
    for q in &queries {
        let out = engine.search(&q.text, &q.vector, Probe::Exhaustive)?;
        baseline_times.push(out.timings.total);
        baseline_rows += out.timings.rows_scanned;
        baselines.push(
            out.response
                .results
                .iter()
                .map(|r| r.id.clone())
                .collect::<Vec<_>>(),
        );
        dense_truth.push(dense_baseline(&raw, &domain, &q.vector, k)?);
    }
    let baseline_lat = Latencies::from_durations(&baseline_times);
    let baseline_rows_avg = baseline_rows / queries.len().max(1);

    println!();
    println!("corpus       {path}");
    println!(
        "             {corpus_rows} rows · {total_clusters} clusters · {} dims · {}",
        space.dim, space.model_id
    );
    println!("queries      {} · k={k} · jitter={jitter}", queries.len());
    println!();
    println!(
        "baseline (exhaustive): p50 {:.1} ms · p95 {:.1} ms · max {:.1} ms · mean {:.1} ms",
        baseline_lat.p50(),
        baseline_lat.p95(),
        baseline_lat.max(),
        baseline_lat.mean()
    );
    println!("             {baseline_rows_avg} rows scanned per query");
    println!();
    // EVAL.md §3: recall@k is the primary metric, routing recall is the
    // diagnostic that says which half to fix, MRR is ranking quality.
    println!(
        "{:>6}  {:>9}  {:>9}  {:>9}  {:>8}  {:>10}  {:>8}  {:>8}  {:>6}  {:>6}  {:>6}",
        "probe", "p50 ms", "p95 ms", "p99 ms", "speedup", "rows/query", "recall", "route/D",
        "L1-rej", "MRR", "top-1"
    );
    println!("{}", "-".repeat(110));

    for probe in &probes {
        let mut times = Vec::with_capacity(queries.len());
        let mut stage = StageTotals::default();
        let mut rows_scanned = 0usize;
        let mut recall_sum = 0.0f32;
        let mut dense_routing_sum = 0.0f32;
        // ! Counted separately. A query layer-1 rejects has an empty probe set,
        // so it scores 0 routing recall — but the cause is domain detection, not
        // clustering, and fixing the wrong one wastes the measurement.
        let mut rejected = 0usize;
        let mut mrr_sum = 0.0f32;
        let mut top1 = 0usize;

        for ((q, baseline), dense) in queries.iter().zip(&baselines).zip(&dense_truth) {
            let out = engine.search(&q.text, &q.vector, Probe::Nearest(*probe))?;
            times.push(out.timings.total);
            stage.add(&out.timings);
            rows_scanned += out.timings.rows_scanned;
            let ids: Vec<String> = out.response.results.iter().map(|r| r.id.clone()).collect();
            recall_sum += recall_at_k(&ids, baseline, k);
            if out.response.detected_domain.is_none() {
                rejected += 1;
            }
            let probed: HashSet<i32> = out.probed.iter().map(|p| p.cluster_id).collect();
            dense_routing_sum += routing_recall(dense, &cluster_of, &probed, k);
            mrr_sum += reciprocal_rank(&ids, baseline);
            if top1_hit(&ids, baseline) {
                top1 += 1;
            }
        }

        let lat = Latencies::from_durations(&times);
        let n = queries.len().max(1);
        #[allow(clippy::cast_precision_loss)]
        let recall = recall_sum / n as f32;
        #[allow(clippy::cast_precision_loss)]
        let dense_routing = dense_routing_sum / n as f32;
        #[allow(clippy::cast_precision_loss)]
        let reject_rate = rejected as f32 / n as f32;
        #[allow(clippy::cast_precision_loss)]
        let mrr = mrr_sum / n as f32;
        #[allow(clippy::cast_precision_loss)]
        let top1_rate = top1 as f32 / n as f32;
        let speedup = if lat.p50() > 0.0 {
            baseline_lat.p50() / lat.p50()
        } else {
            0.0
        };

        println!(
            "{probe:>6}  {:>9.2}  {:>9.2}  {:>9.2}  {:>7.1}×  {:>10}  {:>7.1}%  {:>7.1}%  \
             {:>5.1}%  {:>6.3}  {:>5.1}%",
            lat.p50(),
            lat.p95(),
            lat.p99(),
            speedup,
            rows_scanned / n,
            recall * 100.0,
            dense_routing * 100.0,
            reject_rate * 100.0,
            mrr,
            top1_rate * 100.0
        );
    }

    println!();
    println!(
        "recall  = fraction of the FUSED exhaustive top-{k} that routed search returned
          (end-to-end quality · what the caller actually gets)
route/D = fraction of the DENSE-ONLY exhaustive top-{k} that was in a probed
          cluster · EVAL.md §3 · this is the honest routing measure, because
          the fused baseline contains keyword hits routing cannot reach by design
L1-rej  = share of queries layer-1 refused outright (detected_domain: null).
          These score 0 route/D no matter how many clusters are probed, so a
          flat route/D across the sweep is this, not a clustering problem.
          route/D high + recall low → fix fusion / per-cluster top-k
          route/D low               → fix clustering or raise clusters_probed"
    );
    println!();
    println!("stage breakdown at probe={} (mean ms/query):", probes.last().copied().unwrap_or(5));
    let mut stage = StageTotals::default();
    for q in &queries {
        let out = engine.search(
            &q.text,
            &q.vector,
            Probe::Nearest(probes.last().copied().unwrap_or(5)),
        )?;
        stage.add(&out.timings);
    }
    stage.print(queries.len().max(1));

    Ok(())
}

#[derive(Default)]
struct StageTotals {
    route_domain: Duration,
    route_cluster: Duration,
    leaf_dense: Duration,
    leaf_keyword: Duration,
    exact_path: Duration,
    fuse: Duration,
    hydrate: Duration,
    total: Duration,
}

impl StageTotals {
    fn add(&mut self, t: &vera_engine::StageTimings) {
        self.route_domain += t.route_domain;
        self.route_cluster += t.route_cluster;
        self.leaf_dense += t.leaf_dense;
        self.leaf_keyword += t.leaf_keyword;
        self.exact_path += t.exact_path;
        self.fuse += t.fuse;
        self.hydrate += t.hydrate;
        self.total += t.total;
    }

    fn print(&self, n: usize) {
        #[allow(clippy::cast_precision_loss)]
        let ms = |d: Duration| d.as_secs_f64() * 1e3 / n as f64;
        let total = ms(self.total).max(1e-9);
        for (label, d) in [
            ("layer-1 domain", self.route_domain),
            ("layer-2 cluster", self.route_cluster),
            ("layer-3 dense scan", self.leaf_dense),
            ("layer-3 bm25", self.leaf_keyword),
            ("exact-id (global)", self.exact_path),
            ("rrf fusion", self.fuse),
            ("hydrate", self.hydrate),
        ] {
            println!("  {label:<20} {:>8.3} ms  {:>5.1}%", ms(d), ms(d) / total * 100.0);
        }
        println!("  {:<20} {:>8.3} ms", "total", ms(self.total));
    }
}
