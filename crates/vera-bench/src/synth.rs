//! A synthetic corpus with real cluster structure.
//!
//! ! Exists so the routing experiment can run **before** the real corpus
//! arrives, and so its behavior can be varied deliberately: the parameter that
//! decides whether routing works is how tightly the corpus clusters, and on a
//! real corpus that is fixed and unknown. Here it is a dial (`spread`), which
//! makes it possible to ask "at what point does routing stop paying?" rather
//! than only "does it pay on this one dataset?".
//!
//! Vectors are drawn around `topics` latent centers, which is what a real
//! embedded corpus looks like: documents about the same subject land near each
//! other. A uniformly random corpus has no cluster structure at all, so routing
//! would (correctly) look useless on it — measuring against that would be
//! measuring the fixture, not the engine.

use vera_index::{IngestRow, Matrix, Rng};

/// Roots the generated vocabulary is built from · Indonesian-regulation flavor,
/// so a body is readable when a test fails.
const ROOTS: &[&str] = &[
    "wajib", "pajak", "sanksi", "administrasi", "ketentuan", "umum", "tata", "cara",
    "perpajakan", "penghasilan", "badan", "orang", "pribadi", "tarif", "pengenaan",
    "pemungutan", "penyetoran", "pelaporan", "keberatan", "banding", "surat", "teguran",
    "denda", "bunga", "kenaikan", "restitusi", "kompensasi", "pemeriksaan", "penyidikan",
    "daerah", "pusat", "menteri", "keuangan", "direktorat", "jenderal", "peraturan",
    "pelaksanaan", "perubahan", "pencabutan", "berlaku", "kewajiban", "hak", "subjek",
    "objek", "dasar", "pengurangan", "fasilitas", "insentif", "pembukuan", "pencatatan",
    "putusan", "gugatan", "sengketa", "pengadilan", "hakim", "saksi", "bukti", "dakwaan",
    "kontrak", "perjanjian", "pihak", "klausul", "wanprestasi", "ganti", "rugi", "somasi",
];

const PREFIXES: &[&str] = &["", "me", "pe", "ber", "ter", "di", "ke", "se", "peng", "meng"];
const SUFFIXES: &[&str] = &["", "an", "kan", "nya", "i", "annya"];

const DOC_TYPES: &[&str] = &["UU", "PP", "PERPRES", "PMK", "PERMEN"];

/// A vocabulary whose term frequencies follow **Zipf's law**.
///
/// ! This is the fixture's single most consequential dial for the keyword half,
/// and the benchmark ran without it for a long time. With a closed 50-word
/// vocabulary every term appears in nearly every row, so `ln(N/df) ≈ 0`: an
/// 8-term `OR` matches most of the corpus, FTS5 degenerates into a full scan,
/// and the 291 ms it cost was recorded as "BM25 is 72% of query time"
/// (`METRICS.md` §2.3). That number describes the fixture, ✗ the engine, and
/// **both directions of error were live**: a real corpus could be far faster
/// (selective terms touch few postings) or far slower (500× the rows).
///
/// Real text obeys `df(rank) ∝ rank^−s` with `s ≈ 1`. Under it most terms are
/// rare, most query terms prune hard, and the inverted index does the job it
/// exists to do. Sweeping `s` toward 0 recovers the old fixture, so the
/// sensitivity of every keyword number to this assumption is now measurable
/// rather than assumed.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    terms: Vec<String>,
    /// Cumulative sampling weights · `cdf[i]` is `P(rank <= i)`.
    cdf: Vec<f32>,
}

impl Vocabulary {
    /// Build `size` terms whose sampling weight is `1 / (rank + 1)^exponent`.
    #[must_use]
    pub fn zipf(size: usize, exponent: f32) -> Self {
        let size = size.max(1);
        let mut terms = Vec::with_capacity(size);
        // ! Deduplicated, ✗ assumed distinct. Morphological composition
        // genuinely collides — `wajib`+`an` can equal another root outright —
        // and a duplicated term would be sampled at two different ranks, so the
        // realised frequency distribution would stop being the one the exponent
        // describes. The vocabulary *size* is the dial; it has to be exact.
        let mut seen = std::collections::HashSet::new();
        'outer: for suffix in SUFFIXES {
            for prefix in PREFIXES {
                for root in ROOTS {
                    if terms.len() == size {
                        break 'outer;
                    }
                    let term = format!("{prefix}{root}{suffix}");
                    if seen.insert(term.clone()) {
                        terms.push(term);
                    }
                }
            }
        }
        // Past composition the strings stop mattering; only the count does.
        let mut i = 0usize;
        while terms.len() < size {
            let term = format!("{}{i}", ROOTS[i % ROOTS.len()]);
            if seen.insert(term.clone()) {
                terms.push(term);
            }
            i += 1;
        }

        #[allow(clippy::cast_precision_loss)]
        let weights: Vec<f32> = (0..size)
            .map(|r| ((r + 1) as f32).powf(-exponent))
            .collect();
        let total: f32 = weights.iter().sum();
        let mut cdf = Vec::with_capacity(size);
        let mut running = 0.0f32;
        for w in &weights {
            running += w / total;
            cdf.push(running);
        }
        // Guard the last bucket against float drift so `sample` cannot fall off
        // the end for a draw of 0.9999999.
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Self { terms, cdf }
    }

    /// Draw one term index, Zipf-distributed.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.unit();
        match self
            .cdf
            .binary_search_by(|c| c.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) | Err(i) => i.min(self.cdf.len() - 1),
        }
    }

    #[must_use]
    pub fn term(&self, i: usize) -> &str {
        &self.terms[i.min(self.terms.len() - 1)]
    }

    #[must_use]
    #[allow(dead_code, reason = "part of the fixture surface; exercised by tests")]
    pub fn len(&self) -> usize {
        self.terms.len()
    }
}

/// Shape of a synthetic corpus.
#[derive(Debug, Clone)]
pub struct SynthConfig {
    pub rows: usize,
    pub dim: usize,
    /// Latent topics the vectors are drawn around.
    pub topics: usize,
    /// How far a row drifts from its topic center, `0.0`–`1.0`.
    ///
    /// ! The dial that decides whether routing can work. Near 0 the corpus is
    /// perfectly clustered and routing is free; near 1 it is noise and routing
    /// must lose recall. A real corpus sits somewhere between, and this is how
    /// to bracket it.
    pub spread: f32,
    /// How narrow a cone the whole corpus occupies, `0.0`–`1.0`.
    ///
    /// ! Real transformer embeddings are strongly **anisotropic**: they do not
    /// spread over the sphere, they crowd into a narrow cone, so two unrelated
    /// documents still have a substantial positive cosine. Drawing topic
    /// centers uniformly instead produces a corpus whose mean vector — the
    /// layer-1 anchor — is near zero and points nowhere, which makes domain
    /// detection look broken when it is the fixture that is unrealistic.
    pub anisotropy: f32,
    /// Distinct terms in the generated vocabulary.
    ///
    /// ! The keyword half's equivalent of `spread`. See [`Vocabulary`] for what
    /// the old closed 50-word vocabulary did to every BM25 measurement.
    pub vocabulary: usize,
    /// Zipf exponent · `1.0` is natural language, `0.0` is a uniform vocabulary
    /// (the old fixture, kept reachable so the sensitivity can be measured).
    pub zipf: f32,
    /// Tokens per body. Real regulation chunks run longer; BM25
    /// length-normalizes, so this is a factor in its own right (`FACTORS.md`).
    pub body_tokens: usize,
    /// Terms that characterise one topic · the lexical half of topic structure.
    pub topic_terms: usize,
    pub seed: u64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            rows: 200_000,
            dim: 1024,
            topics: 200,
            spread: 0.35,
            anisotropy: 0.7,
            vocabulary: 20_000,
            zipf: 1.0,
            body_tokens: 18,
            topic_terms: 64,
            seed: 0xC0FFEE,
        }
    }
}

/// Generate rows and their vectors.
#[must_use]
pub fn generate(cfg: &SynthConfig) -> (Vec<IngestRow>, Matrix) {
    let mut rng = Rng::new(cfg.seed);

    // ! Mixing happens between **unit vectors**, ✗ raw components. A random
    // component drawn from [-1,1] has magnitude ~1, while a unit vector's
    // component is ~1/√dim (0.03 at 1024 dims) — so adding raw noise at any
    // visible weight drowns the signal completely and both dials silently stop
    // working. Normalizing each part first makes `spread` and `anisotropy` mean
    // what they say: the fraction of the resulting direction that is noise.
    let base = random_unit(cfg.dim, &mut rng);

    // Latent topic centers, unit-normalized, all inside the corpus cone.
    let mut centers = Matrix::with_capacity(cfg.dim, cfg.topics.max(1));
    for _ in 0..cfg.topics.max(1) {
        let noise = random_unit(cfg.dim, &mut rng);
        centers.push(&mix(&base, &noise, cfg.anisotropy));
    }

    // ! The lexical structure, built once. Topic term lists are themselves
    // Zipf-drawn, so topics **share their common words and diverge in the tail**
    // — which is what real topical text does, and what makes a rare query term
    // genuinely diagnostic of a topic while a common one is not.
    let vocab = Vocabulary::zipf(cfg.vocabulary, cfg.zipf);
    let topic_terms: Vec<Vec<usize>> = (0..cfg.topics.max(1))
        .map(|_| {
            (0..cfg.topic_terms.max(1))
                .map(|_| vocab.sample(&mut rng))
                .collect()
        })
        .collect();

    let mut vectors = Matrix::with_capacity(cfg.dim, cfg.rows);
    let mut rows = Vec::with_capacity(cfg.rows);

    for i in 0..cfg.rows {
        let topic = rng.below(cfg.topics.max(1));
        let center = centers.row(topic);

        let noise = random_unit(cfg.dim, &mut rng);
        vectors.push(&mix(center, &noise, 1.0 - cfg.spread));

        // Half the tokens come from this topic's term list and half from the
        // corpus-wide Zipf draw, so keyword and dense signals correlate the way
        // they do in real text without the lexical signal becoming a giveaway.
        let terms = &topic_terms[topic];
        let body: Vec<&str> = (0..cfg.body_tokens.max(1))
            .map(|_| {
                let pick = if rng.unit() < 0.5 {
                    terms[rng.below(terms.len())]
                } else {
                    vocab.sample(&mut rng)
                };
                vocab.term(pick)
            })
            .collect();

        // ~1 row in 50 carries a citable identifier, as a real corpus does.
        let identifier = (i % 50 == 0).then(|| {
            format!(
                "{} {}/{}",
                DOC_TYPES[topic % DOC_TYPES.len()],
                (i / 50) % 200 + 1,
                1990 + (topic % 35)
            )
        });

        rows.push(IngestRow {
            id: format!("chunk-{i:08}"),
            // ! Nothing but sampled terms. An earlier version appended
            // "nomor {i % 500} tentang …", and those 500 numeric literals were
            // near-uniformly distributed across the corpus — highly selective
            // tokens injected outside the vocabulary model. They made a
            // deliberately uniform fixture *profile* as though it had real term
            // selectivity, which is the precise measurement error this fixture
            // work exists to remove.
            body: body.join(" "),
            source_title: format!("Dokumen {}", topic),
            source_url: format!("https://peraturan.example/doc-{topic}.pdf"),
            locator_page: Some((i % 400) as i32 + 1),
            locator_section: Some(format!("Pasal {}", i % 90 + 1)),
            heading_path: None,
            identifier,
        });
    }

    (rows, vectors)
}

/// Build a query near an existing row · what a genuine query looks like in
/// vector space: close to the answer, not identical to it.
///
/// Returns the perturbed unit vector.
#[must_use]
pub fn query_near(row: &[f32], jitter: f32, rng: &mut Rng) -> Vec<f32> {
    let noise = random_unit(row.len(), rng);
    mix(row, &noise, 1.0 - jitter)
}

/// A uniformly random unit vector.
fn random_unit(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.unit() * 2.0 - 1.0).collect();
    normalize(&mut v);
    v
}

/// `weight` of `a` plus `1 - weight` of `b`, renormalized.
///
/// Both inputs are expected to be unit vectors, so `weight` is directly the
/// share of the result's direction that comes from `a`.
fn mix(a: &[f32], b: &[f32], weight: f32) -> Vec<f32> {
    let mut v: Vec<f32> = a
        .iter()
        .zip(b)
        .map(|(x, y)| x * weight + y * (1.0 - weight))
        .collect();
    normalize(&mut v);
    v
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use vera_core::ProfileBuilder;

    fn cfg(rows: usize) -> SynthConfig {
        SynthConfig {
            rows,
            dim: 32,
            topics: 4,
            spread: 0.2,
            anisotropy: 0.7,
            seed: 7,
            ..SynthConfig::default()
        }
    }

    fn profile(cfg: &SynthConfig) -> vera_core::CorpusProfile {
        let (rows, _) = generate(cfg);
        let mut b = ProfileBuilder::new();
        for r in &rows {
            b.observe(&r.body, &r.source_url, r.identifier.as_deref());
        }
        b.finish()
    }

    #[test]
    fn the_generated_corpus_is_zipfian_and_its_keyword_numbers_transfer() {
        // ! The fixture bug this replaces. `METRICS.md` §2.3 recorded BM25 at
        // 72% of query time over a 50-word vocabulary where every term matched
        // most of the corpus — a full scan reported as an index lookup. A corpus
        // the profiler calls representative is the precondition for that number
        // meaning anything.
        let p = profile(&SynthConfig { rows: 20_000, ..cfg(0) });
        assert!(p.zipf_slope < -0.5, "slope {}", p.zipf_slope);
        assert!(p.mean_idf > 2.0, "mean idf {}", p.mean_idf);
        assert!(
            p.keyword_metrics_are_representative(),
            "{:?}",
            p.keyword_caveat()
        );
    }

    #[test]
    fn a_uniform_vocabulary_reproduces_the_old_fixture_and_is_flagged() {
        // ! The dial has to reach the broken case, or the sensitivity of every
        // keyword number to this assumption stays unmeasurable.
        let p = profile(&SynthConfig {
            rows: 20_000,
            vocabulary: 50,
            zipf: 0.0,
            ..cfg(0)
        });
        assert!(p.vocabulary <= 50);
        // 18 tokens drawn uniformly from 50 terms puts the average term in ~30%
        // of rows — `ln(1/0.3) ≈ 1.2`, an order of magnitude below what real
        // text gives, and nowhere near enough for an inverted index to prune.
        assert!(p.mean_idf < 2.0, "mean idf {}", p.mean_idf);
        assert!(p.zipf_slope > -0.5, "slope {} · should be near flat", p.zipf_slope);
        assert!(!p.keyword_metrics_are_representative());
    }

    #[test]
    fn zipf_sampling_concentrates_on_the_head_without_abandoning_the_tail() {
        let v = Vocabulary::zipf(1_000, 1.0);
        let mut rng = Rng::new(11);
        let mut counts = vec![0usize; v.len()];
        for _ in 0..100_000 {
            counts[v.sample(&mut rng)] += 1;
        }
        // Rank 1 is drawn far more often than rank 100 …
        assert!(counts[0] > counts[99] * 10, "{} vs {}", counts[0], counts[99]);
        // … but the tail is reachable, or there would be no rare terms at all.
        assert!(counts[999] > 0, "the tail was never sampled");
    }

    #[test]
    fn the_vocabulary_holds_exactly_the_requested_number_of_distinct_terms() {
        for size in [10usize, 500, 20_000] {
            let v = Vocabulary::zipf(size, 1.0);
            assert_eq!(v.len(), size);
            let unique: std::collections::HashSet<&str> =
                (0..size).map(|i| v.term(i)).collect();
            assert_eq!(unique.len(), size, "generated terms collided at size {size}");
        }
    }

    #[test]
    fn topics_share_common_terms_and_differ_in_rare_ones() {
        // ! Why topic term lists are themselves Zipf-drawn. If topics had
        // disjoint vocabularies, any single term would identify a topic outright
        // and the keyword half would look far better than it can be.
        let (rows, _) = generate(&SynthConfig { rows: 4_000, topics: 8, ..cfg(0) });
        let terms_of = |topic_url: &str| -> std::collections::HashSet<String> {
            rows.iter()
                .filter(|r| r.source_url == topic_url)
                .flat_map(|r| r.body.split_whitespace().map(str::to_owned))
                .collect()
        };
        let a = terms_of("https://peraturan.example/doc-0.pdf");
        let b = terms_of("https://peraturan.example/doc-1.pdf");
        assert!(!a.is_empty() && !b.is_empty());
        let shared = a.intersection(&b).count();
        assert!(shared > 0, "topics share nothing · the head is not common");
        assert!(
            a.difference(&b).count() > shared / 4,
            "topics are lexically identical · no keyword signal to route on"
        );
    }

    #[test]
    fn it_generates_matching_rows_and_vectors() {
        let (rows, vectors) = generate(&cfg(100));
        assert_eq!(rows.len(), 100);
        assert_eq!(vectors.rows(), 100);
        assert_eq!(vectors.dim(), 32);
    }

    #[test]
    fn every_vector_is_unit_length() {
        // ! The store and the engine both assume unit vectors for cosine.
        let (_, vectors) = generate(&cfg(200));
        for v in vectors.iter_rows() {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "norm {n}");
        }
    }

    #[test]
    fn ids_are_unique() {
        let (rows, _) = generate(&cfg(500));
        let unique: std::collections::HashSet<_> = rows.iter().map(|r| &r.id).collect();
        assert_eq!(unique.len(), 500);
    }

    #[test]
    fn generation_is_reproducible_for_a_seed() {
        let (a, av) = generate(&cfg(50));
        let (b, bv) = generate(&cfg(50));
        assert_eq!(av, bv);
        assert_eq!(
            a.iter().map(|r| &r.body).collect::<Vec<_>>(),
            b.iter().map(|r| &r.body).collect::<Vec<_>>()
        );
    }

    #[test]
    fn spread_controls_how_far_a_row_sits_from_its_topic() {
        // ! The dials mix unit vectors, so `spread` is directly the noise share
        // and the resulting cosine is predictable rather than incidental.
        let mut rng = Rng::new(1);
        let center = random_unit(256, &mut rng);
        for spread in [0.1f32, 0.35, 0.7] {
            let noise = random_unit(256, &mut rng);
            let row = mix(&center, &noise, 1.0 - spread);
            let cos: f32 = row.iter().zip(&center).map(|(a, b)| a * b).sum();
            let expected = (1.0 - spread) / ((1.0 - spread).powi(2) + spread.powi(2)).sqrt();
            assert!(
                (cos - expected).abs() < 0.08,
                "spread {spread}: cos {cos} vs expected ~{expected}"
            );
        }
    }

    #[test]
    fn low_spread_produces_tighter_topics_than_high_spread() {
        // ! The dial has to actually work, or sweeping it proves nothing.
        let tight = generate(&SynthConfig { spread: 0.05, ..cfg(400) }).1;
        let loose = generate(&SynthConfig { spread: 0.9, ..cfg(400) }).1;
        assert!(
            mean_pairwise(&tight) > mean_pairwise(&loose),
            "spread did not loosen the corpus"
        );
    }

    fn mean_pairwise(m: &Matrix) -> f32 {
        let mut total = 0.0;
        let mut n = 0;
        for i in (0..m.rows()).step_by(7) {
            for j in (0..m.rows()).step_by(11) {
                total += m.row(i).iter().zip(m.row(j)).map(|(a, b)| a * b).sum::<f32>();
                n += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        {
            total / n as f32
        }
    }

    #[test]
    fn an_anisotropic_corpus_has_a_meaningful_domain_anchor() {
        // ! Regression for the bug the first benchmark run surfaced: with
        // uniformly-drawn topic centers the corpus mean is ~0 and points
        // nowhere, so layer-1 rejects every genuine query. Real embeddings are
        // not uniform, and neither is the fixture any more.
        // ! Enough topics for the contrast to exist at all: the mean of a few
        // random directions is still ~1/√topics away from each of them, so with
        // 4 topics even a uniform corpus looks like it has an anchor. The
        // washing-out this guards against is a many-topic phenomenon.
        let (_, vectors) = generate(&SynthConfig { anisotropy: 0.7, topics: 64, ..cfg(2_000) });
        let anchor = vera_index::kmeans::domain_anchor(&vectors);
        let mean: f32 = vectors
            .iter_rows()
            .map(|r| r.iter().zip(&anchor).map(|(a, b)| a * b).sum::<f32>())
            .sum::<f32>()
            / vectors.rows() as f32;
        assert!(mean > 0.4, "corpus does not cohere around its anchor: {mean}");
    }

    #[test]
    fn a_uniform_corpus_has_almost_no_anchor_signal() {
        // The contrast that proves the dial is doing the work.
        let (_, vectors) = generate(&SynthConfig { anisotropy: 0.0, topics: 64, ..cfg(2_000) });
        let anchor = vera_index::kmeans::domain_anchor(&vectors);
        let mean: f32 = vectors
            .iter_rows()
            .map(|r| r.iter().zip(&anchor).map(|(a, b)| a * b).sum::<f32>())
            .sum::<f32>()
            / vectors.rows() as f32;
        assert!(mean < 0.3, "expected a weak anchor, got {mean}");
    }

    #[test]
    fn some_rows_carry_a_citable_identifier() {
        let (rows, _) = generate(&cfg(500));
        let with_id = rows.iter().filter(|r| r.identifier.is_some()).count();
        assert!(with_id > 0 && with_id < 500, "{with_id}");
    }

    #[test]
    fn a_query_near_a_row_is_closer_to_it_than_to_a_random_row() {
        let (_, vectors) = generate(&cfg(300));
        let mut rng = Rng::new(3);
        let q = query_near(vectors.row(0), 0.05, &mut rng);
        let to_source: f32 = q.iter().zip(vectors.row(0)).map(|(a, b)| a * b).sum();
        let to_other: f32 = q.iter().zip(vectors.row(299)).map(|(a, b)| a * b).sum();
        assert!(to_source > to_other, "{to_source} vs {to_other}");
    }
}
