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

/// Indonesian-regulation-flavored vocabulary · gives BM25 something real to
/// score and produces queries that overlap bodies the way genuine ones do.
const VOCAB: &[&str] = &[
    "wajib", "pajak", "sanksi", "administrasi", "ketentuan", "umum", "tata", "cara",
    "perpajakan", "penghasilan", "badan", "orang", "pribadi", "tarif", "pengenaan",
    "pemungutan", "penyetoran", "pelaporan", "keberatan", "banding", "surat", "teguran",
    "denda", "bunga", "kenaikan", "restitusi", "kompensasi", "pemeriksaan", "penyidikan",
    "daerah", "pusat", "menteri", "keuangan", "direktorat", "jenderal", "peraturan",
    "pelaksanaan", "perubahan", "pencabutan", "berlaku", "kewajiban", "hak", "subjek",
    "objek", "dasar", "pengurangan", "fasilitas", "insentif", "pembukuan", "pencatatan",
];

const DOC_TYPES: &[&str] = &["UU", "PP", "PERPRES", "PMK", "PERMEN"];

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

    let mut vectors = Matrix::with_capacity(cfg.dim, cfg.rows);
    let mut rows = Vec::with_capacity(cfg.rows);

    for i in 0..cfg.rows {
        let topic = rng.below(cfg.topics.max(1));
        let center = centers.row(topic);

        let noise = random_unit(cfg.dim, &mut rng);
        vectors.push(&mix(center, &noise, 1.0 - cfg.spread));

        // Body words drawn from a topic-biased slice of the vocabulary, so
        // keyword and dense signals correlate the way they do in real text.
        let base = topic * 7 % VOCAB.len();
        let body: Vec<&str> = (0..18)
            .map(|w| {
                let pick = if rng.unit() < 0.6 {
                    (base + w * 3) % VOCAB.len()
                } else {
                    rng.below(VOCAB.len())
                };
                VOCAB[pick]
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
            body: format!("{} nomor {} tentang {}", body.join(" "), i % 500, VOCAB[base]),
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

    fn cfg(rows: usize) -> SynthConfig {
        SynthConfig {
            rows,
            dim: 32,
            topics: 4,
            spread: 0.2,
            anisotropy: 0.7,
            seed: 7,
        }
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
