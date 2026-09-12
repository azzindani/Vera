"""Seed a tiny, deterministic corpus so the store's SQL can be tested anywhere.

The eight integration tests in `crates/store/tests/it_live.rs` need a database
with a loaded corpus. Until now that meant the 181K-chunk spike in `.test/`,
which exists on exactly one machine — so CI ran none of them, and the SQL layer
was the least-tested part of the system. That is where the `plainto_tsquery`
bug lived: it ANDed every term of a question and matched nothing corpus-wide,
and no test noticed for as long as it took a human to ask a real question.

This builds ~30 chunks of real Indonesian legal phrasing with synthetic — but
deterministic — vectors. The vectors are noise around four theme directions, so
cluster membership is genuine and `dense_in_cluster` is really pruning; they
say nothing about embedding quality, and nothing here should be read as an
eval. This fixture tests SQL semantics, filters and the pruning guarantee.
Retrieval quality is measured against the real corpus by `eval/run.py`.

    python fixtures/seed.py                  # DATABASE_URL or the dev default
    python fixtures/seed.py --force          # overwrite a populated database

Safety: refuses to touch a database that already holds a real corpus unless
`--force` is given. Wiping 181K embedded rows costs a GPU-day to rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines" / "pre_embed"))
from sparse import Bm25Vectorizer  # noqa: E402

SCHEMA_SQL = ROOT / "pipelines" / "pre_embed" / "schema.sql"

CORPUS_ID = "fixture-01"
DENSE_DIM = 1024  # must match the engine's pin; it_live.rs asserts on it
SPARSE_DIM = 20_000  # production width, so sparse_literal indices stay valid
SEED = 20260912

# Four theme directions. Documents on a theme cluster together, which is what
# makes `dense_in_cluster` a real test of pruning rather than a tautology.
THEMES = ["energi", "transportasi", "administrasi", "elektronik"]

# ---------------------------------------------------------------------------
# The corpus. Real phrasing from real regulation families, hand-written short.
#
# ! PERATURAN BUPATI 27/2013 appears TWICE with different `about`. That is not
# a mistake: in the real corpus that identifier covers 22 distinct regulations
# from 22 regencies. A test that keys documents on (type, number, year) is
# keying on something that is not identity, and this row is here to keep that
# lesson executable.
# ---------------------------------------------------------------------------
DOCS = [
    {
        "type": "PERATURAN PEMERINTAH", "number": "26", "year": 2009,
        "about": "SANKSI ADMINISTRASI BERUPA DENDA DI BIDANG CUKAI",
        "theme": "administrasi",
        "clauses": [
            ("BAB I", "Pasal 1",
             "Dalam Peraturan Pemerintah ini yang dimaksud dengan cukai adalah "
             "pungutan negara yang dikenakan terhadap barang tertentu yang "
             "mempunyai sifat atau karakteristik yang ditetapkan dalam undang-undang."),
            ("BAB II", "Pasal 3",
             "Sanksi administrasi berupa denda dikenakan terhadap pengusaha pabrik "
             "yang tidak menyampaikan pemberitahuan barang kena cukai sesuai dengan "
             "ketentuan peraturan perundang-undangan."),
            ("BAB V", "Pasal 37",
             "Dalam hal pelanggaran dilakukan untuk pertama kali, sanksi administrasi "
             "berupa denda ditetapkan paling sedikit dua kali nilai cukai dan paling "
             "banyak sepuluh kali nilai cukai yang seharusnya dibayar."),
            ("BAB V", "Pasal 40",
             "Pembayaran denda sebagaimana dimaksud dalam Pasal 37 dilakukan paling "
             "lama tiga puluh hari terhitung sejak tanggal penetapan diterima oleh "
             "yang bersangkutan."),
        ],
    },
    {
        "type": "UNDANG-UNDANG", "number": "30", "year": 2007, "about": "ENERGI",
        "theme": "energi",
        "clauses": [
            ("BAB II", "Pasal 2",
             "Energi dikelola berdasarkan asas kemanfaatan, rasionalitas, efisiensi "
             "berkeadilan, peningkatan nilai tambah, keberlanjutan, kesejahteraan "
             "masyarakat, pelestarian fungsi lingkungan hidup, dan ketahanan nasional."),
            ("BAB III", "Pasal 8",
             "Setiap kegiatan pengelolaan energi wajib mengutamakan penggunaan "
             "teknologi yang ramah lingkungan dan memperhatikan kelestarian fungsi "
             "lingkungan hidup."),
            ("BAB IV", "Pasal 20",
             "Penyediaan energi baru dan energi terbarukan wajib ditingkatkan oleh "
             "Pemerintah dan pemerintah daerah sesuai dengan kewenangannya."),
        ],
    },
    {
        "type": "UNDANG-UNDANG", "number": "4", "year": 2009,
        "about": "PERTAMBANGAN MINERAL DAN BATUBARA", "theme": "energi",
        "clauses": [
            ("BAB III", "Pasal 5",
             "Untuk kepentingan nasional, Pemerintah setelah berkonsultasi dengan "
             "Dewan Perwakilan Rakyat Republik Indonesia dapat menetapkan kebijakan "
             "pengutamaan mineral dan batubara untuk kepentingan dalam negeri."),
            ("BAB III", "Pasal 6",
             "Pemerintah dapat melakukan pembatasan produksi dan ekspor mineral dan "
             "batubara untuk memenuhi kebutuhan dalam negeri."),
            ("BAB XII", "Pasal 96",
             "Pemegang izin usaha pertambangan wajib melaksanakan pengelolaan dan "
             "pemantauan lingkungan pertambangan, termasuk kegiatan reklamasi dan "
             "pascatambang."),
        ],
    },
    {
        "type": "PERATURAN PEMERINTAH", "number": "79", "year": 2013,
        "about": "JARINGAN LALU LINTAS DAN ANGKUTAN JALAN", "theme": "transportasi",
        "clauses": [
            # ! The OR-semantics regression target. A question phrased as a person
            # asks it ("siapa yang berwenang menetapkan kelas jalan provinsi")
            # shares only four of its six lexemes with this clause, so an ANDing
            # tsquery returns nothing at all. See it_live.rs.
            ("BAB IV", "Pasal 18",
             "Penetapan kelas jalan pada jalan provinsi dilakukan oleh gubernur "
             "setelah mendapat pertimbangan dari instansi terkait."),
            ("BAB IV", "Pasal 19",
             "Kelas jalan sebagaimana dimaksud dalam Pasal 18 dinyatakan dengan "
             "rambu lalu lintas yang dipasang pada setiap ruas jalan."),
            ("BAB V", "Pasal 24",
             "Rencana induk jaringan lalu lintas dan angkutan jalan kabupaten "
             "disusun dengan memperhatikan rencana tata ruang wilayah kabupaten."),
        ],
    },
    {
        "type": "UNDANG-UNDANG", "number": "22", "year": 2009,
        "about": "LALU LINTAS DAN ANGKUTAN JALAN", "theme": "transportasi",
        "clauses": [
            ("BAB V", "Pasal 19",
             "Jalan dikelompokkan dalam beberapa kelas berdasarkan fungsi dan "
             "intensitas lalu lintas serta daya dukung menerima muatan sumbu "
             "terberat kendaraan bermotor."),
            ("BAB V", "Pasal 20",
             "Penetapan kelas jalan pada setiap ruas jalan dilakukan oleh "
             "penyelenggara jalan sesuai dengan kewenangannya."),
            ("BAB IX", "Pasal 106",
             "Setiap orang yang mengemudikan kendaraan bermotor di jalan wajib "
             "mengemudikan kendaraannya dengan wajar dan penuh konsentrasi."),
        ],
    },
    {
        "type": "PERATURAN PRESIDEN", "number": "73", "year": 2011,
        "about": "PEMBANGUNAN BANGUNAN GEDUNG NEGARA", "theme": "administrasi",
        "clauses": [
            ("BAB II", "Pasal 3",
             "Pembangunan bangunan gedung negara harus memenuhi persyaratan "
             "administratif meliputi dokumen pembiayaan, status hak atas tanah, "
             "dan dokumen perencanaan teknis."),
            ("BAB II", "Pasal 5",
             "Setiap bangunan gedung negara harus diwujudkan dengan sebaik-baiknya "
             "sehingga mampu memenuhi secara optimal fungsi bangunan."),
        ],
    },
    {
        "type": "PERATURAN PEMERINTAH", "number": "82", "year": 2012,
        "about": "PENYELENGGARAAN SISTEM DAN TRANSAKSI ELEKTRONIK",
        "theme": "elektronik",
        "clauses": [
            ("BAB IV", "Pasal 15",
             "Penyelenggara sistem elektronik wajib menjaga kerahasiaan, keutuhan, "
             "dan ketersediaan data pribadi yang dikelolanya."),
            ("BAB VII", "Pasal 58",
             "Penyelenggara sertifikasi elektronik wajib melakukan pemeriksaan dan "
             "pemastian identitas pemohon sebelum menerbitkan sertifikat elektronik."),
            ("BAB VII", "Pasal 59",
             "Tanda tangan elektronik meliputi tanda tangan elektronik tersertifikasi "
             "dan tanda tangan elektronik tidak tersertifikasi."),
        ],
    },
    {
        "type": "UNDANG-UNDANG", "number": "15", "year": 2011,
        "about": "PENYELENGGARA PEMILIHAN UMUM", "theme": "administrasi",
        "clauses": [
            ("BAB II", "Pasal 35",
             "Undangan rapat pleno beserta bahan rapat disampaikan kepada anggota "
             "paling lambat tiga hari sebelum rapat pleno diselenggarakan."),
            ("BAB II", "Pasal 36",
             "Rapat pleno sah apabila dihadiri oleh sekurang-kurangnya lima orang "
             "anggota yang dibuktikan dengan daftar hadir."),
        ],
    },
    {
        "type": "PERATURAN PEMERINTAH", "number": "24", "year": 2010,
        "about": "PENGGUNAAN KAWASAN HUTAN", "theme": "energi",
        "clauses": [
            ("BAB II", "Pasal 4",
             "Penggunaan kawasan hutan untuk kepentingan pembangunan di luar "
             "kegiatan kehutanan hanya dapat dilakukan di dalam kawasan hutan "
             "produksi dan kawasan hutan lindung."),
            ("BAB VII", "Pasal 25",
             "Izin pinjam pakai kawasan hutan yang telah diterbitkan sebelum "
             "berlakunya Peraturan Pemerintah ini tetap berlaku sampai dengan "
             "berakhirnya jangka waktu izin."),
        ],
    },
    {
        "type": "PERATURAN PEMERINTAH", "number": "62", "year": 2009,
        "about": "HAK KEUANGAN DAN FASILITAS ANGGOTA KOMISI YUDISIAL",
        "theme": "administrasi",
        "clauses": [
            ("BAB I", "Pasal 1",
             "Hak keuangan dan fasilitas bagi Ketua, Wakil Ketua, dan Anggota "
             "Komisi Yudisial disamakan dengan hak keuangan dan fasilitas yang "
             "diberikan kepada Ketua, Wakil Ketua, dan Hakim Agung pada Mahkamah Agung."),
            ("BAB I", "Pasal 2",
             "Hak keuangan sebagaimana dimaksud dalam Pasal 1 dibebankan pada "
             "Anggaran Pendapatan dan Belanja Negara."),
        ],
    },
    {
        "type": "PERATURAN PRESIDEN", "number": "79", "year": 2011,
        "about": "KUNJUNGAN KAPAL WISATA (YACHT) ASING", "theme": "transportasi",
        "clauses": [
            ("BAB IV", "Pasal 14",
             "Pelayanan kedatangan kapal wisata asing dilaksanakan secara terpadu "
             "oleh kementerian yang menyelenggarakan urusan pemerintahan di bidang "
             "perhubungan, keuangan, hukum dan hak asasi manusia, pertahanan, "
             "serta pariwisata."),
            ("BAB IV", "Pasal 15",
             "Permohonan kedatangan kapal wisata asing diajukan secara elektronik "
             "paling lambat tujuh hari sebelum tanggal kedatangan."),
        ],
    },
    {
        "type": "PERATURAN BUPATI", "number": "27", "year": 2013,
        "about": "RETRIBUSI PELAYANAN PASAR KABUPATEN SLEMAN",
        "theme": "administrasi",
        "clauses": [
            ("BAB II", "Pasal 2",
             "Retribusi pelayanan pasar dipungut atas penyediaan fasilitas pasar "
             "tradisional yang dikelola oleh Pemerintah Daerah."),
        ],
    },
    {
        "type": "PERATURAN BUPATI", "number": "27", "year": 2013,
        "about": "IZIN MENDIRIKAN BANGUNAN KABUPATEN BANTUL",
        "theme": "administrasi",
        "clauses": [
            ("BAB II", "Pasal 2",
             "Setiap orang yang akan mendirikan bangunan gedung di daerah wajib "
             "memiliki izin mendirikan bangunan yang diterbitkan oleh Bupati."),
        ],
    },
]

# ! One deliberately unindexable row and one truncated row. Honest flags are
# set at ingest and must survive into every filter; a corpus of only clean rows
# cannot prove that `WHERE indexable` is doing anything.
UNINDEXABLE = {
    "type": "PERATURAN PEMERINTAH", "number": "26", "year": 2009,
    "about": "SANKSI ADMINISTRASI BERUPA DENDA DI BIDANG CUKAI",
    "theme": "administrasi",
    "chapter": "LAMPIRAN", "article": "Lampiran I",
    "body": "sanksi administrasi denda cukai lampiran tabel tabel tabel "
            "angka angka angka kolom kolom baris baris",
}
TRUNCATED_AT = 2  # index of the clause marked truncated_at_source


def unit(rng: random.Random, dim: int) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def near(rng: random.Random, base: list[float], jitter: float) -> list[float]:
    v = [b + rng.gauss(0.0, jitter) for b in base]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def doc_key(about: str) -> str:
    """Stable short suffix distinguishing same-identifier regulations."""
    return hashlib.sha1(about.encode("utf-8")).hexdigest()[:4]


def build_rows() -> list[dict]:
    """Flatten DOCS into chunk rows with themes attached."""
    rows: list[dict] = []
    for doc in DOCS:
        for i, (chapter, article, body) in enumerate(doc["clauses"]):
            rows.append({
                # ! sha1, not hash(): Python randomises str hashing per
                # process, so ids would differ between the seed run and any
                # later re-seed. A fixture that is not reproducible is a trap.
                "id": f"fx-{doc['type'][:2]}-{doc['number']}-{doc['year']}-{i:02d}"
                      f"-{doc_key(doc['about'])}",
                "type": doc["type"], "number": doc["number"], "year": doc["year"],
                "about": doc["about"], "theme": doc["theme"],
                "chapter": chapter, "article": article, "chunk_no": i,
                "body": body,
                "indexable": True,
                "truncated": i == TRUNCATED_AT,
            })
    u = UNINDEXABLE
    rows.append({
        "id": "fx-PE-26-2009-99-lampiran",
        "type": u["type"], "number": u["number"], "year": u["year"],
        "about": u["about"], "theme": u["theme"],
        "chapter": u["chapter"], "article": u["article"], "chunk_no": 99,
        "body": u["body"], "indexable": False, "truncated": False,
    })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if the database already holds a corpus")
    ap.add_argument("--vocab-out", default=str(Path(__file__).parent / "seed.bm25.json"))
    args = ap.parse_args()

    import psycopg

    url = os.environ.get(
        "DATABASE_URL", "host=localhost port=5432 dbname=vera user=vera password=vera"
    )
    rows = build_rows()
    rng = random.Random(SEED)

    pg = psycopg.connect(url)
    cur = pg.cursor()

    # -- safety ------------------------------------------------------------
    # ! Refuse to drop a real corpus. `.test/` holds 181K rows that cost a
    # GPU-day; a fixture seed must never be the thing that deletes them.
    cur.execute("SELECT to_regclass('public.chunks') IS NOT NULL")
    if cur.fetchone()[0]:
        cur.execute("SELECT count(*) FROM chunks")
        existing = cur.fetchone()[0]
        if existing > len(rows) and not args.force:
            print(f"refusing: {existing:,} chunks already here (fixture is "
                  f"{len(rows)}). Pass --force if you really mean it.", file=sys.stderr)
            return 1

    cur.execute("DROP TABLE IF EXISTS clusters, ingest_progress, chunks, corpus_meta")
    schema = (SCHEMA_SQL.read_text(encoding="utf-8")
              .replace("{{DENSE_DIM}}", str(DENSE_DIM))
              .replace("{{SPARSE_DIM}}", str(SPARSE_DIM)))
    cur.execute(schema)
    cur.execute(f"""
        CREATE TABLE clusters (
            id         INTEGER PRIMARY KEY,
            corpus_id  TEXT NOT NULL REFERENCES corpus_meta(id),
            centroid   halfvec({DENSE_DIM}) NOT NULL,
            row_count  BIGINT NOT NULL DEFAULT 0,
            generation INTEGER NOT NULL DEFAULT 1
        )""")

    # -- sparse side: a real BM25 fit over the fixture bodies ---------------
    # Fit is genuine so the sparse vectors correspond to the text, but the
    # declared width is the production 20,000 -- narrowing it would make
    # `sparse_literal(&[(10,..),(500,..)])` reference an out-of-range index.
    vz = Bm25Vectorizer.fit([r["body"] for r in rows])
    vz.dim = SPARSE_DIM
    vocab_sha = vz.save(args.vocab_out)

    cur.execute(
        """INSERT INTO corpus_meta (
               id, run_id, dense_model, dense_dim, dense_pooling, dense_normalize,
               dense_dtype, dense_instruction, tokenizer_sha256,
               sparse_scheme, sparse_dim, sparse_k1, sparse_b,
               sparse_vocab_sha256, sparse_fit_docs, source_db,
               manifest_sha256, notes)
           VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (CORPUS_ID, "fixture", "qwen/qwen3-embedding-0.6b", DENSE_DIM,
         "last-token", True, "float16",
         hashlib.sha256(b"fixture-tokenizer").hexdigest(),
         "bm25", SPARSE_DIM, vz.k1, vz.b, vocab_sha, vz.n_docs, "fixtures/seed.py",
         hashlib.sha256(str(len(rows)).encode()).hexdigest(),
         "Synthetic vectors around four theme directions. Tests SQL semantics "
         "and cluster pruning -- NOT retrieval quality. See fixtures/seed.py."),
    )

    # -- dense side: theme directions, then members near them ---------------
    theme_vecs = {t: unit(rng, DENSE_DIM) for t in THEMES}
    cluster_of = {t: i for i, t in enumerate(THEMES)}

    payload = []
    for r in rows:
        dv = near(rng, theme_vecs[r["theme"]], 0.05)
        payload.append((
            r["id"], CORPUS_ID, r["type"], r["number"], r["year"], r["about"],
            r["chapter"], r["article"], r["chunk_no"], r["body"],
            None,  # source_url: this corpus genuinely has none (invariant 8)
            f"{r['type']} Nomor {r['number']} Tahun {r['year']} tentang {r['about']}",
            r["truncated"], r["indexable"], cluster_of[r["theme"]],
            literal(dv), vz.to_sparsevec(vz.document(r["body"])),
        ))

    cur.executemany(
        """INSERT INTO chunks (
               id, corpus_id, regulation_type, regulation_number, year, about,
               chapter, article, chunk_no, body, source_url, source_title,
               truncated_at_source, indexable, cluster_id, dense, sparse)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                   %s::halfvec,%s::sparsevec)""",
        payload,
    )

    # Centroid = normalized mean of the cluster's members, exactly as spherical
    # k-means leaves it, so probing with a centroid really does surface members.
    for theme, cid in cluster_of.items():
        members = [r for r in rows if r["theme"] == theme]
        cur.execute(
            "SELECT avg(dense::vector)::text FROM chunks WHERE cluster_id = %s", (cid,)
        )
        mean = [float(x) for x in cur.fetchone()[0].strip("[]").split(",")]
        n = math.sqrt(sum(x * x for x in mean)) or 1.0
        cur.execute(
            "INSERT INTO clusters (id, corpus_id, centroid, row_count)"
            " VALUES (%s,%s,%s::halfvec,%s)",
            (cid, CORPUS_ID, literal([x / n for x in mean]), len(members)),
        )

    cur.executemany(
        "INSERT INTO ingest_progress (chunk_id, run_id, corpus_id, stage)"
        " VALUES (%s,'fixture',%s,'done')",
        [(r["id"], CORPUS_ID) for r in rows],
    )
    pg.commit()

    cur.execute("SELECT count(*), count(*) FILTER (WHERE indexable) FROM chunks")
    total, indexable = cur.fetchone()
    print(f"seeded {total} chunks ({indexable} indexable) in {len(THEMES)} clusters")
    print(f"vocab  {vz.n_docs} docs, {len(vz.vocab)} terms, sha {vocab_sha[:12]}")
    pg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
