# Review label evaluasi · untuk peninjau hukum

`review-001.csv` berisi 120 pertanyaan beserta pasal yang dianggap sebagai jawabannya.
Mesin pencari Vera dinilai dengan label ini, jadi label yang salah membuat penilaian salah.
Buka berkas ini di Excel.

Untuk setiap baris, baca `query` (pertanyaan) dan `clause_text` (teks pasal dari korpus),
lalu isi kolom `verdict` dengan salah satu:

| verdict | artinya |
|---|---|
| `BENAR` | pasal ini menjawab pertanyaan dengan benar dan lengkap |
| `SEBAGIAN` | hanya menjawab sebagian, atau perlu pasal lain untuk menjawab |
| `SALAH` | pasal ini tidak menjawab pertanyaan |
| `AMBIGU` | banyak peraturan lain menjawab sama baiknya, sehingga label ini tidak adil |

Kolom `note` boleh diisi alasan singkat, terutama untuk SEBAGIAN, SALAH dan AMBIGU.
Untuk `multi_tier`, kedua pasal harus menjawab. Untuk `underspecified`, pertanyaan memang
sengaja umum, dan setiap pasal yang tercantum harus menjawabnya.

Setelah selesai, simpan sebagai CSV (UTF-8) dan jalankan:

```
python -I dev_tools/eval/draft/review.py import --sheet dev_tools/eval/review/review-001.csv --reviewer "<nama atau peran>"
```

## `review-002.csv` · jawaban lain dari mesin

Berisi 211 pertanyaan yang **tidak** dijawab Vera dengan pasal berlabel di lima hasil
teratas. Setiap baris menampilkan pasal berlabel (`labelled_answer`, `labelled_text`) dan
satu pasal yang dikembalikan mesin (`engine_answer`, `engine_text`, peringkat
`engine_rank`). Peraturan Indonesia sering mengulang pasal yang sama di peraturan lain
(PERDA kabupaten yang berbeda, atau peraturan lama dan penggantinya), sehingga hasil mesin
bisa saja juga benar.

Kolom `kind`: `MIRIP` berarti teks pasal mesin sangat mirip dengan pasal berlabel
(`similarity` mendekati 1); `TERATAS` berarti hasil pertama mesin. Baris `MIRIP` ada di atas.

Isi kolom `verdict`:

| verdict | artinya |
|---|---|
| `JUGA_BENAR` | pasal dari mesin menjawab pertanyaan sama baiknya dengan pasal berlabel |
| `SEBAGIAN` | hanya menjawab sebagian |
| `TIDAK` | tidak menjawab pertanyaan (misalnya daerah atau peraturan yang ditanyakan berbeda) |

Perhatikan bila pertanyaan menyebut daerah, instansi atau peraturan tertentu: pasal yang
sama dari daerah lain **tidak** menjawabnya. Kolom `candidate_chunk` jangan diubah.

Setelah selesai, simpan sebagai CSV (UTF-8) dan jalankan:

```
python -I dev_tools/eval/draft/alternatives.py import --dsn "<dsn>" --sheet dev_tools/eval/review/review-002.csv --reviewer "<nama atau peran>"
```

Label asli tidak diubah. Jawaban `JUGA_BENAR` disimpan terpisah dan hanya dihitung bila
`EVAL_ALTERNATIVES=1`, sehingga nilai terhadap label asli tetap bisa dilaporkan.
