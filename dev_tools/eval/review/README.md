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
