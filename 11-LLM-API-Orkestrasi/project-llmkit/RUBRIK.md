# 📊 Rubrik Penilaian Mandiri — llmkit (Bab 11)

> Isi dengan **jujur** setelah semua 107 test hijau + pertanyaan analisis terjawab.
> Total 100 poin. Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait
> di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `cost.py` — token, biaya per 1 juta token, penjaga anggaran yang melewati | | | |
| `retry.py` — transien vs fatal, backoff terbatas, **tidak menunggu** di akhir | | | |
| `fallback.py` — rantai berurutan, `dicoba`/`lompatan`, `settings` per tautan | | | |
| `cache.py` — kunci lengkap, TTL tick logis, `expired` ≠ `miss` | | | |
| `stream.py` — TTFT, jumlah delta non-kosong, `total_ms` chunk terakhir | | | |
| `tools.py` — skema dari contoh, error stabil, validasi **sebelum** handler | | | |
| `orchestrate.py` — laporan per langkah (biaya, ms, attempts, galat) | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Ketujuh blok dijalankan dan angkanya **dilaporkan** (bukan hanya "tidak error") | | |
| Riwayat retry + jadwal backoff ditampilkan, termasuk kasus fatal dan mentok | | |
| Fallback dilaporkan dengan `dicoba` + `lompatan` (bukan hanya "berhasil") | | |
| Cache: pola hit/miss/expired + jumlah panggilan provider dibanding jumlah permintaan | | |
| Pipeline dibandingkan `maks_percobaan=3` vs `1` (kegagalan terlokalisasi terbukti) | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. Transien vs fatal — biaya + pengalaman pengguna | | |
| 2. Kenapa backoff naik perlahan + kenapa ada batas atas | | |
| 3. Harga fallback & kapan error 503 lebih jujur | | |
| 4. Dua hal wajib tambahan pada kunci cache + risikonya | | |
| 5. Streaming: apa yang benar-benar berubah & kapan tidak layak | | |
| 6. Risiko mengeksekusi argumen model + least-privilege | | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/aksi eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Fungsi murni — input (mis. `messages`, `rantai`) tidak dimodifikasi | | |
| Tidak ada state global; provider baru saat urutan kegagalan harus diulang | | |
| Tidak "mencurangi" test (hardcode angka terkunci di dalam fungsi) | | |
| `sleep_fn` disuntik — tidak ada `time.sleep` nyata di jalur test | | |
| Penanganan input kosong (`messages` kosong, rantai kosong, `chunks` kosong) tidak crash | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

- Test mana yang paling lama kamu perbaiki (urutan `sleep` vs cek percobaan
  terakhir? `expired` vs `miss`? de-dup tool dari `tool_calls`?), dan apa yang
  akhirnya kamu pahami?
- Setelah melihat retry menambah 1000 ms ke latency pengguna, bagaimana itu
  mengubah cara kamu memilih `maks_percobaan` dan `maks_ms`?
- Satu hal yang mau kamu bawa ke Bab 12 (RAG): cache yang sadar konteks retrieval,
  anggaran per-tenant, atau evaluasi otomatis di CI?
