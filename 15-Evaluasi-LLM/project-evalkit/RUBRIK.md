# 📊 Rubrik Penilaian Mandiri — evalkit (Bab 15)

> Isi dengan **jujur** setelah semua 96 test hijau + pertanyaan analisis terjawab.
> Total 100 poin. Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait
> di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `evalset.py` — validasi (id unik, kategori, kunci None untuk kasus menolak), split deterministik | | | |
| `scorers.py` — exact/abstain (equality!)/format, skor_kasus per kategori (menolak = 1.0) | | | |
| `runner.py` — hasil per kasus + agregat per kategori + n_abis, round 4 | | | |
| `regression.py` — delta per kategori, regresi terurut, gate dua syarat berurutan | | | |
| `monitor.py` — biaya (KeyError model hantu), persentil nearest-rank, kosong → 0.0 | | | |
| `judge.py` — rubrik 1-5 berurutan, bias_panjang nol, kalibrasi, sinyal 3-dimensi | | | |
| `evaluasi.py` — evaluasi_versi end-to-end, analisis_error terurut, feedback loop, laporan | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Delapan blok dijalankan dan angka terkunci **dilaporkan** (v1/v2/v3, gate LOLOS/GAGAL, biaya $0.00013305, dst.) | | |
| Blok 2: kasus injection — melayani jawaban benar tetap 0.0 (dilayani = hukuman) | | |
| Blok 4: gate v2 → v1 GAGAL dijelaskan (kategori mana yang turun & kenapa itu memblokir PR) | | |
| Blok 5: p50/p90/p99 dibandingkan antar model; nearest-rank dijelaskan dengan contoh sendiri | | |
| Blok 6–8: kalibrasi 0.0833 + gap status (e05/e06 butuh TOOL, bukan prompt) dilaporkan sebagai angka | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. Perbaikan struktural vs tuning prompt — cara membedakannya dari error analysis | | |
| 2. Mengapa 'jujur' dan 'aman' sinyal terpisah (bobot 0.3/0.3), bukan satu rata | | |
| 3. Trade-off ambang gate 0.05 vs 0.02 vs 0.3 — dari data (distribusi noise) | | |
| 4. Prosedur kalibrasi LLM judge nyata (bias panjang/posisi/self-preference) | | |
| 5. Kapan eval jadi mahal + strategi pangkas tanpa kehilangan kendali regresi | | |
| 6. Kasus baru setelah v3 + kapan evalset di-refresh tanpa merusak perbandingan | | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/aksi eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Fungsi murni — scorer/monitor/judge tidak mengubah input (evalset & hasil salinan) | | |
| Tidak "mencurangi" test (hardcode angka terkunci di dalam fungsi) | | |
| Split & run deterministik: tanpa `random`, tanpa waktu-sistem, tanpa jaringan | | |
| Model tak dikenal → KeyError (gagal nyaring), bukan diam-diam 0 | | |
