# 📊 Rubrik Penilaian Mandiri — agentkit (Bab 14)

> Isi dengan **jujur** setelah semua 57 test hijau + pertanyaan analisis terjawab.
> Total 100 poin. Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait
> di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `tools.py` — validasi skema (required/type/pola/min_panjang), PII dibuang, error = observasi | | | |
| `loop_react.py` — format trace terkunci, langkah final, guardrail iterasi + escallate | | | |
| `guardrail.py` — aksi berisiko persis, sanitasi case-insensitive, rekursif, fungsi murni | | | |
| `memory.py` — konteks urutan asli, ringkasan, cache ternormalisasi + hit/miss | | | |
| `agent.py` — jawab end-to-end, planner statis = bertahap, evaluasi trace 1.0/0.5/0.0 | | | |
| `data.py`/`nlu.py` TIDAK diubah (kontrak GIVEN terjaga) | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Delapan blok dijalankan dan angka terkunci **dilaporkan** (2 langkah status, skor 1.0/0.5/0.0, dst.) | | |
| Blok 2: analisis PII — kunci 'user' dibuang di tool, bukan di prompt | | |
| Blok 4: variasi payload injection diuji sendiri (baris baru, huruf campur) | | |
| Blok 5: trade-off maks_iter kecil vs besar dijelaskan dengan eksperimen | | |
| Blok 6–8: hit/miss cache + skor trace dilaporkan sebagai angka, bukan kesan | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. Aturan vs LLM tool-calling — apa yang tetap sama di mesin agent | | |
| 2. PII di sistem nyata + lapisan pembuangan yang tepat | | |
| 3. Trade-off maks_iter terlalu kecil vs terlalu besar | | |
| 4. Kapan planner bertahap (loop LLM) dibutuhkan + biayanya | | |
| 5. Kunci cache lengkap agar konteks tidak basi (Bab 12) | | |
| 6. Kenapa tanya_balik (0.5) > error (0.0), dan kapan 0.5 masih kurang | | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/aksi eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Fungsi murni — `sanitasi`/`amankan_observasi`/`oversample`-gaya tidak mengubah input | | |
| Error tool tidak pernah melempar exception keluar dari `eksekusi_tool` | | |
| Tidak "mencurangi" test (hardcode angka terkunci di dalam fungsi) | | |
| Trace bisa direproduksi: tanpa `random`, tanpa waktu-sistem, tanpa jaringan | | |
