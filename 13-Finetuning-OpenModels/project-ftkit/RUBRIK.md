# 📊 Rubrik Penilaian Mandiri — ftkit (Bab 13)

> Isi dengan **jujur** setelah semua 44 test hijau + pertanyaan analisis terjawab.
> Total 100 poin. Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait
> di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `sft.py` — format prompt, round-trip JSONL, oversample urutan asli, split interleaved | | | |
| `lora.py` — ΔW = B@A, B nol → delta 0, param = rank×(in+out), gradient check < 1e-6 | | | |
| `quantize.py` — skala simetris, clamp, bit≥32 identik, skala-0 aman | | | |
| `trainer.py` — loss sebelum update, restore bobot TERBAIK, berhenti dini | | | |
| `api.py` — tarif per 1jt, 'lokal' = 0, keputusan API vs GPU | | | |
| `evaluasi.py` — sebelum/sesudah + delta, OOD → jangan tampilkan, dict baru | | | |
| `data.py`/`tokenizer.py`/`model.py` TIDAK diubah (kontrak GIVEN terjaga) | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Delapan blok dijalankan dan angka terkunci **dilaporkan** (loss 0.7396→0.0872, dst.) | | |
| Blok 2: analisis rasio 0,78% — implikasi multi-adapter per klien | | |
| Blok 5: val loss naik setelah epoch 61 dijelaskan (overfit data kontradiktif) | | |
| Blok 6: galat kuantisasi per bit dilaporkan, bukan hanya ukuran | | |
| Blok 7–8: ambang API↔lokal + skor OOD 0.0024 dilaporkan | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. LoRA 0,78% — biaya produksi multi-adapter + kapan rank naik | | |
| 2. Early stopping: val loss naik pasca 61; restore best vs last | | |
| 3. Risiko oversampling berlebihan + alternatifnya | | |
| 4. Kapan Q4 layak / tidak layak | | |
| 5. Faktor yang menggeser ambang API↔lokal | | |
| 6. Ketidakpastian di LLM nyata + lapisan wajib yang bertahan | | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/aksi eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Fungsi murni — `oversample`/`kebijakan_ood` tidak mengubah input | | |
| Tidak ada `random` untuk keputusan dataset (split & oversample deterministik) | | |
| Tidak "mencurangi" test (hardcode angka terkunci di dalam fungsi) | | |
| Loss & akurasi dicatat per epoch ke riwayat (bisa diplot/diaudit) | | |
