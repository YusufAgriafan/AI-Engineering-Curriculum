# 📊 Rubrik Penilaian Mandiri — Lembar Prompt (Bab 10)

> Isi dengan **jujur** setelah semua 157 test hijau + pertanyaan analisis terjawab.
> Total 100 poin. Nilai < 80? Tandai bagian yang lemah, ulangi eksperimen terkait
> di `starter.ipynb`.

## A. Kode Berjalan (35 poin)

| Item | ✅/❌ | Skor (0–7) | Catatan |
|---|---|---|---|
| `sections.py` — bagian wajib terdeteksi, CONTOH opsional, anggaran token | | | |
| `template.py` — `{{}}` single-pass, kurung JSON utuh, escape delimiter | | | |
| `fewshot.py` — pemilihan merata, urutan recency, format bisa diaudit | | | |
| `schema.py` — error stabil, `bool` bukan `integer`, batas dilewati saat null | | | |
| `parse.py` — brace-matching sadar string, perbaikan JSON umum, ValueError jelas | | | |
| `guard.py` — pola dari `data.POLA_INJEKSI`, escape sebelum bungkus | | | |
| `evaluate.py` — lima metrik terpisah, `bandingkan` deterministik | | | |

## B. Kualitas Eksperimen (25 poin)

| Kriteria | Skor (0–5) | Bukti |
|---|---|---|
| Tangga tiga prompt dilaporkan dengan KELIMA metrik (bukan hanya akurasi) | | |
| Kegagalan v1 dipecah per kasus + per field (`salah: [...]`) | | |
| Ablasi few-shot k=0..3 dengan variabel lain terkunci | | |
| Injeksi diuji di ≥2 prompt, termasuk uji "delimiter saja" & "leash saja" | | |
| Temperature sweep (0.0/0.5/0.9) dengan pemisahan format vs isi | | |

## C. Pertanyaan Analisis (30 poin)

| Pertanyaan | Skor (0–5) |
|---|---|
| 1. Apa yang membuat prompt bisa dieksekusi mesin | |
| 2. Dua sebab kegagalan v1 + tambalannya | |
| 3. Apa yang dibeli contoh ke-3 (kasus tepi) | |
| 4. Kapan rela bayar ~11x token prompt produksi | |
| 5. Kenapa delimiter & leash masing-masing tidak cukup + lapisan di produksi | |
| 6. Implikasi temperature pada strategy parsing & structured output | |

**Standar:** jawaban pakai kata-kata sendiri + merujuk angka/aksi eksperimenmu.
Jawaban "kayaknya" = 2 poin max.

## D. Kualitas Kode (10 poin)

| Kriteria | ✅/❌ | Catatan |
|---|---|---|
| Fungsi murni — input tidak dimodifikasi (`pilih_contoh`, `urutkan_contoh`, `merge`-nya template) | | |
| Tidak "mencurangi" test (mis. hardcode jawaban kasus GOLDEN di fungsi) | | |
| Semua acak lewat `random.Random(seed)` — tidak ada `random` global | | |
| Guard: input kosong / teks tanpa JSON / `{}` tidak membuat crash | | |

## E. Refleksi (10 poin)

Tulis 3–5 kalimat:

- Test mana yang paling lama kamu perbaiki (`render` vs `str.format`? `bool` vs
  `int`? brace-matching?), dan apa yang akhirnya kamu pahami?
- Setelah melihat `field_accuracy` bergerak 0.0 → 0.53 → 1.0, bagian mana dari
  cara kamu menulis prompt sehari-hari yang berubah?
- Satu hal yang mau kamu bawa ke Bab 11 (LLM API & orkestrasi): struktur output
  provider (`response_format`/tool calling), cache prompt, atau evaluasi
  otomatis di CI?
