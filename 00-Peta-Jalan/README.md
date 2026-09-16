# Bab 0 — Peta Jalan & Cara Belajar

> Sebelum menulis kode, pahami dulu **peta wilayahnya**. Bab ini adalah "pendahuluan buku" — baca sekali, kembali lagi sewaktu-waktu.

## 🎯 Tujuan Belajar

- Memahami perbedaan **Machine Learning**, **Deep Learning**, dan **AI Engineering**.
- Menyusun rencana belajar 17 bab + capstone.
- Tahu cara membaca notebook `.ipynb` dan mendokumentasikan proyek.
- Menghindari jebakan pembelajaran umum yang membuat progres terasa macet.

---

## 1. Peta Besar: Dari ML ke AI Engineering

```
┌────────────────────────────────────────────────────────────┐
│  TRADISIONAL (Bab 1–8)          │  AI ENGINEERING (Bab 9–17)
│                                 │
│  Data → Features → Model        │  Model dasar sudah jadi (LLM)
│  → Training → Deploy            │  Anda membangun DI ATAS model:
│                                 │  - Prompting (Bab 10)
│  Fokus: statistik, training,    │  - RAG (Bab 12)
│  evaluasi, generalisasi         │  - Agents/tools (Bab 14)
│                                 │  - Evaluasi & observability (15)
│                                 │  - Deployment & cost (16)
└────────────────────────────────────────────────────────────┘
```

**Insight kunci:** AI Engineer ≠ ML Researcher. AI Engineer adalah *software engineer yang membangun produk di atas model AI* — jadi skill software (Bab 1, 16) sama pentingnya dengan skill model.

---

## 2. Kenapa Fondasi Klasik Tetap Wajib?

Meski fokus Anda AI Engineering, bab 1–8 penting karena:

1. **RAG/Agent sering butuh model kecil**: classifier intent, reranker, embedding evaluation — semua pakai fondasi ML klasik.
2. **Evaluasi LLM** (Bab 15) memakai konsep metrik dari Bab 3 & 8.
3. **Fine-tuning** (Bab 13) = deep learning training loop dari Bab 4.
4. Saat LLM gagal, Anda perlu tahu *kapan* ganti ke model klasik (Bab 3).

> **Aturan emas:** Jangan pernah melewati Bab 3 (evaluasi model) dan Bab 8 (struktur proyek). Dua bab ini adalah perbedaan utama antara hobbyist dan professional.

---

## 3. Kenapa Banyak Orang Gagal Lanjut?

### 🚫 Jebakan Umum

| Jebakan | Tanda | Solusi |
|---|---|---|
| Teknologi hopping | Loncat dari satu blog tutorial ke blog lain tanpa selesai 1 latihan | Commit 1 bab = 1 project kecil, bukan cuma baca |
| Tutorial hell | Tonton video sampai habis, tapi nggak pernah nulis kode sendiri | Aturan 1:1 — setiap 20 min baca/video, tulis 1/2 halaman kode atau catatan |
| Takut error | Ragu-ragu ngedit, takut rusak | Error adalah fitur, bukan bug. Baca error baris pertama, bukan seluruh stacktrace |
| Overwhelm bab 1–8 | "Cuma butuh LLM, kenapa harus belajar statistik?" | Bab 1–8 cuma butuh 30–40% waktu total. Sisanya untuk Bab 9+ |

---

## 4. Cara Menggunakan Kurikulum Ini

### Pola Belajar Efektif

- **Urutan:** bab berurutan, tapi Bab 9+ boleh dipelajari paralel dengan Bab 5–8 jika target Anda langsung LLM.
- **Ritme harian (minimal viable):**
  - 20 menit: baca 1 sub-bab
  - 30 menit: lakukan 1 latihan / eksplorasi
  - 10 menit: tulis catatan "apa yang saya pahami hari ini"
- **Notebook:** setiap file `.ipynb` di folder `Bangkit/` adalah "lab praktikum". Jalankan ulang, ubah parameternya, amati efeknya — jangan hanya baca.
- **Dokumentasi:** contoh cara melaporkan progres ada di `Bangkit/LAPORAN BULAN PERTAMA.docx` … `Bulan Keempat.docx`.

### Teknik Dasar Agar Tahan Lama di Memori

1. **Self-explanation:** Saat baca konsep baru, tutup laptop, coba jelaskan dengan kata sendiri (boleh ke dinding/langit).
2. **Chunking:** Bagi materi besar menjadi 3–5 potongan; pelajari satu chunk, istirahat, lanjut.
3. **Retrieval practice:** Setelah baca sub-bab, tutup materi, ditulis ulang tanpa lihat — ini jauh lebih kuat daripada baca ulang.
4. **Spaced repetition:** Review bab sebelumnya saat memulai bab baru (contoh: Bab 2 saat masuk Bab 3, Bab 4 saat Bab 9).
5. **Feynman technique:** Jelaskan ke "orang awam" — jika masih ribet, berarti konsep belum tajam.

---

## 5. Rhitme Belajar yang Realistis

| Target | Durasi per Bab | Total Waktu | Cocok Untuk |
|---|---|---|---|
| Intensif (daily study) | 1 minggu | ~17 minggu | Fokus transisi karier |
| Berat-berat (3x/minggu) | 2 minggu | ~34 minggu | Pelajar sampingan kerja |
| Santai (weekend) | 3–4 minggu | ~50–60 minggu | Eksplorasi jangka panjang |

> Saran: Tentukan tanggal mulai & target selesai, lalu hitung mundur. Contoh: "Saya mulai 10 Sep, target capstone jadi di Januari."

---

## 6. Cara Membaca Notebook Ilmiah (Lab Praktikum)

Notebook `.ipynb` bukan buku teks — itu **laboratorium interaktif**. Cara membacanya:

1. Jalankan semuanya (Kernel → Restart & Run All) dan amati output akhir.
2. Baris per baris, baca cell dengan telepati: "Apa yang diharapkan cell ini?"
3. Ubah 1 parameter (mis. learning rate dari 0.01 → 0.1), jalankan ulang, lihat apa yang berubah.
4. Jika hasilnya aneh: coba pahami kenapa — inilah bagian pembelajaran tertinggi.
5. Jangan nge-copy-paste tanpa paham. Ketik ulang sendiri.

---

## 7. Cross-Reference: Kapan Bab Terkait Dipakai Kembali

| Ketika Anda di... | Bab berapa yang saling terkait? |
|---|---|
| Bab 3 (ML Klasik) | Evaluasi dari Bab 2 (statistik), overfitting dari Bab 4 (regularisasi), koding dari Bab 1 |
| Bab 4 (Deep Learning) | Backprop dari Bab 2 (kalkulus), TF API dari Bab 1 (Python), callback dari Bab 3 (esensi evaluation) |
| Bab 5,6,7 (domain-specific) | Semua praktik dari Bab 3, dan struktur layanan dari Bab 8 |
| Bab 9 (LLM Fondasi) | Attention dari Bab 4/6, tokenisasi dari Bab 6, embedding dari Bab 2/6 |
| Bab 10 (Prompt) | Struktur prompt sebenarnya bahasa natural "API" — berpikir seperti programming (Bab 1) |
| Bab 12 (RAG) | Chunking dari Bab 6 (NLP dasar), embedding/cosine similarity dari Bab 2, evaluasi Bab 15 |
| Bab 13 (Fine-tune) | Training loop GradientTape/Bab 4, overfitting Bab 3, evaluasi Bab 15 |
| Bab 14 (Agent) | Tool calling Bab 11, prompt ReAct Bab 10, retrieval Bab 12, max-iteration dari Bab 1 |
| Bab 15 (Evaluasi) | Metrik Bab 3, pipeline CI/CD Bab 16, experiment log Bab 8 |
| Bab 16 (Deployment) | API design Bab 1/11, Docker Bab 8, logging dari Bab 1 (unit test mindset) |
| Bab 17 (Capstone) | Semua bab di atas — inilah saatnya menyatukannya |

---

## ✅ Checklist Kompetensi

- [ ] Bisa menjelaskan perbedaan ML klasik vs AI Engineering dengan kata sendiri
- [ ] Environment Python + Jupyter siap dan bisa menjalankan notebook
- [ ] Menetapkan target: berapa minggu per bab, kapan capstone
- [ ] Tahu 3 jebakan umum dan cara menghindarinya
- [ ] Paham teknik belajar yang efektif (pilih 2 untuk dipakai)
