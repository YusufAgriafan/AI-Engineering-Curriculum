# Bab 17 — Proyek Akhir (Capstone)

> Saatnya menyatukan Bab 0–16 menjadi satu produk nyata di portofolio. Target akhir: sistem AI yang *dipakai orang lain*, terukur, dan terdeploy — persis seperti laporan akhir Bangkit Anda, tapi level produksi.

## 🎯 Tujuan
Satu sistem AI end-to-end dengan:
- ✅ Kode terstruktur (Bab 8) + unit test (Bab 1)
- ✅ LLM/Agent/RAG dengan prompt sebagai file (Bab 10)
- ✅ Eval otomatis dengan angka (Bab 15)
- ✅ Deploy + monitoring + laporan biaya (Bab 16)

## 1. Pilih Satu Ide

| Ide | Bab inti | Tingkat |
|---|---|---|
| **Asisten tanya-jawab dokumen** (skripsi/regulasi/dokumen kampus) dengan sitasi | 10, 12, 15 | ⭐⭐ |
| **Agent customer-support** (FAQ + cek status pesanan via tools + eskalasi ke manusia) | 11, 14, 15 | ⭐⭐⭐ |
| **Analisis dokumen batch** (ekstrak struktur dari 100 PDF → JSON → dashboard) | 10, 11, 16 | ⭐⭐ |
| **Copilot domain** (asisten penulisan laporan bulanan gaya `Laporan Akhir` Anda) | 10, 13, 15 | ⭐⭐⭐ |
| **Sistem hibrida klasik+LLM** (classifier cepat + LLM untuk kasus sulit) | 3, 11, 15 | ⭐⭐⭐ |

Saran: pilih yang datanya ada di tangan Anda (dokumen Bangkit/portofolio Anda sendiri adalah dataset sah!).

## 2. Kerangka Kerja 6 Minggu

| Minggu | Fokus | Deliverable |
|---|---|---|
| 1 | Problem & data: definisi user, kumpulkan 50–100 contoh nyata, tulis eval set awal | `evalset.yaml` v0 |
| 2 | Prototype pipeline paling sederhana (prompt pertama, RAG pertama) | Demo kasar |
| 3 | Iterasi: error analysis, perbaiki retrieval/prompt, ukur | Skor eval v1 |
| 4 | Produksi: struktur repo, test, Docker, API streaming | Deploy staging |
| 5 | Evals di CI + observability + kontrol biaya | Dashboard & gate |
| 6 | Tulis laporan & presentasi | Laporan akhir + repo publik |

## 3. Struktur Repo Capstone
```
capstone/
├── data/                 # raw/, processed/ (gitignore)
├── prompts/              # prompt versioned (Bab 10)
├── src/
│   ├── ingestion.py      # load-chunk-embed (Bab 12)
│   ├── retrieval.py
│   ├── llm.py            # API + retry + fallback (Bab 11)
│   ├── agent.py          # jika ada (Bab 14)
│   └── api.py            # FastAPI (Bab 16)
├── tests/                # unit test (Bab 1)
├── evals/                # evalset + runner + scorer (Bab 15)
├── reports/              # experiment log + laporan akhir (Bab 8)
├── .github/workflows/    # CI (Bab 16)
└── README.md             # cara menjalankan + arsitektur diagram
```

## 4. Template Laporan Akhir
Ikuti pola laporan Bangkit Anda (`Bangkit/Laporan Akhir_*.pdf`):
1. **Latar belakang & masalah** — siapa user, apa sakitnya
2. **Solusi & arsitektur** — diagram alur data
3. **Metodologi** — keputusan teknis + alasannya (kenapa RAG, kenapa model X)
4. **Evaluasi** — tabel skor eval, perbandingan versi, biaya
5. **Deployment** — URL, monitoring, kendala produksi
6. **Pembelajaran & langkah lanjut**

## 5. Kriteria "Lulus"
- [ ] Demo live untuk orang luar tim tanpa Anda menyetir
- [ ] Skor eval tercatat & meningkat antar versi
- [ ] Biaya per 1.000 permintaan diketahui
- [ ] README memungkinkan orang lain menjalankan dalam < 10 menit

## 6. Setelah Capstone
- Publikasikan: repo + tulisan singkat (blog/LinkedIn) tentang keputusan teknis.
- Terus belajar: paper mingguan (mis. newsletter *Ahead of AI*, *Latent Space*), kontribusi open-source AI tooling.
- Perdalam jalur: deep-dive evaluasi (Bab 15) atau infra open-weight (Bab 13–16) sesuai karier yang dituju.
