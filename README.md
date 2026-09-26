# 📚 Kurikulum AI Engineering — Dari Nol sampai Produksi

> Buku pelajaran AI Engineering yang disusun berdasarkan koleksi materi Bangkit 2024 Anda, diperluas dengan topik AI Engineering modern (LLM, RAG, Agent, MLOps).

**Cara pakai:** Baca bab per bab secara berurutan. Setiap bab punya: tujuan belajar → materi inti → referensi file lokal → latihan → checklist sebelum lanjut.

---

## Struktur Buku

| Bab | Judul | Level |
|-----|-------|-------|
| 0 | [Peta Jalan & Cara Belajar](00-Peta-Jalan/README.md) | Semua |
| 1 | [Fondasi Python untuk AI](01-Python-Fondasi/README.md) | Pemula |
| 2 | [Matematika untuk ML](02-Matematika-ML/README.md) | Pemula |
| 3 | [Machine Learning Klasik](03-ML-Klasik/README.md) | Inti |
| 4 | [Deep Learning & TensorFlow](04-Deep-Learning-TensorFlow/README.md) | Inti |
| 5 | [Computer Vision](05-Computer-Vision/README.md) | Inti |
| 6 | [NLP & Text](06-NLP-Text/README.md) | Inti |
| 7 | [Time Series](07-Time-Series/README.md) | Inti |
| 8 | [ML Produksi: Evaluasi, GAN, Struktur Proyek](08-ML-Produksi/README.md) | Lanjut |
| 9 | [Fondasi LLM (Transformer & Tokenisasi)](09-LLM-Fondasi/README.md) | AI Eng |
| 10 | [Prompt Engineering](10-Prompt-Engineering/README.md) | AI Eng |
| 11 | [Bekerja dengan LLM API & Orkestrasi](11-LLM-API-Orkestrasi/README.md) | AI Eng |
| 12 | [RAG — Retrieval Augmented Generation](12-RAG/README.md) | AI Eng |
| 13 | [Fine-tuning & Open-Weight Models](13-Finetuning-OpenModels/README.md) | AI Eng |
| 14 | [Agentic AI](14-Agentic-AI/README.md) | AI Eng |
| 15 | [Evaluasi LLM](15-Evaluasi-LLM/README.md) | AI Eng |
| 16 | [Deployment & MLOps untuk AI Engineering](16-Deployment-MLOps/README.md) | AI Eng |
| 17 | [Proyek Akhir (Capstone)](17-Capstone/README.md) | AI Eng |

> ⚠️ Jika ada link rusak di tabel di atas, buka langsung folder babnya.

> 🆕 **Materi pendukung** (lab `.ipynb`, kuis berbobot, kunci jawaban, cheatsheet, project starter) sudah tersedia untuk **Bab 1–12**. Rencana belajar intensif minggu pertama: [`MINGGU-01.md`](MINGGU-01.md).

---

## Peta Kompetensi (Skill Map)

```
Bab 0-2  : Fondasi (Python, Matematika, Tools)
Bab 3-8  : Machine Learning & Deep Learning (dari materi Bangkit)
Bab 9-16 : AI Engineering modern (LLM, RAG, Agent, MLOps)
Bab 17   : Integrasi semua — proyek akhir
```

---

## Sumber Materi

1. **Materi lokal (folder `Bangkit/`)** — labs & assignments dari:
   - Machine Learning Specialization (Andrew Ng)
   - Mathematics for ML and Data Science
   - TensorFlow Developer Professional Certificate
   - Advanced Techniques Specialization
   - TensorFlow S2 (GAN)
   - Laporan Bulan 1–4 (`.docx` / `.pdf`)
2. **Referensi online** — dicantumkan per bab (dokumentasi resmi, buku, paper).

---

## Konvensi

- 📁 Setiap bab = 1 folder dengan `README.md`.
- 🔗 Referensi file lokal ditulis relatif dari root repo (`../Bangkit/...`).
- ✅ Setiap bab diakhiri **checklist kompetensi** — jangan lanjut bab sebelum checklist penuh.
- 🧪 File `.ipynb` dibuka dengan **Jupyter** atau **VS Code**.
- 📄 File `.docx`/`.pdf` laporan bulanan = contoh bagaimana mendokumentasikan proyek AI secara profesional.

---

## Langkah Cepat (Quick Start)

```bash
# 1. Buat environment terpisah
python -m venv .venv
source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate

# 2. Install dependensi inti
pip install jupyter numpy pandas matplotlib scikit-learn tensorflow
```

Untuk bab AI Engineering (9+): `pip install openai anthropic langchain langgraph ragas` — dijelaskan detail di bab masing-masing.

---

## Status Materi Pendukung per Bab

| Bab | Lab | Kuis | Kunci | Cheatsheet | Project Starter (TDD) |
|---|---|---|---|---|---|
| 1–5 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 6 — NLP & Text | ✅ | ✅ | ✅ | ✅ | ✅ LM mini |
| 7 — Time Series | ✅ | ✅ | ✅ | ✅ | ✅ Forecasting mini |
| 8 — ML Produksi | ✅ | ✅ | ✅ | ✅ | ✅ Evaluasi produksi (52 test) |
| 9 — LLM Fondasi | ✅ | ✅ | ✅ | ✅ | ✅ Tokenizer & mini-LM (76 test) |
| 10 — Prompt Engineering | ✅ | ✅ | ✅ | ✅ | ✅ Lembar prompt & harness eval (157 test) |
| 11 — LLM API & Orkestrasi | ✅ | ✅ | ✅ | ✅ | ✅ llmkit: cost/retry/fallback/cache/stream/tools/pipeline (107 test) |
| 12 — RAG | ✅ | ✅ | ✅ | ✅ | ✅ ragkit: vectorstore/retriever/abstain/cache/hybrid/rerank/pipeline/evaluate (66 test) |

> Lab/kuis/project sudah terverifikasi otomatis (skrip `_verify_*.py` di tiap folder bab) —
> semua angka terkunci seed dan lolos pengecekan.
