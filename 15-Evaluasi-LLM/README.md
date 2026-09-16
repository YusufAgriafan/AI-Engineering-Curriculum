# Bab 15 — Evaluasi LLM

> "Rasanya lebih bagus" bukan metrik. Bab ini membuat Anda bisa membuktikan bahwa sistem AI Anda baik — dan mendeteksi regresi sebelum user menyadarinya. Ini kompetensi paling membedakan AI engineer profesional.

## 🎯 Tujuan Belajar
- Menyusun eval set (golden dataset) & metrik: exact, heuristik, model-graded, human.
- Membangun eval pipeline otomatis yang jalan setiap kali prompt/model berubah.
- Observability: tracing, logging, monitoring produksi.

## 1. Materi Inti

### 1.1 Kumpulkan Contoh & Golden Dataset
- 30–100 kasus nyata (dari log, user, tim), termasuk **edge case**: input kosong, bahasa campur, prompt injection, pertanyaan di luar cakupan.
- Untuk tiap kasus: input, output ideal (atau kriteria), kategori kesulitan.
- Split seperti Bab 3: dev (untuk iterasi) vs test (disentuh hanya saat rilis besar).

### 1.2 Jenis Metrik
| Jenis | Contoh | Kapan |
|---|---|---|
| Deterministik | exact match, JSON valid, sitasi ada, latency | Format, struktur |
| Heuristik | panjang, kata terlarang, korelasi lexicon | Cepat & murah |
| **Model-graded (LLM-as-judge)** | rubrik 1–5: akurasi, relevansi, faithfulness | Kualitas linguistik |
| Human | rating ahli, A/B, thumbs up/down | Ground truth & kalibrasi judge |

- **LLM-as-judge**: beri judge rubrik eksplisit + contoh skala; waspadai bias (panjang, posisi). Kalibrasi dengan sampling human review.
- Untuk RAG: **faithfulness** (jawab sesuai konteks?) & **answer relevancy** → pakai `ragas` (Bab 12).

### 1.3 Eval Pipeline
```
evalset.yaml  →  runner (panggil sistem)  →  scorer  →  laporan (tabel + diff vs versi lalu)
```
- Jalankan di CI: setiap PR yang menyentuh prompt/model/konfigurasi → eval wajib hijau.
- Selalu bandingkan **versi vs versi** (regression testing), bukan angka absolut.
- Error analysis: klasteri kasus gagal (tema? panjang input? bahasa?) → perbaiki yang paling sering.

### 1.4 Observability Produksi
- **Trace** setiap permintaan: prompt, konteks RAG, tool calls, token, biaya, latensi (Langfuse / LangSmith / OpenTelemetry).
- Dashboard: error rate, halusinasi terdeteksi, biaya/hari, p95 latency.
- Feedback loop: thumbs down user → masuk antrian review → jadi kasus eval baru. Lingkaran selesai.

## 2. Latihan
1. Buat `evalset.yaml` 30 kasus untuk sistem RAG Bab 12 (termasuk 5 kasus "harus menjawab tidak tahu").
2. Runner + scorer: exact match untuk "tidak tahu", model-graded rubrik untuk sisanya; laporkan skor per kategori.
3. Integrasikan ke GitHub Actions: gagalkan PR jika skor turun > 2 poin.
4. Tambahkan Langfuse: trace 10 permintaan, temukan 1 bottleneck latensi.

## 📚 Referensi Online
- [OpenAI Cookbook — evaluation](https://cookbook.openai.com/examples/evaluation)
- [RAGAS docs](https://docs.ragas.io/)
- [Langfuse docs (open-source LLM observability)](https://langfuse.com/docs)
- [Hamel Husain — Your AI product needs evals](https://hamel.dev/blog/posts/evals/)
- [Anthropic — evals guide](https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests)

## ✅ Checklist Kompetensi
- [ ] Punya golden dataset + runner + scorer yang bisa dijalankan satu perintah
- [ ] Mengkalibrasi LLM-judge terhadap penilaian manusia
- [ ] Sistem punya tracing produksi dengan biaya & latensi per request
