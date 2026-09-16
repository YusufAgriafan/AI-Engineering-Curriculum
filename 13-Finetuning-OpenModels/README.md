# Bab 13 — Fine-tuning & Open-Weight Models

> Kadang prompting + RAG tidak cukup: butuh gaya/behavior khusus, latensi rendah, privasi data, atau biaya per-token nol. Di sinilah open-weight model & fine-tuning masuk.

## 🎯 Tujuan Belajar
- Kapan fine-tuning, kapan RAG, kapan cukup prompting (decision matrix).
- Menjalankan open-weight model (LLaMA, Qwen, Mistral) lokal via Ollama/vLLM.
- Fine-tuning dengan LoRA/QLoRA — memakai skill GradientTape dari Bab 4.

## 1. Materi Inti

### 1.1 Decision Matrix
| Masalah | Solusi pertama | Naik ke |
|---|---|---|
| Pengetahuan privat berubah-ubah | **RAG** (Bab 12) | RAG + fine-tune ringan |
| Format/gaya khusus & konsisten | Prompt + few-shot | **Fine-tune** kecil |
| Kualitas reasoning kurang | Model lebih besar | Fine-tune domain |
| Privasi/offline/biaya API | **Open-weight lokal** | Fine-tune sendiri |
| Latensi tinggi butuh murah | Model kecil + prompt pendek | Distilasi / fine-tune kecil |

Aturan praktis: **Prompt dulu → RAG → fine-tune**. Fine-tuning bukan alat menanam fakta (rawan halusinasi) — itu tugas RAG; fine-tuning mengubah *perilaku & format*.

### 1.2 Open-Weight Model & Serving
- Keluarga populer: **LLaMA** (Meta), **Qwen** (Alibaba, kuat multilingual), **Mistral**, **Gemma** (Google).
- Ukuran: 1B–3B (edge/murah), 7B–14B (sweet spot), 70B+ (kualitas tinggi, butuh GPU).
- Serving:
  - **Ollama** — cara termudah di laptop (`ollama run qwen2.5:7b`).
  - **vLLM** — throughput produksi (continuous batching, paged attention).
  - **llama.cpp** — CPU/Mac, kuantisasi GGUF.
- OpenAI-compatible endpoint: banyak server open-weight mengekspos API ala OpenAI → kode Bab 11 bisa dipakai ulang tanpa diubah.

### 1.3 Kuantisasi
- FP16 → INT8 → INT4: model 7B dari ~14 GB → ~4 GB, akurasi turun sedikit.
- Trade-off: ukuran/kecepatan vs kualitas; selalu eval (Bab 15) sebelum/ sesudah.

### 1.4 Fine-tuning: SFT & LoRA
- **SFT (Supervised Fine-Tuning)**: latih pada pasangan instruksi→respons (format JSONL).
- **LoRA**: latih adapter kecil (rank r, mis. 8–64) di atas bobot beku → butuh VRAM jauh lebih kecil; **QLoRA** = LoRA di atas model 4-bit.
- Hyperparameter penting: learning rate (1e-4 ~ 2e-4 untuk LoRA), epoch (1–3), rank, max seq length.
- Data: kualitas > kuantitas; 500–5.000 contoh berkualitas sering cukup untuk perilaku/format; hindari data basi atau duplikat.
- Tools: **Unsloth** (efisien, gratis tier Colab), **Axolotl**, **HF TRL (`SFTTrainer`)**, **PEFT**.
- Sambungan: konsepnya identik dengan training loop Bab 4 — loss, overfitting, val split, early stopping tetap berlaku.

## 2. Latihan
1. Jalankan `qwen2.5:3b` via Ollama; tukar endpoint dari `openai` SDK ke `http://localhost:11434/v1` — buktikan kode Bab 11 tetap jalan.
2. Buat dataset 300 contoh (tanya→jawab gaya tertentu, JSONL). Fine-tune model 3B dengan QLoRA di Colab/Unsloth; bandingkan output sebelum/sesudah dengan eval sederhana.
3. Kuantisasi model ke Q4, ukur perbedaan kualitas & kecepatan.

## 📚 Referensi Online
- [Ollama Docs](https://ollama.com/docs)
- [vLLM Docs](https://docs.vllm.ai/)
- [Hugging Face PEFT (LoRA)](https://huggingface.co/docs/peft/index)
- [Unsloth — fine-tune notebook gratis](https://github.com/unslothai/unsloth)
- [HF TRL — SFT](https://huggingface.co/docs/trl/index)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)

## ✅ Checklist Kompetensi
- [ ] Menjalankan open-weight model lokal & menghubungkannya ke kode API
- [ ] Menentukan RAG vs fine-tuning dengan alasan, bukan hype
- [ ] Menyelesaikan satu fine-tune LoRA end-to-end + eval hasilnya
