# 📄 Cheatsheet — Fine-tuning & Open-Weight Models (Bab 13)

> Review cepat. Uji diri: tutup file ini, tulis ulang decision matrix +
> matematika LoRA + kontrak early stopping dari ingatan, baru cek.

---

## 1. Decision Matrix: Prompt → RAG → Fine-tune

| Masalah | Solusi pertama | Naik ke |
|---|---|---|
| Fakta privat berubah-ubah | **RAG** (Bab 12) | RAG + fine-tune ringan |
| Format/gaya khusus & konsisten | Prompt + few-shot | **Fine-tune kecil** |
| Domain sangat spesifik | Prompt sistem panjang | Fine-tune domain |
| Privasi / offline / biaya API | **Open-weight lokal** | Fine-tune sendiri |

| Pendekatan | Mengubah | Tidak mengubah |
|---|---|---|
| Prompting | konteks | bobot |
| RAG | fakta saat inferensi | bobot — fakta bisa diperbarui |
| Fine-tuning | perilaku & format | pengetahuan faktual yang diaudit |

**Jebakan:** fine-tuning bukan alat menanam fakta — rawan halusinasi, tidak
bisa diaudit. Fakta = RAG; perilaku = fine-tune; sering dikombinasi.

---

## 2. Open-Weight & Serving

- Keluarga: LLaMA (Meta) · **Qwen (kuat multilingual)** · Mistral · Gemma · DeepSeek-distill.
- Ukuran: 1B–3B edge/task sempit · **7B–14B sweet spot** · 70B+ butuh GPU server.
- Serving: **Ollama** (laptop) · **vLLM** (produksi, continuous batching) ·
  llama.cpp (CPU/GGUF) · TGI (HF).
- Hampir semua mengekspos **API kompatibel OpenAI** → ganti `base_url`
  (`http://localhost:11434/v1`), kode orkestrasi Bab 11 tetap jalan.

---

## 3. Kuantisasi: Ukur, Jangan Asumsi

```
skala = max|w| / (2^(bit-1) - 1)        # simetris per-tensor
q     = clamp(round(w/skala))           # dekuanti: w' = q × skala
```

Angka terkunci (7B, SEED 13): fp16 **14 GB / 0.82** → int8 **7 GB / 0.81** →
q4 **3.5 GB / 0.78**. fp16→int8 hampir gratis; int8→q4 mulai terasa.
Ukuran turun linear terhadap bit; **skor tidak linear** → evaluasi per task.
QLoRA = base 4-bit + adapter bf16 → fine-tune 7B di GPU 16 GB.

---

## 4. Dataset SFT (JSONL ala Alpaca)

```json
{"instruksi": "...", "input": "...", "respons": "..."}
```

- Format prompt TERKUNCI: `instruksi + \n\n + Input: ... + \n\n + Jawaban:`
  — format bergoyang = model bingung.
- **500–5.000 contoh berkualitas** > puluhan ribu berisik. Kualitas > kuantitas.
- Split deterministik interleaved: `(i*7 + seed) % 10 < rasio*10` — split yang
  berubah tiap run = evaluasi tak bisa dibandingkan. 8 contoh rasio 0.25 → (5, 3).
- **Oversampling** (data timpang 5/3): duplikasi minoritas deterministik,
  **urutan asli dipertahankan** (menggrup per label = distribusi batch berubah).
  Risiko duplikasi berlebih: memorisasi; alternatif: augmentation, class weight.

---

## 5. LoRA: ΔW = B @ A

```
A: rank × dim_in  (acak kecil)
B: dim_out × rank (NOL → ΔW awal 0 → model mulai UTUH)   ← jangan dua-duanya acak
param = rank × (dim_in + dim_out)      # BUKAN rank × dim_in × dim_out
```

Angka terkunci: 4096×4096 rank 16 → 131.072 param dari 16.7 jt =
**rasio 0.0078125 ≈ 0.78%**. Gradient check (central diff h=1e-5):
**max err 6.4e-12** — gradien analitik harus cocok numerik SEBELUM dipercaya.

- Rank 8–64 untuk gaya/format; naikkan saat domain beda jauh.
- 1 base + banyak adapter (per klien/task); QLoRA untuk VRAM terbatas.
- Tools: Unsloth (cepat, Colab) · Axolotl (YAML) · HF TRL + PEFT (standar).

---

## 6. Loop SFT & Early Stopping (Bab 4 tetap berlaku)

```
data bersih (lr=0.5, 60 ep) : train_loss 0.7396 → 0.0872 | val_acc 1.0 | tidak dini
overfit (2 label dibalik, lr=3.0) : berhenti ep 66 | BEST epoch 61
  val_loss best 0.4184  vs  akhir 0.4188   | acc di bobot best 0.75
```

| Kontrak | Kenapa |
|---|---|
| loss dicatat dari p **SEBELUM** update | sesudah update = angka lain & menyembunyikan kesulitan awal |
| val loss/acc dihitung **tiap epoch** | dasar keputusan berhenti |
| berhenti bila val tak membaik `sabar` epoch | overfit data kontradiktif |
| restore bobot **TERBAIK** (bukan terakhir) | lupa `best=(list(w),b)` → model overfit ke produksi |

Hyperparameter SFT LoRA nyata: lr **1e-4 ~ 2e-4** · epoch **1–3** (SFT overfit
cepat) · warmup ~3% · cosine decay · goyangkan SATU variabel per eksperimen.

---

## 7. Biaya: API vs Lokal = Fungsi Volume

```
biaya/req ≈ token_in/1e6 × harga_in + token_out/1e6 × harga_out   # 'lokal' = 0
```

Angka terkunci (1k in/300 out): api_besar $0.0055 · api_mini $0.00033 ·
lokal $0. Bulanan: 2.000 req/hari = $19.8 → **api**; 10.000 req/hari ≈ $99 →
**lokal** (GPU $60/bln). Breakeven ≈ ($60/30) / $0.00033 ≈ **6.061 req/hari**
(asumsi modul). Penggeser ambang: harga token, privasi, uptime jadi
tanggunganmu, utilisasi GPU, latensi/throughput. Keputusan = angka, bukan tren.

---

## 8. Evaluasi Jujur + OOD (abstain, Bab 12)

```
sebelum 0.75 → sesudah 1.0 (delta +0.25)     # tanpa ini = "rasanya lebih pintar"
yakin = |p - ambang| × 2 ; yakin < 0.6 → JANGAN tampilkan, arahkan ke manusia
OOD "Lumayan standar saja sih" → skor 0.4988, yakin 0.0024 → tertahan ✅
```

`kebijakan_ood` mengembalikan **dict baru** (input utuh untuk audit). LLM nyata
tak punya satu angka yakin → logprob, self-consistency, judge rubrik (Bab 15);
lapisan abstain tetap wajib di kode.

---

## 9. Anti-pattern

| Anti-pattern | Gejala | Solusi |
|---|---|---|
| Fine-tune untuk menanam fakta | halusinasi + basi saat data berubah | RAG untuk fakta |
| Split/oversample pakai `random` murni | evaluasi tak bisa dibandingkan antar run | formula deterministik (seed) |
| Oversample menggrup per label | urutan asli berubah → distribusi batch bergeser | pertahankan urutan asli |
| A dan B keduanya acak | model rusak SEBELUM training | **B = nol** |
| Percaya gradien tanpa check | training "jalan" tapi tak konvergen | gradient check < 1e-6 |
| Restore bobot terakhir | model overfit ke produksi | restore best_epoch |
| Loss dicatat setelah update | kurva menyesatkan | catat p sebelum update |
| Q4 di semua kasus | kualitas turun di task sensitif | ukur skor per task |
| Keputusan API/lokal dari tren | biaya meledak / GPU mangkrak | breakeven dari volume |
| Fine-tune tanpa eval sebelum/sesudah | "rasanya lebih pintar" | delta akurasi + OOD gate |

---

## 10. Checklist Review 30 Detik

- [ ] Decision matrix bisa dijelaskan: prompt → RAG → fine-tune, dengan alasan.
- [ ] Format prompt SFT konsisten; split & oversample deterministik, urutan asli.
- [ ] ΔW = B@A dengan B nol; param = rank×(in+out); rasio 0.78% @ rank 16.
- [ ] Gradient check lolos sebelum loop training dipercaya.
- [ ] Early stopping: val tiap epoch, sabar, restore BEST (61 ≠ 66).
- [ ] Kuantisasi: ukuran linear, skor tidak — evaluasi per task (0.82/0.81/0.78).
- [ ] Keputusan API vs lokal dari volume: breakeven ≈ 6.061 req/hari.
- [ ] Eval sebelum/sesudah (0.75→1.0) + OOD abstain (yakin 0.0024 tertahan).
- [ ] Semua angka terkunci SEED 13, tanpa GPU, tanpa API, bisa direproduksi.
