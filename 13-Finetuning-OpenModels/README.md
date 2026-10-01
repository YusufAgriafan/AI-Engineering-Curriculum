# Bab 13 — Fine-tuning & Open-Weight Models

> Kadang prompting + RAG tidak cukup: butuh gaya/behavior khusus, latensi rendah,
> privasi data, atau biaya per-token nol. Di sinilah open-weight model &
> fine-tuning masuk. Bab ini melatih **seluruh siklus fine-tuning tanpa GPU**:
> dataset SFT → LoRA → kuantisasi → trainer dengan early stopping → keputusan
> biaya API vs lokal → evaluasi sebelum/sesudah + kebijakan OOD.

## 🗂️ Materi Pendukung Bab Ini

| Materi | File | Isi |
|---|---|---|
| 🧪 Lab | [`01_lab_finetuning.ipynb`](01_lab_finetuning.ipynb) | 8 bagian: dataset & split → SFT JSONL + oversampling → matematika LoRA + gradient check → kuantisasi → trainer SGD + early stopping → overfit & restore-best → biaya API vs lokal → evaluasi + OOD |
| 📝 Kuis | [`02_kuis_finetuning.ipynb`](02_kuis_finetuning.ipynb) | 10 soal (6 PG + 4 coding), 22 poin, dinilai otomatis, **mandiri** |
| 🔑 Kunci | [`03_kunci_jawaban_kuis_finetuning.ipynb`](03_kunci_jawaban_kuis_finetuning.ipynb) | Jawaban + **kenapa**, kode referensi tiap soal coding |
| 📄 Cheatsheet | [`cheatsheet-finetuning.md`](cheatsheet-finetuning.md) | Ringkas 1 halaman untuk review cepat |
| 🏗️ Project | [`project-ftkit/`](project-ftkit/README.md) | Paket `ftkit` TDD **44 test**: sft, lora, quantize, trainer, api, evaluasi |

Materi lab & kuis **tanpa API key dan tanpa GPU**: "model" yang di-fine-tune
adalah klasifier tiny (`ftkit/model.py` — mean embedding → 1 logit → sigmoid)
yang dilatih SGD sungguhan di CPU (< 0,1 detik). Semua angka (loss 0.7396 →
0.0872, best epoch 61, rasio LoRA 0.78%, Q4 3.5 GB, breakeven ~6.061 req/hari)
**terkunci SEED 13** dan diverifikasi otomatis (`_verify_lab.py`,
`_verify_kuis.py`, `project-ftkit/_verify_project.py`).

---

## 🎯 Tujuan Belajar

- Kapan fine-tuning, kapan RAG, kapan cukup prompting (**decision matrix**).
- Menjalankan open-weight model (LLaMA, Qwen, Mistral, Gemma) lokal via
  Ollama/vLLM — dan mengeksposnya sebagai API kompatibel OpenAI.
- Membuat dataset SFT (JSONL ala Alpaca) + menangani ketidakseimbangan label.
- Memahami matematika **LoRA/QLoRA**: ΔW = B@A, rank, rasio parameter.
- Kuantisasi FP16 → INT8 → INT4: trade-off ukuran/kecepatan vs kualitas.
- Loop SFT dengan early stopping & **restore bobot terbaik** (bukan terakhir).
- Keputusan biaya: API per-token vs GPU lokal — fungsi volume, bukan selera.
- Evaluasi jujur sebelum/sesudah + kebijakan OOD ("kurang yakin" → abstain).

---

## 1. Materi Inti

### 1.1 Decision Matrix: Prompt, RAG, atau Fine-tune?

```
Masalah                          → Solusi pertama        Naik ke
─────────────────────────────────────────────────────────────────────
Fakta privat berubah-ubah        → RAG (Bab 12)          RAG + fine-tune ringan
Format/gaya khusus & konsisten   → Prompt + few-shot     Fine-tune kecil
Bahasa/domain sangat spesifik    → Prompt sistem panjang Fine-tune domain
Privasi / offline / biaya API    → Open-weight lokal     Fine-tune sendiri
Latensi & biaya per-token        → Model kecil + prompt  Distilasi / fine-tune
```

**Aturan praktis: Prompt dulu → RAG → fine-tune.**

Kenapa urutannya begitu? Karena ketiganya mengubah hal yang berbeda:

| Pendekatan | Mengubah apa | Tidak mengubah apa |
|---|---|---|
| Prompting | **Konteks** yang dilihat model | Bobot model |
| RAG | **Fakta** yang tersedia saat inferensi | Bobot model — fakta bisa diperbarui tanpa latih ulang |
| Fine-tuning | **Perilaku & format** (bobot) | Pengetahuan faktual baru (rawan halusinasi bila dipaksa) |

Jebakan klasik: fine-tuning untuk "menanam fakta". Model 7B yang di-fine-tune
dengan 1.000 dokumen kebijakan **tetap bisa halusinasi** — ia belajar *gaya*
jawaban, bukan *fakta* yang bisa diaudit. Fakta privat/berubah = RAG; perilaku
& format konsisten = fine-tune. Keduanya sering **dikombinasi**: fine-tune
memberi gaya jawaban, RAG menyuplai fakta saat query time.

### 1.2 Open-Weight Models & Serving

| Keluarga | Pemilik | Catatan |
|---|---|---|
| **LLaMA 3.x** | Meta | Ekosistem terbesar, lisensi khusus |
| **Qwen 2.5 / 3** | Alibaba | Sangat kuat multilingual (termasuk Indonesia) |
| **Mistral / Mixtral** | Mistral AI | Efisien, lisensi longgar |
| **Gemma 2/3** | Google | Ringan, bagus untuk edge |
| **DeepSeek-R1-distill** | DeepSeek | Reasoning, distilasi dari model besar |

Ukuran & peran:

- **1B–3B** — edge/CI/klasifikasi murah; sering cukup untuk task sempit yang
  di-fine-tune (persis analogi `ftkit` kita: kepala kecil di atas base).
- **7B–14B** — *sweet spot*; kuantisasi Q4 muat di GPU konsumen (7B Q4 ≈ 3.5 GB,
  angka terkunci project).
- **30B–70B+** — kualitas tinggi; butuh GPU server atau kuantisasi agresif.

Serving lokal:

| Tool | Kapan dipakai | Ciri |
|---|---|---|
| **Ollama** | Laptop/development | `ollama run qwen2.5:7b` — semudah Docker pull |
| **vLLM** | Produksi, throughput tinggi | Continuous batching, paged attention |
| **llama.cpp** | CPU / Mac (GGUF) | Kuantisasi agresif, tanpa GPU |
| **HF TGI** | Produksi ekosistem HF | Multi-GPU, streaming |

Kuncinya: kebanyakan server open-weight mengekspos **API kompatibel OpenAI**
(`http://localhost:11434/v1` untuk Ollama). Artinya kode orkestrasi Bab 11 —
klien, retry, fallback, cache, tools — **tetap jalan tanpa diubah**, cukup ganti
`base_url`. Fine-tune ini yang mengubah "otaknya", bukan cara memanggilnya.

### 1.3 Kuantisasi: Trade-off yang Harus Diukur

Bobot model disimpan FP16 (2 byte/param). Kuantisasi memampatkan ke INT8/INT4 —
angka terkunci project (7B):

| Varian | Bit | Ukuran (7B) | Kompresi | Skor |
|---|---|---|---|---|
| fp16 | 16 | 14.0 GB | 2.0× | 0.82 |
| int8 | 8 | 7.0 GB | 4.0× | 0.81 |
| q4 | 4 | **3.5 GB** | 8.0× | 0.78 |

Matematika simetris per-tensor (dipraktikkan di lab Bagian 4):

```
skala = max|w| / (2^(bit-1) - 1)
q     = clamp(round(w / skala), -levels, +levels)
w'    = q × skala          ← bobot dekuantisasi (yang dipakai saat forward)
```

Pelajaran penting dari angka di atas: **fp16 → int8 hampir gratis** (0.82 →
0.81), **int8 → q4 mulai terasa** (0.78). Ukuran GB turun linear terhadap bit,
tapi skor **tidak** linear — jangan berasumsi; ukur per task. Q4 layak bila
kualitas yang turun < biaya VRAM/latensi yang dihemat; TIDAK layak untuk
output yang sensitif-kualitas (ringkasan kontrak, jawaban medis).

> Kuantisasi juga diterapkan **ke adapter LoRA** (QLoRA): base model disimpan
> 4-bit, adapter tetap bf16 — fine-tuning 7B muat di GPU 16 GB.

### 1.4 Dataset SFT (Supervised Fine-Tuning)

Format JSONL ala Alpaca — satu objek per baris:

```json
{"instruksi": "Klasifikasikan sentimen ulasan berikut. Jawab HANYA satu kata: positif atau negatif.",
 "input": "Barang bagus, pengiriman cepat",
 "respons": "positif"}
```

Kualitas > kuantitas: **500–5.000 contoh berkualitas** sering cukup mengubah
perilaku/format; puluhan ribu data berisik justru mengajari model pola salah.
Praktik penting (semuanya dipraktikkan di lab Bagian 1–2):

1. **Konsistensi format** — format prompt terkunci (`instruksi + \n\n +
   Input: ... + \n\n + Jawaban:`); format yang bergoyang = model bingung.
2. **Split deterministik** — 8 contoh rasio 0.25 → (5, 3) dengan formula
   interleaved `(i*7 + seed) % 10 < rasio*10`; jangan `random.shuffle` murni —
   split yang berubah tiap run = evaluasi yang tak bisa dibandingkan.
3. **Oversampling untuk ketidakseimbangan** — FORMAT_SFT sengaja timpang
   (5 positif / 3 negatif); `oversample(per_label=6)` → 12 baris (6/6) dengan
   **duplikasi deterministik yang mempertahankan urutan asli**. Duplikasi
   berlebihan berisiko memorisasi — alternatif: data augmentation, class weight,
   atau ambil data minoritas tambahan.

### 1.5 LoRA: Matematika Adapter Kecil

Full fine-tune memperbarui **semua** bobot (7B ≈ 16,7 jt param di contoh
4096×4096; LLM nyata 7 miliar). LoRA membekukan base dan melatih **adapter
rank-rendah**:

```
W' = W + ΔW          ΔW = B @ A
A: rank × dim_in     (diinisialisasi acak kecil)
B: dim_out × rank    (diinisialisasi NOL → ΔW awal = 0, model mulai utuh!)
param terlatih = rank × (dim_in + dim_out)
```

Angka terkunci project: 4096×4096 rank 16 → **131.072 param dari 16.7 jt =
rasio 0.0078125 ≈ 0.78%**. Gradient check analitik vs numerik (central diff
h=1e-5): **max err 6.4e-12** — gradien yang benar adalah syarat percaya loop
training.

Kenapa B nol itu penting: saat adapter dipasang, model berperilaku **persis
base model** (ΔW = 0), lalu bergerak pelan saat dilatih. Kegagalan umum:
menginisialisasi A **dan** B acak → model langsung rusak sebelum training
mulai.

Praktik produksi:

- **Rank 8–64** cukup untuk task gaya/format; rank naik saat domain beda jauh.
- Satu base + **banyak adapter kecil** (per klien/per task) — ganti adapter =
  ganti perilaku, base tetap di VRAM sekali.
- **QLoRA** = LoRA di atas base 4-bit → fine-tune 7B di GPU konsumen.
- Tools: **Unsloth** (2× lebih cepat, hemat VRAM, gratis di Colab),
  **Axolotl** (konfig YAML), **HF TRL `SFTTrainer` + PEFT** (paling standar).

### 1.6 Loop SFT & Early Stopping — Pelajaran Bab 4 Tetap Berlaku

Training LLM 7B = forward → loss → gradien → update → val — **siklus yang sama
dengan Bab 4**, hanya skala param yang berbeda. Di lab/project kamu melihatnya
di model tiny (SGD per contoh, loss BCE, < 0,1 detik/run):

```
Angka terkunci (data bersih, lr=0.5, 60 epoch):
  train_loss: 0.7396 → 0.0872 | val_acc akhir 1.0 | tidak berhenti dini

Angka terkunci (2 label train SENGAJA dibalik + lr=3.0 → overfit):
  berhenti dini di epoch 66 | best epoch 61
  val_loss best 0.4184  vs  val_loss akhir 0.4188  | acc di bobot best = 0.75
```

Dua jebakan yang diuji test:

1. **Loss dicatat dari p SEBELUM update** tiap contoh — mencatat sesudah update
   memberi angka berbeda dan menyembunyikan kesulitan awal training.
2. **Restore bobot TERBAIK, bukan terakhir** — val loss memburuk setelah epoch
   61 (0.4184 → 0.4188); lupa `best = (list(w), b)` mengirim model yang
   overfit ke produksi. Early stopping **tanpa** restore = hanya berhenti
   di waktu yang tepat dengan bobot yang salah.

Hyperparameter praktis untuk SFT LoRA nyata: learning rate **1e-4 ~ 2e-4**
(mulai 2e-4 untuk rank kecil), epoch **1–3** (SFT overfit cepat!), max seq
length sesuai data, warmup ~3%, cosine decay. Goyangkan satu variabel per
eksperimen — variance antar run SFT nyata besar; andalkan eval set, bukan
perasaan (Bab 15).

### 1.7 Biaya: API vs Lokal adalah Fungsi Volume

Tarif ilustratif konsisten Bab 11 (USD per 1 jt token): api_besar 2.50/10.00,
api_mini 0.15/0.60, lokal 0 (biaya marginal token = 0; GPU jadi biaya tetap).

```
Angka terkunci (1.000 token in / 300 out per permintaan):
  api_besar $0.0055/req | api_mini $0.00033/req | lokal $0
  bulanan 2.000 req/hari api_mini = $19.8  → api menang
  bulanan 10.000 req/hari api_mini ≈ $99   → GPU $60/bln menang
  breakeven = ($60/30 hari) / $0.00033 ≈ 6.061 req/hari
```

Faktor yang menggeser ambang di dunia nyata: harga token berubah, kebutuhan
privasi (data tidak boleh keluar), uptime jadi tanggung jawabmu (Bab 16),
utilisasi GPU (model lain bisa share GPU), dan latensi/throughput. Keputusan
deployment = **angka**, bukan tren.

### 1.8 Evaluasi Jujur + Kebijakan OOD (Sambungan Bab 12)

Fine-tuning tanpa evaluasi = "rasanya lebih pintar" — tidak bisa dibandingkan.
Angka terkunci project:

```
sebelum fine-tune : akurasi val 0.75
sesudah fine-tune : akurasi val 1.0   (delta +0.25)
OOD "Lumayan standar saja sih" → skor 0.4988, yakin 0.0024 → JANGAN tampilkan
```

Konsep Bab 12 (abstain gate) kembali sebagai **kebijakan OOD**: model
menghitung `yakin = |p - ambang| * 2`; bila `yakin < skor_min (0.6)` → jangan
tampilkan label, arahkan ke manusia. LLM nyata tidak punya satu angka "yakin"
— pendekatannya: logprob token, self-consistency, rubrik judge (Bab 15), atau
classifier kecil di atas embedding. Lapisan abstain **tetap wajib** di kode.

---

## 2. Latihan

1. Jalankan `qwen2.5:3b` via Ollama; arahkan `openai` SDK ke
   `http://localhost:11434/v1` — buktikan kode Bab 11 tetap jalan.
2. Selesaikan **lab** [`01_lab_finetuning.ipynb`](01_lab_finetuning.ipynb)
   (8 bagian, angka terkunci) lalu jawab pertanyaan analisis di Bagian 8.
3. Kuis mandiri [`02_kuis_finetuning.ipynb`](02_kuis_finetuning.ipynb)
   (10 soal, 22 poin) — baru buka kunci setelah mencoba serius.
4. **Project TDD** [`project-ftkit/`](project-ftkit/README.md): lengkapi 6
   modul TODO sampai **44 test hijau**:

   ```bash
   cd project-ftkit
   python -m unittest discover -s tests -v
   ```

   Lalu kerjakan `starter.ipynb` (eksperimen + pertanyaan analisis), isi
   `RUBRIK.md`, dan baru bandingkan dengan `solusi/ftkit_ref.py`.
   Lihat angka terkunci tanpa menjalankan apa pun: `python _calib.py`.
5. Buat dataset SFT 300 contoh (JSONL) untuk gaya jawaban tertentu; fine-tune
   model 3B dengan QLoRA (Colab + Unsloth); bandingkan output sebelum/sesudah
   dengan eval set 30 kasus (Bab 15) — bukan "kelihatannya lebih baik".
6. Kuantisasi model ke Q4 (Ollama/llama.cpp); ukur perbedaan kualitas &
   kecepatan di task yang sama.

---

## 📚 Referensi Online

| Sumber | Tipe | Keterangan |
|---|---|---|
| [Ollama Docs](https://ollama.com/docs) | Dokumentasi | Menjalankan open-weight model lokal |
| [vLLM Docs](https://docs.vllm.ai/) | Dokumentasi | Serving throughput produksi |
| [HF PEFT (LoRA)](https://huggingface.co/docs/peft/index) | Dokumentasi | Referensi LoRA/QLoRA resmi |
| [HF TRL — SFTTrainer](https://huggingface.co/docs/trl/index) | Dokumentasi | SFT/DPO pipeline standar |
| [Unsloth notebooks](https://github.com/unslothai/unsloth) | Notebook | Fine-tune gratis & cepat di Colab |
| [LoRA paper](https://arxiv.org/abs/2106.09685) | Paper | Paper asli LoRA |
| [QLoRA paper](https://arxiv.org/abs/2305.14314) | Paper | LoRA di atas base 4-bit |
| [Alpaca dataset format](https://github.com/tatsu-lab/stanford_alpaca) | Repo | Format JSONL instruksi→respons |
| [The Alignment Handbook](https://github.com/huggingface/alignment-handbook) | Repo | Resep SFT/DPO produksi |

---

## ✅ Checklist Kompetensi

- [ ] Menjalankan open-weight model lokal & menghubungkannya ke kode API (Bab 11 tetap jalan)
- [ ] Menentukan RAG vs fine-tuning dengan alasan, bukan hype
- [ ] Membuat dataset SFT JSONL + split & oversampling deterministik
- [ ] Menjelaskan matematika LoRA (ΔW = B@A, B nol, rasio 0.78% @ rank 16) dan kapan rank naik
- [ ] Menjalankan satu fine-tune LoRA end-to-end + eval sebelum/sesudah berbasis angka
- [ ] Menjelaskan trade-off kuantisasi (fp16/int8/q4) dengan angka ukur sendiri
- [ ] Early stopping + restore-best dipahami (best epoch 61 ≠ epoch terakhir 66)
- [ ] Keputusan API vs lokal dihitung dari volume (breakeven ~6.061 req/hari)
- [ ] Kebijakan OOD/abstain aktif (Bab 12) — model boleh bilang "kurang yakin"
- [ ] Kuis bab ini selesai (22 poin) — atau review kunci sampai paham **kenapa**
- [ ] Project `ftkit` 44 test hijau + pertanyaan analisis terjawab
