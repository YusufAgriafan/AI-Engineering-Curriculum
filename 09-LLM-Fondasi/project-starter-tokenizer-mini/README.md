# 🏗️ Project Starter — Tokenizer & Mini-LM dari Nol (Bab 9)

> Kamu membangun **satu sistem bahasa mini end-to-end** dari nol: BPE trainer +
> tokenizer dengan vocab & round-trip, embedding yang menyimpan struktur makna,
> positional encoding sinusoidal, scaled dot-product attention dengan causal mask,
> mini language model yang dilatih dengan next-token prediction, dan strategi
> sampling (greedy / top-k / top-p). Semua numpy murni, TDD **76 test**.

## Struktur

```
project-starter-tokenizer-mini/
├── README.md              ← kamu di sini
├── RUBRIK.md              ← penilaian mandiri
├── starter.ipynb          ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb           ← notebook SOLUSI (buka setelah selesai / mentok)
├── tok_mini/              ← paket yang kamu isi (numpy murni)
│   ├── data.py            ← konstanta terkunci (JANGAN DIUBAH)
│   ├── bpe.py             ← TODO: pair counting, merge, latih, apply
│   ├── tokenizer.py       ← TODO: vocab, encode/decode, round-trip
│   ├── embedding.py       ← TODO: cosine, cari_mirip, analogi
│   ├── position.py        ← TODO: positional encoding sinusoidal
│   ├── attention.py       ← TODO: softmax stabil + self-attention
│   ├── lm.py              ← SEBAGIAN GIVEN: loss_fn + train_lm (TODO kamu)
│   └── sampling.py        ← TODO: greedy, top-k, top-p
├── tests/                 ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_bpe.py
│   ├── test_tokenizer.py
│   ├── test_embedding.py
│   ├── test_position.py
│   ├── test_attention.py
│   ├── test_lm.py
│   └── test_sampling.py
├── _verify_project.py     ← verifier otomatis (patch referensi + jalankan test)
└── solusi/
    └── tok_mini_ref.py    ← implementasi referensi (untuk cek mandiri)
```

## Dataset: Mini-Corpus Terkunci

```
low low lower lowest newer newer wider wider new new new      (seed 9)
```

Kenapa sekecil ini? **Struktur kebenarannya diketahui** — kita tahu persis merge
yang harus ditemukan (`er`, `ew`, `er</w>`, `new`), pola yang harus dipelajari
mini-LM (A-B bergantian), dan nucleus yang harus terbentuk ({1,3,5} pada
p=0.6). Di data nyata kebenaran itu tak pernah ada; test yang kamu buat hijau di
sini yang menggantikannya sebagai bukti pemahaman.

## Apa yang Kamu Bangun

1. **BPE dari nol** — pair counting deterministik (tie-break leksikografis),
   merge berulang, encode kata baru dengan fallback karakter: teks apa pun bisa
   di-encode tanpa `<UNK>`.
2. **Tokenizer end-to-end** — vocab 0..V-1 tanpa celah, `encode_teks` →
   `decode_teks` round-trip persis; kata sering hemat token, kata baru pecah.
3. **Embedding** — cosine aman nol, retrieval top-n, analogi vektor; bukti
   struktur muncul dari DATA (bukan label) via fake task terkalibrasi.
4. **Positional encoding** — sin/cos bounded, properti `PE_p·PE_q = f(p−q)`.
5. **Self-attention** — softmax stabil, `sqrt(d_k)`, causal mask; encoder vs
   decoder hanya berbeda mask.
6. **Mini-LM** — loss next-token dengan SHIFT target; training finite-difference;
   loss 1.386 → < 0.01 dan generalisasi ke prompt yang tak pernah dilihat.
7. **Sampling** — greedy/top-k/top-p dengan temperature; nucleus nonlinier
   (p=0.3 → 1 token, p=0.9 → 7 token).

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_bpe.py`.
2. Isi satu modul, jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan yang disarankan: `bpe.py` → `tokenizer.py` → `embedding.py` →
   `position.py` → `attention.py` → `lm.py` → `sampling.py` sampai **semua
   76 test hijau** (catatan: `test_lm.py` butuh ~20 detik — finite-difference).
4. Buka `starter.ipynb` — kerjakan 6 eksperimen & jawab **Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/tok_mini_ref.py` / `solusi.ipynb`.

> 💡 Kalau test macet: cek (1) tie-break leksikografis saat seri di `best_pair`,
> (2) merges di-apply BERURUTAN di encode, (3) target loss adalah `ids[1:]`
> (shift!), (4) softmax LOKAL pada kandidat top-k/top-p, bukan seluruh vocab.
> Empat itu penyebab 90% bug di project ini.

## Pertanyaan Analisis (bagian penilaian!)

1. Anatomi token: kenapa `newer` 2 token tapi `lowliest` 9 — dan apa implikasinya
   bagi biaya API LLM?
2. Kurva total token vs jumlah merge: bentuknya bagaimana, dan kenapa tokenizer
   produksi berhenti di ~50rb–250rb merge?
3. Struktur kelompok muncul di matrix cosine TANPA label. Mekanismenya apa — dan
   apa efek menaikkan dimensi terhadap noise cosine?
4. Baris attention mana yang identik antara encoder & decoder, kenapa persis
   baris itu — dan apa gejala jika causal mask dilupakan saat training?
5. Tiga strategi sampling menghasilkan 'kepribadian' berbeda. Konfigurasi apa
   yang kamu pilih untuk chatbot layanan pelanggan, dan apa risikonya?
6. Mini-LM memprediksi benar pola yang tak pernah dilihat (`[1] -> 0`). Apa yang
   sebenarnya dipelajari model, dan apa kaitannya dengan generalisasi LLM besar?

## Prasyarat

```bash
pip install numpy matplotlib
```

> Bab ini memakai kembali pola dari Bab 4 (training loop) dan Bab 6 (language
> modeling). Yang diuji bukan kecanggihan model tapi **pemahaman mekanika di
> balik LLM** — tokenizer, attention, dan sampling yang nanti kamu pakai
> setiap hari sebagai AI Engineer lewat API.
