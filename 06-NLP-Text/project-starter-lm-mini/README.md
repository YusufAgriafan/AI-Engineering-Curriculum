# 🏗️ Project Starter — LM Mini dari Nol (Bab 6)

> Kamu membangun **language model mini** dari nol: tokenisasi + padding, skip-gram
> embedding (negative sampling), bigram LM dengan temperature sampling, dan RNN
> char-level dengan BPTT — lalu membuktikan embedding bermakna dan RNN mengalahkan
> bigram. Semua numpy murni.

## Struktur

```
project-starter-lm-mini/
├── README.md              ← kamu di sini
├── RUBRIK.md              ← penilaian mandiri
├── starter.ipynb          ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb           ← notebook SOLUSI (buka setelah selesai / mentok)
├── lm_mini/               ← paket yang kamu isi (numpy murni)
│   ├── corpus.py          ← korpus mini terkunci (JANGAN DIUBAH)
│   ├── text.py            ← TODO: tokenize, vocab, encode aman, padding
│   ├── embeddings.py      ← TODO: pasangan skip-gram + skip-gram negative sampling
│   ├── languagemodel.py   ← TODO: bigram matrix, EE, sampling temperature
│   └── rnn.py             ← TODO: langkah RNN, forward, BPTT
├── tests/                 ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_text.py
│   ├── test_embeddings.py ← termasuk GRADIENT CHECK skip-gram
│   ├── test_languagemodel.py
│   └── test_rnn.py        ← termasuk GRADIENT CHECK BPTT
└── solusi/
    └── lm_mini_ref.py     ← implementasi referensi (untuk cek mandiri)
```

## Dataset: Korpus Mini Terkunci

Dua korpus kecil bahasa Indonesia (`corpus.py`, seed terkunci):

1. **Korpus topik** — kalimat satu-topik (hewan / kendaraan / makanan) + kata sifat
   positif/negatif + kata fungsional. Kategori kata diketahui — tapi hanya untuk
   **evaluasi** (purity), tidak pernah diberikan ke model.
2. **Korpus LM** — kalimat sederhana pola `<nama> <verba> <objek>` dan
   `<verba> <objek> itu <sifat>`; level karakter untuk bigram & RNN.

Kenapa task ini? Struktur makna (topik, kutub sentimen, pola kata) **muncul sendiri**
dari ko-okurensi. Kalau pipeline-mu benar, embedding mengelompok per kategori tanpa
pernah diberi tahu — demonstrasi paling murni dari *distributional hypothesis*.

## Apa yang Kamu Bangun

1. **Tokenisasi + padding** — vocab deterministik (terurut), OOV aman, pad/truncate
   post. Fondasi semua NLP; salah di sini = salah di semua lapisan di atasnya.
2. **Skip-gram + negative sampling** — loss `−log σ(v·u) + Σ −log σ(−v·u_neg)`,
   gradien `(σ(z) − y)·konteks` diverifikasi **gradient check** numerik.
3. **Bukti makna** — nearest-neighbor purity per kategori ≥ 0.9; avg-embedding +
   regresi logistik mengalahkan random embedding pada sentimen.
4. **Bigram LM** — smoothing `(1+α)`, cross-entropy (nats/char), sampling temperature
   yang benar perilakunya (T rendah repetitif, T tinggi beragam).
5. **RNN char-level** — langkah tanh, forward batch, **BPTT** dengan akumulasi gradien
   sepanjang waktu (gradient check), lalu val acc > bigram.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_text.py`.
2. Isi satu modul, jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan yang disarankan: `text.py` → `embeddings.py` → `languagemodel.py` →
   `rnn.py` sampai **semua 40 test hijau**.
4. Buka `starter.ipynb` — kerjakan eksperimen & jawab **Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/lm_mini_ref.py` / `solusi.ipynb`.

> 💡 Kalau gradient check gagal padahal loss turun: cek (1) gradien dihitung pada
> v/u **asli** sebelum update, (2) akumulasi `+=` sepanjang waktu di BPTT (bukan `=`),
> (3) normalisasi per baris di matriks bigram. Tiga itu penyebab 90% bug di project ini.

## Pertanyaan Analisis (bagian penilaian!)

1. Skip-gram tidak pernah diberi tahu kategori. **Kenapa** kata "kucing" dan "anjing"
   jadi tetangga, sementara "kucing" dan "nasi" terpisah? Jelaskan mekanismenya,
   bukan cuma hasilnya.
2. Apa yang hilang dari embedding kalau negative sampling **dihapus** (hanya pasangan
   asli)? (hint: semua vektor bisa "menang" bersama — coba eksperimen kecil di
   starter.ipynb, Bandingkan purity dengan dan tanpa negative.)
3. Kenapa avg-embedding bisa klasifikasi sentimen sedangkan one-hot mean hanya mampu
   menghafal? Kaitkan dengan hasil eksperimen fitur random-embedding-mu.
4. Temperature: tunjukkan dua sampel generasi bigram-mu (T rendah vs tinggi), lalu
   jelaskan kapan kamu memilih masing-masing saat memakai LLM API (Bab 11).
5. RNN mengalahkan bigram — **tapi apakah RNN-mu paham bahasa?** Apa yang sebenarnya
   dipelajari hidden state, dan apa bukti bahwa konteks 1 karakter tidak cukup?
6. Vanishing gradient: mengapa gradien RNN menurun sepanjang waktu, dan apa dua
   solusi arsitektural yang lahir dari masalah ini (arahkan ke Bab 9)?

## Prasyarat

```bash
pip install numpy matplotlib
```

> Bab ini memakai kembali pola dari Bab 4–5: lazy init bobot, backward dengan
> akumulasi `+=`, dan gradient check numerik. Konvolusi diganti konteks kata;
> pooling diganti hidden state.
