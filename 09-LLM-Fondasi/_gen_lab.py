"""Generate 01_lab_llm_fondasi.ipynb for Bab 9.

Struktur cell mengikuti pola Bab 8 (agar _verify_lab.py bisa memverifikasi):
  - cell GIVEN  : setup data / fungsi yang sudah jadi
  - cell TODO   : HANYA fungsi yang diisi mahasiswa (di-skip verifier, di-inject ref)
  - cell CEK    : assert murni (tidak mendefinisikan fungsi)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_lab_llm_fondasi.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = []

# ============================ HEADER ============================
cells.append(md(
    "# 🧪 Lab 9 — Fondasi LLM: Tokenisasi, Attention & Position dari Nol",
    "",
    "> Pendamping materi Bab 9. Prinsip sama dengan Lab 4–8: **semua dibangun dari nol",
    "> dengan numpy** — BPE, embedding yang belajar struktur, konteks window, positional",
    "> encoding, self-attention (encoder & decoder), sampai mini language model 2 blok",
    "> + strategi sampling. Supaya `AutoTokenizer`, `scaled dot-product attention`, dan",
    "> `temperature` tidak pernah jadi kotak hitam. Seed 9 terkunci — semua reproducible.",
    "",
    "| Bagian | Topik | Waktu (menit) |",
    "|---|---|---|",
    "| 1 | BPE dari nol: pair counting, merge, encode kata baru | 25 |",
    "| 2 | Token embedding yang belajar struktur (cosine) | 20 |",
    "| 3 | Konteks window & mean pooling (kenapa attention ditemukan) | 20 |",
    "| 4 | Positional encoding sinusoidal | 20 |",
    "| 5 | Self-attention + causal mask (encoder vs decoder) | 25 |",
    "| 6 | Mini Language Model: training loop, loss turun | 35 |",
    "| 7 | Sampling: greedy, temperature, top-k, top-p | 20 |",
    "| 8 | Ringkasan: apa yang tadi kamu bangun | 5 |",
    "",
    "Setiap latihan punya cell **✅ Cek** — jalankan, kalau ✅ berarti pemahamanmu benar.",
))

# ============================ BAGIAN 1 ============================
cells.append(md(
    "## Bagian 1 — BPE dari Nol: Subword Itu Dipelajari, Bukan Ditetapkan",
    "",
    "Vocabulary LLM modern (30rb–250rb entri) tidak dibuat tangan — ia **dipelajari**",
    "dari korpus dengan algoritma seperti Byte Pair Encoding. Mini-korpus kita:",
    "",
    "```",
    "low low lower lowest newer newer wider wider new new new",
    "```",
    "",
    "Aturan BPE yang kamu bangun:",
    "- Tiap kata dipecah jadi karakter + penanda akhir kata `</w>`.",
    "- Hitung semua pasangan bersebelahan; merge pasangan **paling sering**;",
    "  seri? pilih leksikografis terkecil (urutan kamus) — supaya deterministik.",
    "- Ulangi `n_merge` kali. Setiap merge = satu entri baru di vocabulary.",
))
cells.append(code(
    "from collections import Counter",
    "",
    "CORPUS = ['low', 'low', 'lower', 'lowest', 'newer', 'newer',",
    "          'wider', 'wider', 'new', 'new', 'new']",
    "",
    "def ke_simbol(kata):",
    "    # GIVEN: 'low' -> ('l', 'o', 'w', '</w>')",
    "    return tuple(kata) + ('</w>',)",
))
cells.append(code(
    "def hitung_pasangan(simbol_per_kata):",
    "    \"\"\"Counter semua pasangan bersebelahan dari list of tuple simbol.",
    "    Return collections.Counter.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "def merge_pasangan(simbol_per_kata, pasangan):",
    "    \"\"\"Gabungkan SETIAP kemunculan bersebelahan `pasangan` jadi satu simbol.",
    "    Contoh: ('l','o','w','</w>') dengan merge ('l','o') -> ('lo','w','</w>').",
    "    Return list of tuple baru.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "def best_pair(counter):",
    "    \"\"\"Pasangan dengan frekuensi tertinggi; seri -> leksikografis terkecil.",
    "    Petunjuk: min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0].\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# sym = [ke_simbol(k) for k in CORPUS]",
    "# print(hitung_pasangan(sym).most_common(5))",
    "# print(merge_pasangan(sym, ('e', 'r')))",
))
cells.append(code(
    "# ✅ Cek latihan 1.1",
    "sym = [ke_simbol(k) for k in CORPUS]",
    "pc = hitung_pasangan(sym)",
    "assert pc[('w', '</w>')] == 5, 'w</w> muncul di low(2) lowest(1) newer(1) wider(1)'",
    "assert pc[('e', 'r')] == 5 and pc[('n', 'e')] == 5 and pc[('e', 'w')] == 5",
    "assert pc[('l', 'o')] == 4, 'low 2x + lower 1x + lowest 1x'",
    "print('✅ Latihan 1.1 benar! Pair counting kamu presisi.')",
))
cells.append(code(
    "# ✅ Cek latihan 1.2 — tie-break deterministik",
    "assert best_pair(pc) == ('e', 'r'), f'dapat {best_pair(pc)} — lima pasangan seri 5, (e, r) terkecil leksikografis'",
    "print('✅ Latihan 1.2 benar! Merge pertama: (e, r).')",
    "print('   Deterministik = bisa diaudit. Tokenizer acak = reproduksibilitas hancur.')",
))
cells.append(code(
    "def latih_bpe(corpus, n_merge):",
    "    \"\"\"Jalankan n_merge iterasi BPE. Return list of (pasangan, frekuensi) berurutan.",
    "    Berhenti lebih awal jika tidak ada pasangan lagi.\"\"\"",
    "    # TODO: mulai dari [ke_simbol(k) for k in corpus]; loop: best_pair ->",
    "    # catat (pair, count) -> merge_pasangan.",
    "    raise NotImplementedError",
    "",
    "def encode_kata(kata, merges):",
    "    \"\"\"Encode kata BARU dengan daftar merges (urutan penting!).",
    "    Return tuple simbol. Karakter tak dikenal tetap jadi simbol sendiri —",
    "    itulah kekuatan subword: fallback selalu ada.\"\"\"",
    "    # TODO: mulai dari ke_simbol(kata); untuk tiap merge berurutan,",
    "    # merge jika pasangannya ada di simbol.",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# merges = latih_bpe(CORPUS, 4)",
    "# print(merges)",
    "# print(encode_kata('lower', merges))",
))
cells.append(code(
    "# ✅ Cek latihan 1.3 — urutan merge",
    "merges = latih_bpe(CORPUS, 4)",
    "urutan = list(merges)",
    "assert urutan[0] == (('e', 'r'), 5)",
    "assert urutan[1] == (('e', 'w'), 5), f'dapat {urutan[1]}'",
    "assert urutan[2] == (('er', '</w>'), 5), f'dapat {urutan[2]}'",
    "assert urutan[3] == (('n', 'ew'), 5), f'dapat {urutan[3]}'",
    "print('✅ Latihan 1.3 benar! 4 merge pertama: e-r, e-w, er-</w>, n-ew.')",
    "print('   Merge ketiga menempelkan </w>: BPE membangun token dari yang sering.')",
))
cells.append(code(
    "# ✅ Cek latihan 1.4 — encode kata baru",
    "assert encode_kata('lower', merges) == ('l', 'o', 'w', 'er</w>')",
    "assert encode_kata('new', merges) == ('new', '</w>'), 'merge ke-4 (n, ew) menciptakan token new'",
    "assert encode_kata('low', merges) == ('l', 'o', 'w', '</w>'), 'low kalah frekuensi dari er/ew'",
    "print('✅ Latihan 1.4 benar! encode = apply merges yang sudah dipelajari, berurutan.')",
    "print('   Kata langka tetap pecah per karakter — fallback selalu ada, makanya LLM')",
    "print('   bisa men-encode teks APA PUN tanpa token <UNK>.')",
))
cells.append(md(
    "### Mikro-eksperimen",
    "",
    "- Tambah frekuensi `new` (jadikan 20x). Merge pertama berubah? Kenapa?",
    "- Encode `newestt` (typo). Bagaimana hasilnya dibanding `newest`?",
    "- Naikkan `n_merge` ke 8 — token yang muncul lebih dulu: `low</w>` atau `lowest</w>`? Mengapa?",
    "",
    "> **Koneksi:** GPT-2 punya ~50k merge; kata Inggris umum biasanya 1 token,",
    "> kata Indonesia sering 2–4 — salah satu alasan token Indonesia lebih mahal.",
))

# ============================ BAGIAN 2 ============================
cells.append(md(
    "## Bagian 2 — Token Embedding: Vektor yang Belajar Struktur",
    "",
    "Embedding = tabel lookup `V x D` yang **dipelajari**, bukan tabel kode. Untuk",
    "melihat bahwa geometrinya menyimpan informasi, kita latih embedding 12 token",
    "dengan *fake task* yang direkayasa: token indeks 0, 4, 8 (sebut saja kelompok",
    "vokal) menerima signal `+1.5` di dua dimensi pertama setiap epoch. Kalau setelah",
    "training token satu kelompok berkerumun di ruang cosine, berarti **struktur",
    "memang muncul dari data** — persis seperti embedding kata asli.",
))
cells.append(code(
    "def cos_sim(a, b):",
    "    \"\"\"Cosine similarity dua vektor: dot / (norm * norm). Return float.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# print(cos_sim(np.array([1., 0.]), np.array([1., 0.])))   # 1.0",
    "# print(cos_sim(np.array([1., 0.]), np.array([0., 1.])))   # 0.0",
))
cells.append(code(
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "",
    "rng = np.random.default_rng(9)   # seed terkunci — seluruh lab reproducible",
    "",
    "V, D = 12, 8",
    "idx_vokal = [0, 4, 8]",
    "idx_konsonan = [1, 2, 5]",
    "SIGNAL = np.array([1.5, 1.5, 0, 0, 0, 0, 0, 0])",
    "",
    "def latih_embedding(vocab=V, d=D, epochs=300, seed=9):",
    "    # GIVEN: fake task — tiap epoch kelompok vokal digeser +SIGNAL, semua",
    "    # embedding didecay sedikit supaya tidak meledak. Return matrix (vocab, d).",
    "    r = np.random.default_rng(seed)",
    "    W = r.normal(0, 1, (vocab, d))",
    "    for _ in range(epochs):",
    "        W[idx_vokal] += SIGNAL",
    "        W *= 0.99",
    "    return W",
))
cells.append(code(
    "# ✅ Cek latihan 2.1 — struktur muncul di geometri",
    "W = latih_embedding()",
    "assert -1.0 <= cos_sim(W[0], W[1]) <= 1.0",
    "c_vokal = np.mean([cos_sim(W[i], W[j]) for i in idx_vokal for j in idx_vokal if i < j])",
    "c_kons = np.mean([cos_sim(W[i], W[j]) for i in idx_vokal for j in idx_konsonan])",
    "print(f'cos(vokal-vokal) rata-rata : {c_vokal:.4f}')",
    "print(f'cos(vokal-konsonan) rata   : {c_kons:.4f}')",
    "assert c_vokal > 0.4, f'expected ~0.55, dapat {c_vokal:.4f}'",
    "assert c_vokal > c_kons + 0.2, 'struktur harus terlihat jelas'",
    "print('✅ Latihan 2.1 benar! Token satu keluarga berkerumun di ruang cosine.')",
    "print('   Inilah prasyarat RAG (Bab 12): similaritas = kemiripan makna.')",
))
cells.append(code(
    "def analogi(a, b, c, W):",
    "    \"\"\"Selesaikan a : b :: c : ? — cari indeks vocab dengan cos tertinggi",
    "    ke vektor (W[b] - W[a] + W[c]), KECUALI a, b, c sendiri. Return int.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# print(analogi(0, 4, 1, W))",
))
cells.append(code(
    "# ✅ Cek latihan 2.2 — arah di ruang embedding bisa di-query",
    "target = SIGNAL / np.linalg.norm(SIGNAL)",
    "skor = [cos_sim(W[i], target) for i in range(V)]",
    "urut = np.argsort(skor)[::-1]",
    "assert urut[0] in idx_vokal, f'token paling searah signal harus vokal, dapat {urut[0]}'",
    "print('✅ Latihan 2.2 benar! Arah di ruang embedding = fitur yang bisa di-query.')",
    "print(f'   Top-3 searah signal: {urut[:3].tolist()} (kelompok vokal = {idx_vokal})')",
    "print('   Analogi king - man + woman = queen di word2vec: properti GEOMETRI yang sama.')",
    "",
    "hasil = analogi(0, 4, 1, W)",
    "print(f'   analogi(0, 4, 1) -> token {hasil} (harus searah kelompok vokal: {hasil in idx_vokal})')",
))
cells.append(md(
    "### Mikro-eksperimen",
    "",
    "- Ganti `seed=9` ke `seed=1`. Struktur (vokal berkerumun) tetap muncul? Harusnya ya —",
    "  struktur berasal dari DATA, seed hanya menggeser noise.",
    "- Naikkan `D` dari 8 ke 64. Rata-rata cosine antar kelompok turun mendekati 0 —",
    "  di dimensi tinggi vektor acak hampir ortogonal. Model besar butuh BANYAK contoh",
    "  per keluarga agar cluster-nya tetap terbaca.",
))

# ============================ BAGIAN 3 ============================
cells.append(md(
    "## Bagian 3 — Konteks Window & Mean Pooling: Attention Sebelum Ada Attention",
    "",
    "Cara paling sederhana memberi tiap token rasa konteks: gandengkan embedding",
    "tetangga kiri-kanannya (window `k`) — persis Conv1D di Bab 5. Lalu satukan",
    "seluruh kalimat jadi satu vektor dengan **mean pooling**. Kita akan buktikan",
    "kenapa mean pooling tidak bisa membedakan A-menyerang-B vs B-menyerang-A —",
    "dan itulah alasan positional encoding + attention ditemukan.",
))
cells.append(code(
    "def konteks_window(X, k):",
    "    \"\"\"Tiap baris output = concat embedding token [i-k .. i+k] (clamp ke tepi).",
    "    X: (n, d) -> output: (n, (2k+1)*d).\"\"\"",
    "    # TODO: untuk tiap i, kumpulkan X[j] untuk j dari i-k sampai i+k",
    "    # (dengan min/max agar tidak keluar rentang), lalu concatenate.",
    "    raise NotImplementedError",
    "",
    "def mean_pool(X):",
    "    \"\"\"(n, d) -> (d,) rata-rata SEMUA baris.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# X = rng.normal(0, 1, (5, 16))",
    "# print(konteks_window(X, 2).shape)",
    "# print(mean_pool(X).shape)",
))
cells.append(code(
    "X5 = rng.normal(0, 1, (5, 16))   # GIVEN data uji (seed 9)",
))
cells.append(code(
    "# ✅ Cek latihan 3.1 — window & clamp tepi",
    "C5 = konteks_window(X5, 2)",
    "assert C5.shape == (5, 80), f'5 token x (2*2+1) window x 16 dim, dapat {C5.shape}'",
    "assert np.allclose(C5[1], np.concatenate([X5[0], X5[0], X5[1], X5[2], X5[3]])), 'token 1 = [X0,X0,X1,X2,X3] (clamp kiri)'",
    "assert np.allclose(C5[3], np.concatenate([X5[1], X5[2], X5[3], X5[4], X5[4]])), 'token 3 = [X1..X4,X4] (clamp kanan)'",
    "print('✅ Latihan 3.1 benar! Konteks window = melihat tetangga dengan bobot STATIS.')",
    "print('   Di attention (Bagian 5): bobotnya dipelajari per-pasangan-token.')",
))
cells.append(code(
    "def vektor_kalimat(ids, W):",
    "    \"\"\"Representasi kalimat klasik (word2vec era): rata-rata embedding token.",
    "    Tanpa window, tanpa posisi. Return vektor (d,).\"\"\"",
    "    # TODO: X = W[np.asarray(ids)]; return mean_pool(X)",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 3.2 — mean pooling = bag of words",
    "Wa = rng.normal(0, 1, (4, 16))",
    "s1 = vektor_kalimat([0, 1, 2], Wa)   # A menyerang B",
    "s2 = vektor_kalimat([2, 1, 0], Wa)   # B menyerang A (dibalik)",
    "assert np.allclose(s1, s2), 'mean pooling: urutan lenyap — dua makna, satu vektor'",
    "print('✅ Latihan 3.2 benar! s1 == s2 — mean pooling TIDAK bisa bedakan subjek vs objek.')",
    "",
    "# Bonus: dengan window k=1, vektor jadi beda — tapi dari mana bedanya?",
    "s1w = mean_pool(konteks_window(Wa[np.asarray([0, 1, 2])], 1))",
    "s2w = mean_pool(konteks_window(Wa[np.asarray([2, 1, 0])], 1))",
    "print(f'   dengan window k=1: selisih max = {np.abs(s1w - s2w).max():.4f}')",
    "print('   selisih itu murni efek tepi (clamping), bukan pemahaman hubungan antar kata.')",
    "print('   → Positional encoding (Bagian 4) + attention (Bagian 5) yang menyelesaikan ini.')",
))

# ============================ BAGIAN 4 ============================
cells.append(md(
    "## Bagian 4 — Positional Encoding: Menyuntikkan Urutan",
    "",
    "Self-attention itu permutation-invariant — tanpa posisi, A-menyerang-B =",
    "B-menyerang-A (Bagian 3 baru saja membuktikannya). Solusi orisinal Transformer:",
    "tambahkan **sinusoidal PE** ke embedding:",
    "",
    "```",
    "PE(pos, 2i)   = sin(pos / 10000^(2i/d))",
    "PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
    "```",
    "",
    "Desain ini jenius karena PE hanya fungsi dari POSISI: nilai bounded [-1, 1],",
    "bisa di-extend ke posisi yang tidak pernah dilihat saat training, dan",
    "dot product PE[p] . PE[q] hanya fungsi dari selisih (p - q) — relative position.",
))
cells.append(code(
    "def positional_encoding(max_len, d):",
    "    \"\"\"Return (max_len, d): kolom genap sin, ganjil cos; freq = 10000^(-2i/d).",
    "    Petunjuk vektorisasi: div = np.exp(-np.log(10000) * np.arange(0, d, 2) / d);",
    "    PE[:, 0::2] = np.sin(pos * div); PE[:, 1::2] = np.cos(pos * div).\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "def rpe(pos, d):",
    "    \"\"\"Satu baris tabel PE di atas — PE untuk satu posisi. Return (d,) float64.\"\"\"",
    "    # TODO: return positional_encoding(pos + 1, d)[pos]",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# PE = positional_encoding(12, 16)",
    "# print(PE.shape, PE[0, :4])",
))
cells.append(code(
    "# ✅ Cek latihan 4.1 — properti kunci PE",
    "PE = positional_encoding(12, 16)",
    "assert PE.shape == (12, 16)",
    "assert np.all(PE >= -1) and np.all(PE <= 1), 'sin/cos bounded'",
    "assert np.allclose(PE[0, 0::2], 0.0), 'sin(0) = 0'",
    "assert np.allclose(PE[0, 1::2], 1.0), 'cos(0) = 1'",
    "assert np.allclose(PE[1, 0], np.sin(1.0)), 'pos=1, freq tertinggi (i=0): sin(1)'",
    "assert np.allclose(rpe(3, 16), PE[3]), 'rpe = satu baris tabel'",
    "print('✅ Latihan 4.1 benar! PE[0] = [0,1,0,1,...]; PE[3,:4] =', PE[3, :4].round(4))",
))
cells.append(code(
    "# ✅ Cek latihan 4.2 — relative position lewat dot product",
    "PE20 = positional_encoding(20, 16)",
    "d1 = PE20[0] @ PE20[3]",
    "d2 = PE20[5] @ PE20[8]",
    "d3 = PE20[2] @ PE20[5]",
    "assert np.allclose([d1, d2, d3], d1, atol=1e-9), 'dot PE_p·PE_q hanya fungsi selisih (p-q)'",
    "assert not np.allclose(d1, PE20[0] @ PE20[5]), 'jarak beda -> dot beda'",
    "print('✅ Latihan 4.2 benar! PE_p . PE_q = fungsi murni dari (p - q).')",
    "print('   Identitas: sin p sin q + cos p cos q = cos((p-q)f), dijumlah per frekuensi.')",
    "print('   Model bisa membaca RELATIVE position dari q·k — tanpa tabel tambahan.')",
    "print(f'   dot jarak 3 = {d1:.6f} | dot jarak 5 = {PE20[0] @ PE20[5]:.6f}')",
))
cells.append(code(
    "ids = [0, 1, 2]",
    "ids_rev = [2, 1, 0]",
    "d_model = 16",
    "Wfull = rng.normal(0, 1, (4, d_model))   # GIVEN (seed 9)",
))
cells.append(code(
    "def embedding_dengan_posisi(ids, W, PE_tab):",
    "    \"\"\"Sum embedding + PE per posisi. Return (len(ids), d_model).\"\"\"",
    "    # TODO: X = W[np.asarray(ids)]; return X + PE_tab[:len(ids)]",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 4.3 — posisi menyelamatkan makna",
    "PE16 = positional_encoding(10, d_model)",
    "H1 = embedding_dengan_posisi(ids, Wfull, PE16)",
    "H2 = embedding_dengan_posisi(ids_rev, Wfull, PE16)",
    "assert not np.allclose(H1, H2), 'dengan PE, urutan tidak lagi tak terlihat'",
    "E1 = Wfull[np.asarray(ids)]",
    "E2 = Wfull[np.asarray(ids_rev)]",
    "assert np.allclose(E1, E2[::-1]), 'tanpa PE: kalimat dibalik = baris dibalik (simetri)'",
    "assert not np.allclose(H1, H2[::-1]), 'dengan PE: simetri pecah — posisi absolut ada'",
    "print('✅ Latihan 4.3 benar! Tanpa PE: H(flip) = flip(H) — simetri penuh.')",
    "print('   Dengan PE: simetri pecah. A-menyerang-B != B-menyerang-A.')",
))

# ============================ BAGIAN 5 ============================
cells.append(md(
    "## Bagian 5 — Self-Attention: Q, K, V dan Causal Mask",
    "",
    "Sekarang inti Transformer. Tiap token diproyeksikan ke tiga peran:",
    "",
    "```",
    "Q = X @ Wq   (apa yang kucari?)    K = X @ Wk   (apa yang kutawarkan?)",
    "V = X @ Wv   (informasi yang kubawa)",
    "",
    "Scores = Q @ K^T / sqrt(d_k)       # kecocokan tiap pasangan token",
    "Attn   = softmax per baris         # bobot tiap baris jumlah 1",
    "Output = Attn @ V                  # campuran berbobot informasi tetangga",
    "```",
    "",
    "Untuk **decoder** (GPT-style) tambahkan **causal mask**: posisi i dilarang",
    "melihat posisi j > i — kalau tidak, training next-token prediction = curang.",
    "Implementasi: tambahkan -1e9 ke skor segitiga atas sebelum softmax.",
))
cells.append(code(
    "np.random.seed(9)   # Q, K, V terkunci agar angka di cek reproducible",
    "Q = np.random.randn(4, 8)",
    "K = np.random.randn(4, 8)",
    "V = np.random.randn(4, 8)",
))
cells.append(code(
    "def softmax_stabil(x, axis=-1):",
    "    \"\"\"Softmax numerically stable: kurangi max sebelum exp.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "def self_attention(Q, K, V, causal=False):",
    "    \"\"\"Return (output, attn_weights); Q, K, V: (n, d_k).",
    "    causal=True -> tambahkan -1e9 ke skor segitiga atas sebelum softmax.\"\"\"",
    "    # TODO: scores = Q @ K.T / sqrt(d_k);",
    "    # if causal: scores = scores + np.triu(np.full((n, n), -1e9), k=1);",
    "    # attn = softmax_stabil(scores, axis=-1); return attn @ V, attn",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# out, attn = self_attention(Q, K, V)",
    "# print(out.shape, attn.shape, attn.sum(axis=-1))",
))
cells.append(code(
    "# ✅ Cek latihan 5.1 — properti dasar",
    "out, attn = self_attention(Q, K, V)",
    "assert attn.shape == (4, 4) and out.shape == (4, 8)",
    "assert np.allclose(attn.sum(axis=-1), 1.0), 'tiap baris softmax jumlah 1'",
    "assert np.all(attn > 0), 'softmax selalu positif'",
    "print('✅ Latihan 5.1 benar! out (4,8), attn (4,4); baris attn jumlah 1.')",
))
cells.append(code(
    "# ✅ Cek latihan 5.2 — kenapa dibagi sqrt(d_k)?",
    "def softmax_naive(x, axis=-1):",
    "    e = np.exp(x)",
    "    return e / e.sum(axis=axis, keepdims=True)",
    "",
    "scores_scaled = Q @ K.T / np.sqrt(8)",
    "scores_raw = Q @ K.T",
    "naive = softmax_naive(scores_scaled)",
    "hancur = softmax_naive(scores_raw)",
    "assert np.allclose(attn, naive), 'hasil kamu = softmax(scores / sqrt(d_k))'",
    "assert not np.allclose(attn, hancur), 'tanpa scaling, distribusi jadi ekstrem'",
    "print('✅ Latihan 5.2 benar! Tanpa / sqrt(d_k): varians skor ~ d_k -> softmax satu-hot.')",
    "print('   Satu-hot = gradien mati. sqrt(d_k) menjaga varians skor ~ 1.')",
    "print(f'   max prob: tanpa scaling {hancur.max():.4f} vs dengan scaling {attn.max():.4f}')",
))
cells.append(code(
    "# ✅ Cek latihan 5.3 — causal mask",
    "out_m, attn_m = self_attention(Q, K, V, causal=True)",
    "assert np.allclose(np.triu(attn_m, k=1), 0.0), 'posisi masa depan HARUS 0'",
    "assert np.allclose(attn_m[0], [1, 0, 0, 0]), 'token 0 cuma lihat dirinya'",
    "assert np.allclose(attn_m.sum(axis=-1), 1.0), 'tetap distribusi valid'",
    "S9 = Q @ K.T / np.sqrt(8)",
    "expected = softmax_stabil(np.array([[S9[2, 0], S9[2, 1], S9[2, 2], -1e9]]))[0]",
    "assert np.allclose(attn_m[2], expected), 'baris 2 = softmax([s0, s1, s2, -1e9])'",
    "print('✅ Latihan 5.3 benar! Mask = segitiga atas -1e9, softmax sisanya biasa.')",
    "print('   Baris 0 cuma lihat token 0: GPT belajar next-token TANPA mengintip masa depan.')",
    "print('   attn_m[2] =', attn_m[2].round(3))",
))
cells.append(md(
    "### Mikro-eksperimen",
    "",
    "- Bandingkan `attn` vs `attn_m`: baris mana yang berubah? (Baris terakhir paling banyak —",
    "  dia punya paling banyak masa depan yang kini terlarang.)",
    "- Ganti seed Q/K/V: properti struktural (baris jumlah 1, mask 0) TIDAK pernah berubah.",
    "- Buat d_k besar (mis. Q, K: (4, 512)) dan lihat softmax TANPA scaling makin ekstrem —",
    "  inilah kenapa scaling bukan opsional.",
))

# ============================ BAGIAN 6 ============================
cells.append(md(
    "## Bagian 6 — Mini Language Model: Training dari Nol",
    "",
    "Rakit semuanya: embedding + PE + 2 blok dense (ReLU) + output projection,",
    "dilatih dengan **next-token prediction**. Dataset: pola `A B A B A B A B`",
    "(vocab 4 token) — cukup untuk melihat loss turun dan model menangkap pola,",
    "tanpa GPU. Gradient dihitung **finite-difference** — sengaja, agar fokusmu di",
    "ARSITEKTUR (backprop manual sudah kamu bayar di Bab 4; di Bab 13 keluar lagi",
    "lewat training loop TensorFlow).",
    "",
    "```",
    "ids → We[ids] + PE → blok1(ReLU) → blok2(ReLU) → Wo → logits (V4)",
    "loss = cross-entropy(logits[:-1], ids[1:])   ← shift: prediksi token BERIKUTNYA",
    "```",
))
cells.append(code(
    "V4, D4, NLAY = 4, 16, 2",
    "SEQ = [0, 1, 0, 1, 0, 1, 0, 1]   # A B A B A B A B",
))
cells.append(code(
    "def one_hot(ids, vocab):",
    "    \"\"\"(n,) int -> (n, vocab) one-hot float.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "def cross_entropy(logits, targets):",
    "    \"\"\"Mean cross-entropy. logits: (n, vocab), targets: (n,) int.",
    "    Pakai log-softmax stabil: z = logits - max; logp = z - log(sum(exp(z))).\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "def softmax_rows(x):",
    "    \"\"\"Softmax baris-per-baris (versi stabil).\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# print(one_hot([1, 2], 4))",
    "# print(cross_entropy(np.zeros((2, 4)), np.array([1, 2])))  # ~ln(4) = 1.386",
))
cells.append(code(
    "# ✅ Cek latihan 6.1",
    "oh = one_hot([1, 2], 4)",
    "assert oh.shape == (2, 4) and oh.sum(axis=1).tolist() == [1.0, 1.0]",
    "assert oh[0, 1] == 1.0 and oh[1, 2] == 1.0",
    "ce = cross_entropy(np.zeros((2, 4)), np.array([1, 2]))",
    "assert abs(ce - np.log(4)) < 1e-6, f'uniform -> CE = ln(vocab), dapat {ce:.4f}'",
    "print('✅ Latihan 6.1 benar! CE(uniform, target apa pun) = ln(vocab) =', round(np.log(4), 4))",
    "print('   → Baseline teoretis: model buta harus loss ~ 1.386.')",
))
cells.append(code(
    "def init_params(seed=9, d=D4, vocab=V4):",
    "    # GIVEN: parameter awal (seed 9) — We, PE (beku), 2 blok, output projection",
    "    r = np.random.default_rng(seed)",
    "    return {",
    "        'We': r.normal(0, 0.5, (vocab, d)),",
    "        'PE': positional_encoding(32, d) * 0.1,",
    "        'W1': r.normal(0, 0.5, (d, d)), 'b1': np.zeros(d),",
    "        'W2': r.normal(0, 0.5, (d, d)), 'b2': np.zeros(d),",
    "        'Wo': r.normal(0, 0.1, (d, vocab)), 'bo': np.zeros(vocab),",
    "    }",
    "",
    "def forward(params, ids):",
    "    # GIVEN: embedding + PE, lalu 2 blok dense ReLU, lalu logits.",
    "    # (Versi mini yang sengaja tanpa attention di dalam loop — gradient",
    "    # finite-difference di cell berikutnya tetap benar untuk arsitektur apa pun.)",
    "    n = len(ids)",
    "    X = params['We'][np.asarray(ids)] + params['PE'][:n]",
    "    A1 = np.maximum(X @ params['W1'] + params['b1'], 0.0)",
    "    A2 = np.maximum(A1 @ params['W2'] + params['b2'], 0.0)",
    "    return A2 @ params['Wo'] + params['bo']",
    "",
    "def loss_fn(params, ids):",
    "    # GIVEN: next-token CE — logits di posisi t diprediksi dari ids[t+1]",
    "    logits = forward(params, ids)",
    "    return cross_entropy(logits[:-1], np.asarray(ids[1:]))",
))
cells.append(code(
    "# ✅ Cek latihan 6.2 — loss awal dekat ln(4) (model masih buta)",
    "params = init_params()",
    "loss0 = loss_fn(params, SEQ)",
    "assert 1.0 < loss0 < 1.9, f'loss awal harus dekat ln(4)=1.386, dapat {loss0:.3f}'",
    "print(f'✅ Latihan 6.2 benar! Loss awal = {loss0:.3f} ~ ln(4) = 1.386 (belum tahu apa-apa).')",
))
cells.append(code(
    "def numeric_grads(f, params, seq, eps=1e-5):",
    "    # GIVEN: finite-difference gradient untuk semua parameter float.",
    "    grads = {}",
    "    for k, arr in params.items():",
    "        g = np.zeros_like(arr)",
    "        flat, gflat = arr.ravel(), g.ravel()",
    "        for i in range(len(flat)):",
    "            orig = flat[i]",
    "            flat[i] = orig + eps; fp = f(params, seq)",
    "            flat[i] = orig - eps; fm = f(params, seq)",
    "            flat[i] = orig",
    "            gflat[i] = (fp - fm) / (2 * eps)",
    "        grads[k] = g",
    "    return grads",
    "",
    "def train_lm(params, seq, epochs=150, lr=0.06):",
    "    # GIVEN: gradient descent murni; PE sengaja TIDAK dilatih (fixed).",
    "    losses = []",
    "    trainables = [k for k in params if k != 'PE']",
    "    for _ in range(epochs):",
    "        grads = numeric_grads(loss_fn, params, seq)",
    "        for k in trainables:",
    "            params[k] = params[k] - lr * grads[k]",
    "        losses.append(loss_fn(params, seq))",
    "    return params, losses",
))
cells.append(code(
    "# ✅ Cek latihan 6.3 — training: loss turun jauh di bawah baseline",
    "trained, losses = train_lm(init_params(), SEQ, epochs=150, lr=0.06)",
    "assert len(losses) == 150",
    "assert losses[-1] < 0.5, f'loss akhir harus < 0.5, dapat {losses[-1]:.3f}'",
    "assert losses[-1] < losses[0] * 0.4, 'harus turun drastis dari loss awal'",
    "print(f'✅ Latihan 6.3 benar! Loss: {losses[0]:.3f} -> {losses[-1]:.3f} dalam 150 epoch.')",
    "",
    "plt.figure(figsize=(7, 3.5))",
    "plt.plot(losses)",
    "plt.axhline(np.log(4), color='gray', ls='--', lw=1, label='baseline buta ln(4)=1.386')",
    "plt.xlabel('epoch'); plt.ylabel('next-token CE'); plt.title('Mini-LM: Loss Training')",
    "plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()",
))
cells.append(code(
    "def prediksi_berikut(params, ids):",
    "    \"\"\"Return indeks token dengan logits tertinggi di posisi terakhir.\"\"\"",
    "    # TODO: logits = forward(params, ids); return int(np.argmax(logits[-1]))",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 6.4 — model menangkap pola A-B bergantian",
    "assert prediksi_berikut(trained, [0]) == 1, 'A -> prediksi B'",
    "assert prediksi_berikut(trained, [0, 1]) == 0, 'A B -> prediksi A'",
    "assert prediksi_berikut(trained, [0, 1, 0]) == 1, 'A B A -> prediksi B'",
    "print('✅ Latihan 6.4 benar! Model menangkap pola bergantian A-B.')",
    "print('   Kemampuan yang sama dipakai LLM besar untuk melanjutkan teks apa pun.')",
))

# ============================ BAGIAN 7 ============================
cells.append(md(
    "## Bagian 7 — Sampling: Temperature, Top-k, Top-p",
    "",
    "LLM menghasilkan logits, bukan teks. Cara logits menjadi token = **strategi",
    "sampling** — inilah kenapa jawaban LLM kadang kreatif, kadang kaku:",
    "",
    "```",
    "greedy        : argmax — deterministik, cenderung repetitif",
    "temperature T : logits / T sebelum softmax; T kecil = tajam, T besar = datar",
    "top-k         : sampling hanya dari k token probabilitas tertinggi",
    "top-p         : sampling dari nucleus — token terkecil yang kumulatifnya >= p",
    "```",
))
cells.append(code(
    "V8 = 8",
    "LOGITS8 = np.array([0.1, 2.2, 0.5, 1.3, 0.1, 0.9, 0.2, 0.4])   # GIVEN, terkunci",
))
cells.append(code(
    "def sample_greedy(logits):",
    "    \"\"\"Return int argmax.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "def sample_topk(logits, k, rng, T=1.0):",
    "    \"\"\"Sampling dari k token probabilitas tertinggi (setelah temperature).",
    "    Return int.\"\"\"",
    "    # TODO: idx = argsort(logits)[::-1][:k];",
    "    # p = softmax_stabil(logits[idx] / T); return rng.choice(idx, p=p)",
    "    raise NotImplementedError",
    "",
    "def sample_topp(logits, p_min, rng, T=1.0):",
    "    \"\"\"Nucleus: kumpulkan token terurut menurun sampai kumulatif prob >= p_min",
    "    (minimal 1 token). Return int.\"\"\"",
    "    # TODO: order = argsort(logits)[::-1]; p = softmax(order/T);",
    "    # cum = cumsum(p); kc = 1 + searchsorted(cum, p_min, side='left');",
    "    # sampling dari order[:kc]",
    "    raise NotImplementedError",
    "",
    "# --- uji sendiri ---",
    "# r = np.random.default_rng(9)",
    "# print(sample_greedy(LOGITS8), sample_topk(LOGITS8, 2, r), sample_topp(LOGITS8, 0.9, r))",
))
cells.append(code(
    "# ✅ Cek latihan 7.1",
    "r9 = np.random.default_rng(9)",
    "assert sample_greedy(LOGITS8) == 1, 'argmax [0.1, 2.2, ...] adalah indeks 1'",
    "for _ in range(50):",
    "    assert sample_topk(LOGITS8, 2, r9) in (1, 3), 'top-2 hanya dari {1, 3}'",
    "for _ in range(30):",
    "    assert sample_topp(LOGITS8, 0.6, r9) in (1, 3, 5), 'nucleus p=0.6 = {1, 3, 5}'",
    "for _ in range(30):",
    "    assert sample_topp(LOGITS8, 0.3, r9) == 1, 'nucleus p=0.3 = {1} saja (deterministik)'",
    "print('✅ Nucleus: p=0.3 -> {1} (greedy-like); p=0.6 -> {1,3,5}; p=0.9 -> 7 token!')",
    "print('   p tinggi hampir tidak memotong apa pun — p rendah yang mengubah perilaku.')",
    "counts = {1: 0, 3: 0}",
    "r100 = np.random.default_rng(0)",
    "for _ in range(200):",
    "    counts[sample_topk(LOGITS8, 2, r100)] += 1",
    "print('✅ Latihan 7.1 benar! Greedy = 1; top-2 hanya {1, 3}.')",
    "print(f'   Distribusi top-2 (200x): {counts} — token 1 lebih sering, prob-nya lebih besar.')",
))
cells.append(code(
    "# ✅ Cek latihan 7.2 — efek temperature",
    "def dist_after_T(logits, T):",
    "    return softmax_stabil(np.asarray(logits, dtype=float) / T)",
    "",
    "p_rendah = dist_after_T(LOGITS8, 0.5)",
    "p_tinggi = dist_after_T(LOGITS8, 2.0)",
    "assert p_rendah.max() > dist_after_T(LOGITS8, 1.0).max() > p_tinggi.max(), 'T kecil -> tajam; T besar -> datar'",
    "print(f'✅ Latihan 7.2 benar! Max prob: T=0.5 -> {p_rendah.max():.3f}, T=2.0 -> {p_tinggi.max():.3f}')",
    "print('   T -> 0 mirip greedy; T -> besar mirip uniform. T = kenop kreatif vs kaku.')",
    "",
    "plt.figure(figsize=(7, 3.5))",
    "x = np.arange(V8)",
    "plt.bar(x - 0.25, dist_after_T(LOGITS8, 0.5), width=0.25, label='T=0.5 (tajam)')",
    "plt.bar(x,        dist_after_T(LOGITS8, 1.0), width=0.25, label='T=1.0')",
    "plt.bar(x + 0.25, dist_after_T(LOGITS8, 2.0), width=0.25, label='T=2.0 (datar)')",
    "plt.xticks(x); plt.xlabel('token id'); plt.ylabel('probabilitas')",
    "plt.title('Efek Temperature terhadap Distribusi Sampling')",
    "plt.legend(); plt.tight_layout(); plt.show()",
))
cells.append(md(
    "### Mikro-eksperimen",
    "",
    "- `top_k=1` = greedy? Buktikan dengan 20 sample berturut-turut.",
    "- `p_min=1.0` = sampling dari semua token. Bedanya dengan tanpa top-p sama sekali?",
    "- Chatbot butuh konsisten: T kecil + top-p 0.9. Penulis cerita butuh variasi: T 1.0–1.3.",
    "  **Kenapa?** Tulis jawabanmu sebelum lanjut.",
    "",
    "> **Koneksi ke Bab 10:** parameter `temperature` dan `top_p` di API LLM persis",
    "> mengubah fungsi `dist_after_T` dan nucleus yang barusan kamu bangun.",
))

# ============================ BAGIAN 8 ============================
cells.append(md(
    "## Bagian 8 — Ringkasan: Angka-Angka yang Kamu Pegang",
    "",
    "| Bagian | Bukti pemahaman |",
    "|---|---|",
    "| 1. BPE | 4 merge pertama: e-r, e-w, er-</w>, n-ew; encode kata baru via merges |",
    "| 2. Embedding | cos(vokal-vokal) ~ 0.55 vs cos(vokal-konsonan) ~ 0.24 |",
    "| 3. Konteks | window clamp tepi; mean pooling = bag of words (urutan lenyap) |",
    "| 4. PE | PE[0] = [0,1,0,1,...]; PE_p·PE_q = f(p−q); H(dibalik) = flip(H) |",
    "| 5. Attention | baris-sum 1; causal baris 0 = [1,0,0,0]; sqrt(d_k) mencegah satu-hot |",
    "| 6. Mini-LM | loss 1.386 -> < 0.5; prediksi A-B bergantian benar |",
    "| 7. Sampling | greedy = 1; top-2 = {1,3}; T naik -> distribusi datar |",
    "",
    "### Koneksi ke bab lain",
    "",
    "- **Bab 10:** sampling strategy = parameter API (`temperature`, `top_p`).",
    "- **Bab 12 (RAG):** cosine similarity Bagian 2 = inti vector search.",
    "- **Bab 13 (Fine-tuning):** training loop Bagian 6 = versi mini dari training TF.",
    "- **Bab 15 (Evaluasi):** loss turun != jawaban bagus — LLM butuh evaluasi lain.",
    "",
    "Lanjut ke: **kuis** (`02_kuis_llm_fondasi.ipynb`) lalu **project starter**",
    "(`project-starter-tokenizer-mini/`) — bangun tokenizer mini end-to-end yang",
    "bisa men-encode teks apa pun, plus embedding + mini-LM.",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT, f"({len(cells)} cells)")
