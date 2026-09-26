"""Generator notebook kunci jawaban kuis Bab 12 (RAG) — menulis 03_kunci_jawaban_kuis_rag.ipynb.

Pola sama dengan _gen_kunci.py Bab 11: jawaban + KENAPA (bukan sekadar benar/salah),
kode referensi tiap soal coding, bisa dijalankan penuh (setup sama dengan kuis).
"""
import json
from pathlib import Path

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [source]})


def code(source):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [source]})


md("""# 🔑 Kunci Jawaban — Kuis Bab 12 (RAG)

> Buka **setelah** mencoba. Fokus pembahasan: **kenapa**, bukan sekadar benar/salah.
> Semua contoh angka di bawah terkunci (hashing deterministik) — kamu bisa
> menjalankan ulang dan mendapat angka yang sama persis.""")

md("""## Setup (sama dengan kuis — untuk menjalankan kode referensi)""")

code("""import hashlib
import math
import re

_DIM = 128
_TOKEN_PAT = re.compile(r'[a-z0-9]+')
_OFFSET = [int.from_bytes(hashlib.sha256(f'kuis12-offset-{i}'.encode()).digest()[:4], 'big') / 2**32
           for i in range(_DIM)]


def tokenisasi(teks):
    return _TOKEN_PAT.findall(str(teks).lower())


def _bucket(gram, salt=''):
    digest = hashlib.sha256(f'kuis12|{salt}|{gram}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') % _DIM, (1.0 if digest[4] % 2 == 0 else -1.0)


def vektor(teks):
    vec = [0.0] * _DIM
    tokens = tokenisasi(teks)
    for tok in tokens:
        idx, tanda = _bucket(tok)
        vec[idx] += tanda
        if len(tok) > 3:
            idx5, t5 = _bucket(tok[:5], salt='pre')
            vec[idx5] += 0.3 * t5
    for a, b in zip(tokens, tokens[1:]):
        idx, tanda = _bucket(a + '_' + b, salt='bi')
        vec[idx] += 0.5 * tanda
    bercampur = [v + 0.01 * _OFFSET[i] for i, v in enumerate(vec)]
    norm = math.sqrt(sum(v * v for v in bercampur))
    return [v / norm for v in bercampur] if norm else [0.0] * _DIM


CHUNKS = [
    {'chunk_index': 0, 'source': 'kebijakan-ketenagakerjaan.pdf', 'page': 1,
     'text': 'CUTI TAHUNAN. Karyawan tetap mendapatkan 12 hari cuti tahunan per tahun kalender. '
             'Cuti tahunan harus digunakan dalam tahun yang sama dan tidak dapat ditumpuk ke tahun berikutnya.'},
    {'chunk_index': 1, 'source': 'kebijakan-cuti-kesehatan.pdf', 'page': 1,
     'text': 'CUTI SAKIT. Cuti sakit maksimal 5 hari per kejadian tanpa dokumen medis. '
             'Cuti sakit lebih dari 90 hari masuk asuransi dan dikonsultasikan dengan HR.'},
    {'chunk_index': 2, 'source': 'faq-karyawan.pdf', 'page': 1,
     'text': 'FAQ CUTI. Bagaimana cara mengajukan cuti? Submit form cuti tahunan via portal HR '
             'minimal 14 hari sebelumnya. Berapa hari cuti tahunan? 12 hari per tahun kalender untuk karyawan tetap.'},
]

Q1 = 'Berapa hari cuti tahunan untuk karyawan tetap?'
Q6 = 'Berapa harga tiket pesawat ke Bali?'

SEP_TIDAK_ADA = 'tidak ada di dokumen'
print('setup siap')""")

md("""## Bagian A — Pilihan Ganda""")

md("""### Soal 1 — Jawaban: **b) load → chunk → embed → index → retrieve → generate**

Urutan itu bukan selera — tiap langkah **mengonsumsi output langkah sebelumnya**:

```
LOAD     : dokumen mentah (PDF/HTML/DOCX) → teks
CHUNK    : teks panjang → potongan yang muat di konteks LLM tanpa rusak maknanya
EMBED    : tiap chunk → vektor (bisa dihitung hanya SETELAH potongannya final!)
INDEX    : simpan vektor + metadata agar bisa dicari cepat
RETRIEVE : kueri → vektor kueri → cari chunk terdekat (query time)
GENERATE : konteks terpilih + pertanyaan → jawaban + sitasi
```

Kenapa opsi lain salah:

- **a)** meng-embed sebelum load/chunk = meng-embed apa? Dan retrieve sebelum
  index berarti mencari di rak kosong.
- **c)** chunk sebelum load dan generate sebelum retrieve = dua operasi mundur.
- **d)** hanya `retrieve → generate` yang urutannya benar; sisanya terbalik.

Dua pembagian yang penting diingat: LOAD-CHUNK-EMBED-INDEX = **ingestion**
(dijalankan sekali / periodic, mahal), RETRIEVE-GENERATE = **query time**
(dijalankan tiap permintaan, harus cepat). Salah memindahkan langkah antar
fase = index basi atau biaya tak terkendali.""")

md("""### Soal 2 — Jawaban: **c) chunk kehilangan konteks; informasi lintas-kalimat tidak terjawab utuh**

Chunk yang terlalu kecil memecah **unit makna**. Dua kegagalan konkret:

1. *Kehilangan konteks lokal* — "Cuti sakit maksimal 5 hari" dan "tanpa dokumen
   medis" bisa terpisah ke dua chunk; tiap chunk sekarang punya makna berbeda
   (atau tidak bermakna), embedding-nya menyimpang, retrieval meleset.
2. *Kegagalan lintas-chunk* — pertanyaan yang jawabannya tersebar di dua kalimat
   ("berapa hari + apa syaratnya") tidak pernah muncul utuh di satu konteks;
   generator hanya melihat setengah bukti.

Kenapa opsi lain salah:

- **a)** kebalikan kenyataan: chunk terlalu kecil justru *menurunkan* presisi
  retrieval karena sinyal kata per chunk terlalu tipis.
- **b)** halusinasi tidak hilang dengan chunk kecil; yang menahan halusinasi
  adalah prompt ber-aturan + abstain gate + sitasi (Bagian 5 lab).
- **d)** ukuran & overlap adalah keputusan desain dengan trade-off terukur —
  di lab Bagian 1 kamu mengukur dampaknya ke hit@k, bukan menebak.

Aturan praktis (produksi): 300–800 token per chunk, overlap 10–20%, potong di
batas natural (kalimat/paragraf/heading). Dan karena semua keputusan chunking
mengubah embedding → **index harus dibangun ulang** saat chunking berubah.""")

md("""### Soal 3 — Jawaban: **b) abstain bila skor terbaik di bawah ambang**

Vector store **selalu** mengembalikan "paling mirip" — ia tidak tahu arti
"tidak ada". q6 (tiket pesawat ke Bali) tetap mendapat chunk terbaik dengan skor
~0.052 (bandingkan: kueri dalam korpus ~0.45–0.67). Skor rendah itu satu-satunya
sinyal yang jujur: *kemiripan leksikal kebetulan, bukan jawaban*.

Abstain gate mengubah sinyal itu jadi perilaku produk:

```
skor_terbaik >= ambang  -> biarkan jawaban lewat
skor_terbaik <  ambang  -> timpa: "tidak ada di dokumen", kosongkan sitasi
sudah abstain           -> dibiarkan (jangan turunkan abstain jadi jawaban!)
```

Kenapa opsi lain salah:

- **a)** memaksa LLM menjawab dari konteks tak relevan = undangan halusinasi —
  dan yang dibaca pengguna tetap salah walau modelnya "hanya mengarang dari konteks".
- **c)** menurunkan ambang untuk menghapus abstain = menyembunyikan gejala;
  kueri tak terjawab jadi dijawab salah dengan percaya diri.
- **d)** masalahnya bukan index: korpus memang tidak memuat info tiket pesawat.
  Embed ulang 1000x tidak menghasilkan informasi baru.

Trade-off yang perlu diukur (Bab 15): ambang terlalu tinggi → banyak abstain
padahal jawabannya ada (recall turun); terlalu rendah → jawaban salah lolos
(precision turun). Di lab: 0.30 bekerja untuk korpus ini; uji 0.15 dan 0.45.""")

md("""### Soal 4 — Jawaban: **c) hybrid menemani makna (vektor) dengan exact match (BM25), digabung RRF**

Embedding menangkap *makna*, tapi lemah untuk **token eksak**: nama produk,
kode tiket, nomor dokumen, istilah teknis. Dua kata berbeda bisa saling menggeser
vektor (hash collision / polisemi), dan istilah langka bisa "hilang" di antara
kata umum. BM25 kebalikannya: buta makna tapi **sangat teliti pada kata** (idf
memberi bobot besar pada kata langka yang cocok).

Hybrid = ambil top-k dari keduanya, lalu fusi dengan RRF:

```
skor_rrf(chunk) = Σ_daftar  1/(k + peringkat + 1)     # peringkat 0-based
```

Angka terkunci (project Bab 12): pure vektor hit@3 = 1.0 di korpus ini, tapi
hit@1 = 0.8 — dan hybrid memperbaiki stabilitas peringkat pada kueri
ber-kata-kunci. Untuk kueri "iPhone 15" pada korpus tanpa "iPhone", BM25 tetap
jalan dengan skor 0 (tidak crash) — vektor yang mencoba berdasarkan makna umum.

Kenapa opsi lain salah:

- **a)** hybrid bukan pengganti embedding; ia bekerja *bersama* embedding.
- **b)** hybrid menjalankan **dua** pencarian + fusi → lebih mahal, bukan lebih
  cepat. Yang dibeli adalah *kualitas peringkat*, bukan kecepatan.
- **d)** biaya embedding kueri tetap ada (vektor tetap dicari); BM25 menambah
  biaya tokenisasi + skoring, bukan menghapus.

Kapan layak (pertanyaan analisis project #4): kueri pendek ber-kata-kunci eksak,
korpus campuran istilah teknis. Kapan tidak: korpus kecil seragam dengan latensi
ketat — cukup satu retriever.""")

md("""### Soal 5 — Jawaban: **a) 1/61 — hanya kontribusi daftar tempat ia muncul**

RRF menjumlahkan per daftar; daftar yang tidak memuat chunk berkontribusi nol:

```
peringkat 0 (0-based) di satu daftar -> 1/(60 + 0 + 1) = 1/61
muncul juga di daftar kedua, peringkat 1 -> tambah 1/(60 + 1 + 1) = 1/62
total -> 1/61 + 1/62
```

Bukti terkunci dari kuis (soal 9): `[[c2, c0], [c1, c0]]` →
`{2: 1/61, 0: 2/62, 1: 1/61}` — c0 menang dengan 2/62 = 0.0323 walau ia peringkat
kedua di kedua daftar; konsistensi di banyak daftar mengalahkan peringkat atas
di satu daftar.

Kenapa opsi lain salah:

- **b)** `k=60` adalah *konstanta peredam*, bukan skor maksimum; ia menekan
  dominasi peringkat-atas supaya sinyal dari daftar kedua masih berarti.
- **c)** tidak ada syarat "muncul di semua daftar" — justru itu gunanya fusi.
- **d)** k memperkecil tiap kontribusi (1/(k+r+1)), bukan menjadi skor.

Dua jebakan klasik (terjadi di kode nyata): memakai peringkat **1-based** saat
rumus mengharapkan 0-based (semua skor bergeser, urutan hybrid berubah), dan
melupakan `+1` sehingga peringkat terakhir mendapat pembagian nol.""")

md("""### Soal 6 — Jawaban: **b) menghemat embedding + pencarian untuk kueri sama/senilai; jawaban tetap digenerate segar**

Yang mahal dan berulang di query time adalah **retrieval** (embed kueri +
pencarian +ambil chunks). Cache retrieval menyimpan *konteks terpilih* per kunci
kueri ternormalisasi (`strip().lower()` — "Berapa" dan "berapa" adalah permintaan
yang sama). Angka terkunci project: pola `miss, hit, hit, miss` → hanya **2
pencarian** untuk 4 permintaan, hit-rate 0.5, latensi hit 2 ms vs miss 16 ms.

Bedanya dengan cache LLM (Bab 11): yang di-cache **bukan jawaban** — generate
tetap berjalan tiap permintaan. Kenapa desain ini justru benar:

1. Kualitas jawaban bisa naik tanpa cache basi (versi prompt/generator baru tetap
   menghasilkan jawaban baru dari konteks lama yang masih valid).
2. Konteks retrieval lebih stabil daripada jawaban: chunk index berubah hanya
   saat korpus berubah.
3. TTL tetap wajib — korpus berubah (kebijakan direvisi) → cache harus kedaluwarsa.

Kenapa opsi lain salah:

- **a)** itu cache jawaban (tingkat provider/prompt), bukan cache retrieval;
  menggabungkan keduanya tanpa sadar = jawaban basi disajikan percaya diri.
- **c)** cache tidak menaikkan kualitas; ia hanya menghindari pekerjaan ulang
  yang hasilnya identik.
- **d)** index tetap inti; cache menutupi *pengulangan*, bukan kebutuhan index.

Produksi: kunci harus mencakup semua yang mengubah hasil — kueri (ternormalisasi),
**versi index/korpus**, top_k, filter metadata. Lupa versi index = pengguna
menerima konteks dari korpus lama setelah dokumen diperbarui.""")

md("""## Bagian B — Coding""")

md("""### Soal 7 — `cosine`

Tiga jebakan: (1) vektor **nol** harus menghasilkan `0.0`, bukan `ZeroDivisionError`
atau NaN — vektor nol sah muncul dari teks tanpa token dikenal; (2) normalisasi
kedua vektor, bukan hanya salah satu; (3) jangan hitung panjang vektor berbeda
sebagai error — `zip` memotong ke yang terpendek (kontrak soal).""")

code("""def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

print(round(cosine(vektor(Q1), vektor(Q1)), 6))   # 1.0
print(cosine([1.0, 0.0], [0.0, 1.0]))             # 0.0 (tegak lurus)
print(cosine([0.0, 0.0], [1.0, 1.0]))             # 0.0 (bukan error!)
print(cosine([1.0, 0.0], [-1.0, 0.0]))            # -1.0 (berlawanan arah)""")

md("""### Soal 8 — `cari_vektor`

Empat hal yang diuji: skor cosine (bukan dot product mentah), **urutan menurun**,
**tie-break deterministik** (index kecil dulu supaya hasil bisa direproduksi dan
diuji), dan penghormatan `top_k`. Angka terkunci — perhatikan FAQ (chunk 2) menang
top-1 untuk Q1: ia memuat kata kueri berulang ("cuti tahunan", "12 hari",
"karyawan tetap") — fenomena yang sama dengan hit@1 = 0.8 di project!""")

code("""def cari_vektor(kueri, top_k=2):
    skor = []
    for i, c in enumerate(CHUNKS):
        skor.append((cosine(vektor(kueri), vektor(c['text'])), i))
    urut = sorted(skor, key=lambda t: (-t[0], t[1]))[:top_k]
    return [{'chunk': CHUNKS[i], 'score': s, 'index': i} for s, i in urut]

for h in cari_vektor(Q1, 3):
    print(h['index'], round(h['score'], 4), h['chunk']['source'])
print('skor terbaik q6:', round(cari_vektor(Q6, 1)[0]['score'], 4))   # 0.052""")

md("""### Soal 9 — `gabung_rrf`

Jebakan utama: peringkat **0-based** → penyebut `k + rank + 1`, bukan `k + rank`
(divisi nol saat rank 0!) dan bukan `k + rank` dengan rank 1-based (offset satu
mengubah urutan akhir). Kedua, akumulasi: chunk yang muncul di beberapa daftar
**menjumlah** kontribusi (`skor.get(idx, 0.0) + ...`).""")

code("""def gabung_rrf(daftar_dapat, k=60):
    skor = {}
    for dapat in daftar_dapat:
        for rank, item in enumerate(dapat):
            idx = item['chunk']['chunk_index']
            skor[idx] = skor.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return skor

a = [{'chunk': {'chunk_index': 2}}, {'chunk': {'chunk_index': 0}}]
b = [{'chunk': {'chunk_index': 1}}, {'chunk': {'chunk_index': 0}}]
skor = gabung_rrf([a, b])
print({k_: round(v, 6) for k_, v in skor.items()})
print('urutan:', sorted(skor, key=lambda k_: (-skor[k_], k_)))   # [0, 1, 2]""")

md("""### Soal 10 — `abstaikan`

Tiga kontrak yang paling sering dilanggar (dan diuji): (1) kembalikan **dict
BARU** — input tidak boleh berubah (pemanggil mungkin masih memakainya untuk
audit); (2) `skor_terbaik >= ambang` **lolos** — tepat di ambang tidak abstain
(batas inklusif); (3) hasil yang **sudah abstain dibiarkan** — jangan pernah
turunkan abstain jadi jawaban, dan jangan menambahkan `abstaikan_oleh` pada
abstain yang bukan palingan gate. Kunci `abstaikan_oleh` adalah bukti audit:
*skor* yang memicu timpaan.""")

code("""def abstaikan(hasil, ambang=0.30):
    skor = hasil.get('skor_terbaik', 0.0)
    if hasil.get('abstain') or skor >= ambang:
        return dict(hasil)
    keluar = dict(hasil)
    keluar['jawaban'] = SEP_TIDAK_ADA
    keluar['abstain'] = True
    keluar['abstaikan_oleh'] = skor
    return keluar

h = {'jawaban': '12 hari', 'sitasi': [0], 'abstain': False,
     'skor_terbaik': cari_vektor(Q6, 1)[0]['score']}
out = abstaikan(h)
print(out['jawaban'], '| abstain oleh:', round(out['abstaikan_oleh'], 4))
print('input utuh  :', h['jawaban'])   # tidak berubah""")

md("""## Rekap Kunci Pilihan Ganda

| Soal | Kunci | Inti |
|---|---|---|
| 1 | **b** | load → chunk → embed → index → retrieve → generate |
| 2 | **c** | chunk kecil kehilangan konteks & gagal lintas-kalimat |
| 3 | **b** | skor rendah → abstain, bukan jawab paksa |
| 4 | **c** | hybrid = makna (vektor) + exact match (BM25), fusi RRF |
| 5 | **a** | RRF: hanya daftar yang memuat chunk yang berkontribusi |
| 6 | **b** | cache konteks = hemat retrieval; jawaban tetap segar |

Coding: `cosine` (vektor nol → 0.0) · `cari_vektor` (urut + tie-break) ·
`gabung_rrf` (Σ 1/(k+rank+1), 0-based) · `abstaikan` (dict baru, batas inklusif,
abstain dibiarkan).""")

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent / "03_kunci_jawaban_kuis_rag.ipynb"
out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"OK menulis {out} — {len(CELLS)} cells")
