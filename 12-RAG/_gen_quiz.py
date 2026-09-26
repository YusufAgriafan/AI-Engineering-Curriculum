"""Generator notebook kuis Bab 12 (RAG) — menulis 02_kuis_rag.ipynb.

Pola sama dengan _gen_quiz.py Bab 11: 10 soal (6 PG + 4 coding), 22 poin,
penilaian otomatis, mandiri (encoder mini + korpus mini disediakan di setup,
tanpa jaringan, tanpa API key). Semua angka terkunci hasil _calib_kuis.py
(hash deterministik, bukan acak).
"""
import json
from pathlib import Path

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [source]})


def code(source):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [source]})


md("""# 📝 Kuis Bab 12 — RAG: Retrieval Augmented Generation

10 soal (6 pilihan ganda + 4 coding), total **22 poin**. Penilaian otomatis di cell terakhir.

**Aturan:** kerjakan tanpa membuka materi/lab. Soal coding diisi di bawah `# TODO`,
lalu jalankan cell penilaian. Kuis ini **mandiri** — encoder mini deterministik +
korpus mini sudah disediakan di cell setup (tanpa jaringan, tanpa API key).
Angka-angka (skor cosine, RRF) terkunci karena hashing-nya deterministik:
teks yang sama → vektor yang sama, di komputer siapa pun.""")

md("""## Setup (GIVEN) — Encoder Mini Deterministik + Korpus Mini

`vektor(teks)` mengubah teks jadi vektor ternormalisasi L2 dengan **hashing
trick** (versi mini dari `ragkit/embed.py`): tiap kata dipetakan ke satu bucket
via hash SHA-256 (tanda +/- dari bit hash), lalu vektor dinormalisasi.
Konsekuensinya: teks yang berbagi kata → cosine similarity tinggi; teks tanpa
kata sama → cosine ≈ 0 (bisa sedikit negatif karena tanda hash).

Korpus mini: 3 chunk kebijakan HR (sama temanya dengan lab/project).
""")

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

print('setup siap —', len(CHUNKS), 'chunk, dimensi', _DIM)""")

md("""## Bagian A — Pilihan Ganda (isi `JAWABAN`)""")

code("""# Isi jawaban pilihan ganda di sini, contoh: JAWABAN[1] = 'a'

JAWABAN = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
}""")

md("""### Soal 1 — Urutan pipeline RAG

Manakah urutan pipeline RAG yang BENAR?

- a) embed → load → chunk → retrieve → index → generate
- b) load → chunk → embed → index → retrieve → generate
- c) chunk → load → index → embed → generate → retrieve
- d) retrieve → embed → index → chunk → load → generate""")

md("""### Soal 2 — Trade-off chunk size

Kamu memotong dokumen menjadi chunk **sangat kecil** (mis. 80 karakter, tanpa
overlap). Konsekuensi yang PALING mungkin:

- a) Retrieval jadi lebih akurat karena tiap chunk sangat fokus
- b) Jawaban halusinasi pasti hilang
- c) Chunk kehilangan konteks — kalimat yang terpotong jadi tidak bermakna dan
  pertanyaan yang butuh dua kalimat tidak terjawab utuh
- d) Tidak ada konsekuensi; ukuran chunk hanya soal selera""")

md("""### Soal 3 — Skor luar korpus

Kueri q6 "Berapa harga tiket pesawat ke Bali?" tidak dijawab korpus kebijakan HR.
Vector store tetap mengembalikan hasil terbaik dengan skor ~0.05 (rendah).
Perlakuan yang BENAR:

- a) Tetap jawab pakai chunk skor tertinggi — LLM pasti bisa memilah sendiri
- b) Abstain: bila skor terbaik di bawah ambang, jawab "tidak ada di dokumen" —
  lebih jujur daripada menjawab dari konteks yang tidak relevan
- c) Turunkan ambang supaya semua kueri mendapat jawaban
- d) Hapus korpusnya dan embed ulang — masalahnya pasti di index""")

md("""### Soal 4 — Kenapa hybrid search

Vector search bisa meleset untuk kueri dengan **kata kunci eksak** (nama produk,
kode tiket). Fungsi utama hybrid search (BM25 + vektor):

- a) Menggantikan embedding model dengan yang lebih besar
- b) Mempercepat pencarian dibanding vector search murni
- c) Menemani kemiripan makna (vektor) dengan exact match kata kunci (BM25),
  lalu menggabungkan keduanya — biasanya dengan Reciprocal Rank Fusion
- d) Memangkas biaya embedding kueri menjadi nol""")

md("""### Soal 5 — Skor RRF

Reciprocal Rank Fusion dengan `k=60`: chunk A peringkat **0** (0-based) di satu
daftar dan **tidak muncul** di daftar lain. Skor RRF-nya:

- a) 1/61 — bila ia muncul di satu daftar saja, hanya kontribusi daftar itu
- b) 1/60 — peringkat 0 berarti skor maksimal
- c) 0 — ia harus muncul di semua daftar
- d) 60 — kontribusinya sebesar konstanta k""")

md("""### Soal 6 — Cache konteks retrieval

Cache retrieval menyimpan **hasil pencarian** (chunks) per kueri, bukan jawaban
LLM. Manfaat utamanya:

- a) Meniadakan biaya generate LLM — jawaban langsung dari cache
- b) Menghemat embedding kueri + pencarian pada kueri yang sama/senilai; jawaban
  tetap digenerate segar dari konteks yang tersimpan
- c) Meningkatkan kualitas retrieval karena chunks di-cache lebih baik dari hasil baru
- d) Menggantikan kebutuhan index vector""")

md("""## Bagian B — Coding""")

md("""### Soal 7 — `cosine(a, b)` (3 poin)

`cosine(a, b)` → cosine similarity dua vektor (list of float):

- `cosine(x, x)` = 1.0 untuk vektor apa pun yang bukan nol;
- bila salah satu vektor **nol** (norm 0), kembalikan `0.0` — bukan error, bukan NaN;
- bila `b` lebih panjang dari `a`, pasangkan hanya elemen yang ada
  (`zip` cukup — soal ini sengaja menuntut perilaku itu).

Contoh: `cosine([1.0, 0.0], [0.0, 1.0])` → `0.0`.""")

code("""def cosine(a, b):
    # TODO
    pass

# print(round(cosine(vektor(Q1), vektor(Q1)), 6))   # 1.0
# print(cosine([1.0, 0.0], [0.0, 1.0]))             # 0.0
# print(cosine([0.0, 0.0], [1.0, 1.0]))             # 0.0""")

md("""### Soal 8 — `cari_vektor(kueri, top_k)` (5 poin)

`cari_vektor(kueri, top_k=2)` → list dict hasil pencarian cosine di `CHUNKS`:

- vektor chunk dihitung dari `vektor(c['text'])`, vektor kueri dari `vektor(kueri)`;
- tiap hasil: `{'chunk': <chunk>, 'score': <cosine>, 'index': <posisi di CHUNKS>}`;
- **urut skor menurun**; seri dipecah dengan index kecil dulu (tie-break deterministik);
- `top_k` membatasi jumlah hasil.

Angka terkunci (deterministik — bukan kebetulan): `cari_vektor(Q1, 3)` →
index `[2, 0, 1]` (FAQ menang top-1!) dengan skor `[0.6659, 0.4546, 0.2897]`;
skor terbaik kueri luar korpus `Q6` = `0.0520`.""")

code("""def cari_vektor(kueri, top_k=2):
    # TODO
    pass

# for h in cari_vektor(Q1, 3):
#     print(h['index'], round(h['score'], 4), h['chunk']['source'])
# print('skor terbaik q6:', round(cari_vektor(Q6, 1)[0]['score'], 4))""")

md("""### Soal 9 — `gabung_rrf(daftar_dapat, k=60)` (5 poin)

Reciprocal Rank Fusion. `daftar_dapat` = list of list of hasil
(`{'chunk': {'chunk_index': ...}, ...}`) dari beberapa pencarian:

- skor chunk = **Σ 1/(k + peringkat + 1)** — peringkat **0-based** (kesalahan
  offset satu mengubah urutan hybrid!);
- chunk yang muncul di beberapa daftar **menjumlah** kontribusinya;
- return **dict** `{chunk_index: skor}` — tidak perlu mengurutkan.

Contoh terkunci: `daftar = [[c2, c0], [c1, c0]]` (c0 = chunk_index 0, dst.) →
`{2: 1/61, 0: 2/62, 1: 1/61}` — chunk 0 menang karena muncul di dua daftar.""")

code("""def gabung_rrf(daftar_dapat, k=60):
    # TODO
    pass

# a = [{'chunk': {'chunk_index': 2}}, {'chunk': {'chunk_index': 0}}]
# b = [{'chunk': {'chunk_index': 1}}, {'chunk': {'chunk_index': 0}}]
# print({k_: round(v, 6) for k_, v in gabung_rrf([a, b]).items()})""")

md("""### Soal 10 — `abstaikan(hasil, ambang=0.30)` (5 poin)

`abstaikan(hasil, ambang=0.30)` — gate "tidak ada di dokumen":

- `hasil` = dict berisi minimal `{'jawaban', 'abstain', 'skor_terbaik'}`;
- bila `skor_terbaik < ambang` dan hasil **belum** abstain → kembalikan **dict
  BARU** dengan: `jawaban = 'tidak ada di dokumen'`, `abstain = True`,
  `abstaikan_oleh = skor_terbaik`;
- bila sudah abstain ATAU skor >= ambang → salinan hasil tanpa perubahan
  (tanpa kunci `abstaikan_oleh`);
- `skor_terbaik` yang hilang dianggap `0.0` (langsung abstain);
- **input tidak boleh berubah**.

Angka terkunci: skor q6 `0.0520` < `0.30` → abstain oleh gate.""")

code("""def abstaikan(hasil, ambang=0.30):
    # TODO
    pass

# h = {'jawaban': '12 hari', 'abstain': False, 'skor_terbaik': cari_vektor(Q6, 1)[0]['score']}
# out = abstaikan(h)
# print(out['jawaban'], '| abstain oleh:', out.get('abstaikan_oleh'))
# print('input utuh:', h['jawaban'])""")

md("""## 🧮 Penilaian Otomatis""")

code("""SEP_TIDAK_ADA = 'tidak ada di dokumen'
total_bobot = 22


def jalankan_kuis():
    skor = 0
    rincian = []

    def cek_pilihan(nomor, kunci, bobot=1):
        nonlocal skor
        benar = JAWABAN.get(nomor) == kunci
        skor += bobot if benar else 0
        rincian.append((nomor, benar))

    def cek_kode(nomor, uji, bobot=1):
        nonlocal skor
        benar = False
        try:
            benar = bool(uji())
        except Exception as e:
            print(f'  soal {nomor}: error -> {type(e).__name__}: {e}')
        skor += bobot if benar else 0
        rincian.append((nomor, benar))

    def _safe(nama, *args, **kwargs):
        fn = globals().get(nama)
        if fn is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    cek_pilihan(1, 'b')
    cek_pilihan(2, 'c')
    cek_pilihan(3, 'b')
    cek_pilihan(4, 'c')
    cek_pilihan(5, 'a')
    cek_pilihan(6, 'b')

    # Soal 7 - cosine
    def uji7():
        assert abs(_safe('cosine', [1.0, 0.0], [0.0, 1.0])) < 1e-9
        assert abs(_safe('cosine', [1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-9
        assert _safe('cosine', [0.0, 0.0], [1.0, 1.0]) == 0.0
        assert _safe('cosine', [1.0, 1.0], [0.0, 0.0]) == 0.0
        assert _safe('cosine', [], [1.0]) == 0.0            # vektor kosong = norm 0
        assert abs(_safe('cosine', [1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-9
        return True
    cek_kode(7, uji7, bobot=2)

    # Soal 8 - cari_vektor
    def uji8():
        hasil = _safe('cari_vektor', Q1, 3)
        assert hasil is not None, 'cari_vektor belum jadi'
        assert [h['index'] for h in hasil] == [2, 0, 1], [h['index'] for h in hasil]
        assert abs(hasil[0]['score'] - 0.6659) < 1e-3, hasil[0]['score']
        assert abs(hasil[1]['score'] - 0.4546) < 1e-3, hasil[1]['score']
        assert abs(hasil[2]['score'] - 0.2897) < 1e-3, hasil[2]['score']
        assert hasil[0]['chunk']['source'] == 'faq-karyawan.pdf'
        top1 = _safe('cari_vektor', Q1, 1)
        assert len(top1) == 1
        h6 = _safe('cari_vektor', Q6, 1)[0]
        assert abs(h6['score'] - 0.0520) < 1e-3, h6['score']
        urutan_skor = [h['score'] for h in _safe('cari_vektor', 'cuti sakit dokter', 3)]
        assert urutan_skor == sorted(urutan_skor, reverse=True)
        return True
    cek_kode(8, uji8, bobot=5)

    # Soal 9 - gabung_rrf
    def uji9():
        a = [{'chunk': {'chunk_index': 2}}, {'chunk': {'chunk_index': 0}}]
        b = [{'chunk': {'chunk_index': 1}}, {'chunk': {'chunk_index': 0}}]
        skor = _safe('gabung_rrf', [a, b])
        assert skor is not None, 'gabung_rrf belum jadi'
        assert abs(skor[0] - 2 / 62) < 1e-9, skor
        assert abs(skor[1] - 1 / 61) < 1e-9, skor
        assert abs(skor[2] - 1 / 61) < 1e-9, skor
        satu = _safe('gabung_rrf', [a], k=10)
        assert abs(satu[2] - 1 / 11) < 1e-9, satu
        kosong = _safe('gabung_rrf', [])
        assert kosong == {}, kosong
        return True
    cek_kode(9, uji9, bobot=5)

    # Soal 10 - abstaikan
    def uji10():
        h = {'jawaban': '12 hari', 'sitasi': [0], 'abstain': False,
             'skor_terbaik': 0.0649}
        out = _safe('abstaikan', h)
        assert out is not None, 'abstaikan belum jadi'
        assert out['jawaban'] == SEP_TIDAK_ADA, out['jawaban']
        assert out['abstain'] is True
        assert abs(out['abstaikan_oleh'] - 0.0649) < 1e-9
        assert h['jawaban'] == '12 hari', 'input tidak boleh berubah'

        h2 = {'jawaban': '12 hari', 'abstain': False, 'skor_terbaik': 0.30}
        out2 = _safe('abstaikan', h2, 0.30)                 # tepat ambang: lolos
        assert out2['jawaban'] == '12 hari' and 'abstaikan_oleh' not in out2

        h3 = {'jawaban': SEP_TIDAK_ADA, 'abstain': True, 'skor_terbaik': 0.10}
        out3 = _safe('abstaikan', h3)                       # sudah abstain: dibiarkan
        assert out3['abstain'] is True and 'abstaikan_oleh' not in out3

        out4 = _safe('abstaikan', {'jawaban': 'x', 'abstain': False})   # tanpa skor
        assert out4['abstain'] is True and out4['jawaban'] == SEP_TIDAK_ADA
        return True
    cek_kode(10, uji10, bobot=4)

    print('=' * 46)
    for nomor, benar in rincian:
        print(f'  Soal {nomor:>2}: {"BENAR" if benar else "salah"}')
    print('=' * 46)
    print(f'SKOR AKHIR: {skor}/{total_bobot}')
    if skor == total_bobot:
        print('Sempurna! Lanjut ke project starter ragkit.')
    elif skor >= 17:
        print('Bagus! Review soal yang salah, lalu lanjut.')
    else:
        print('Ulangi bagian terkait di lab, coba lagi besok.')


jalankan_kuis()""")

md("""---

Sudah mencoba serius? Bandingkan pendekatanmu dengan
`03_kunci_jawaban_kuis_rag.ipynb` — fokus pada **kenapa**,
bukan sekadar jawaban benar/salah.""")

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent / "02_kuis_rag.ipynb"
out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"OK menulis {out} — {len(CELLS)} cells")
