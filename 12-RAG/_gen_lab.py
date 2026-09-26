"""Generate 01_lab_rag.ipynb untuk Bab 12.

Struktur cell mengikuti pola Bab 7-11 (agar _verify_lab.py bisa memverifikasi):
  - cell GIVEN  : setup data / embedding palsu / generator yang sudah jadi
  - cell TODO   : HANYA fungsi/kelas yang diisi mahasiswa (di-skip verifier, di-inject ref)
  - cell CEK    : assert murni (tidak mendefinisikan fungsi)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_lab_rag.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = []

# ============================ HEADER ============================
cells.append(md(
    "# 🧪 Lab 12 — RAG: Retrieval Augmented Generation dari Nol",
    "",
    "> Pendamping materi Bab 12. Kamu membangun pipeline RAG **tanpa framework**:",
    "> chunk → embed → index → retrieve → prompt → generate → abstain → evaluasi.",
    "> Setiap langkah diukur; tidak ada \"rasanya oke\".",
    "",
    "Embedding di lab ini **palsu dan deterministik** (`project-ragkit/ragkit/embed.py`):",
    "hashing n-gram yang membuat teks yang berbagi kata berbagi vektor. Generator-nya",
    "ekstraktif — hanya mengutip kalimat konteks, tidak bisa mengarang. Dua pilihan",
    "itu membuat semua angka terkunci: tanpa jaringan, tanpa API key, tanpa biaya.",
    "",
    "| Bagian | Topik | Waktu (menit) |",
    "|---|---|---|",
    "| 0 | Setup: korpus, chunking, embedding palsu | 10 |",
    "| 1 | Chunking: ukuran & overlap | 15 |",
    "| 2 | Embedding & cosine similarity | 15 |",
    "| 3 | Vector store & retrieval | 20 |",
    "| 4 | Evaluasi retrieval: hit@k | 20 |",
    "| 5 | Prompt RAG + generator + abstain gate | 25 |",
    "| 6 | Hybrid search (BM25 + RRF) & reranker | 25 |",
    "| 7 | Pipeline end-to-end + biaya | 20 |",
    "| 8 | Ringkasan & pertanyaan analisis | 10 |",
    "",
    "Setiap latihan punya cell **✅ Cek** — jalankan; kalau ✅ berarti pemahamanmu benar.",
    "Semua angka terkunci SEED 12.",
))

# ============================ BAGIAN 0 ============================
cells.append(md(
    "## Bagian 0 — Setup: Korpus & Mesin Deterministik",
    "",
    "`project-ragkit/ragkit/` menyediakan (GIVEN):",
    "",
    "- `data.DOKUMEN` — 3 dokumen kebijakan HR (6 halaman);",
    "- `embed.EncoderEmbedding` — hashing n-gram → vektor ternormalisasi L2;",
    "- `chunking.buat_semua_chunks` — potong per kalimat, ukuran 320 / overlap 64;",
    "- `generator.GeneratorEkstraktif` — jawab HANYA dari konteks, punya abstain.",
))
cells.append(code(
    "import math",
    "import sys",
    "from pathlib import Path",
    "",
    "# Cari folder project-ragkit (GIVEN layer ada di sana)",
    "_akar = Path.cwd()",
    "while _akar != _akar.parent and not (_akar / 'project-ragkit' / 'ragkit').is_dir():",
    "    _akar = _akar.parent",
    "sys.path.insert(0, str(_akar / 'project-ragkit'))",
    "",
    "from ragkit.data import (DOKUMEN, KUERI_UJI, KUERI_LUAR_CORPUS, TOLOK_UKUR,",
    "                         SEP_DETEKSI_TIDAK_ADA)",
    "from ragkit.chunking import (buat_semua_chunks, pecah_kalimat, UKURAN_CHUNK,",
    "                             OVERLAP_CHUNK)",
    "from ragkit.embed import EncoderEmbedding, cosine, tokenisasi",
    "from ragkit.generator import GeneratorEkstraktif, jawab, prompt_rag",
    "",
    "print('GIVEN layer dari:', _akar / 'project-ragkit')",
    "print('ukuran chunk   :', UKURAN_CHUNK, 'karakter | overlap:', OVERLAP_CHUNK)",
    "print('dokumen        :', [d['source'] for d in DOKUMEN])",
    "print('kueri emas     :', [q['id'] for q in KUERI_UJI])",
    "print('luar korpus    :', KUERI_LUAR_CORPUS['teks'])",
))
cells.append(code(
    "chunks = buat_semua_chunks()",
    "print('total chunk:', len(chunks))",
    "for c in chunks:",
    "    print('[%d] %-32s p%d len=%d' % (c['chunk_index'], c['source'], c['page'], len(c['text'])))",
    "",
    "print()",
    "print('contoh chunk [0]:')",
    "print(chunks[0]['text'][:220])",
))

# ============================ BAGIAN 1 ============================
cells.append(md(
    "## Bagian 1 — Chunking: Ukuran & Overlap",
    "",
    "Chunking = memotong dokumen agar muat di konteks LLM **tanpa merusak makna**.",
    "Aturan yang kamu isi: kumpulkan kalimat sampai melewati `ukuran`, simpan chunk,",
    "lalu mulai chunk berikutnya dari **kalimat ekor** yang masih muat di `overlap`.",
    "",
    "Trade-off: chunk kecil = fokus tapi kehilangan konteks; chunk besar = utuh tapi",
    "bising. Di sini ukurannya karakter (320/64); produksi memakai token (300–800).",
))
cells.append(code(
    "def potong_teks(teks, ukuran=320, overlap=64):",
    "    \"\"\"Potong teks per kalimat: maksimal `ukuran` karakter, overlap `overlap`.",
    "",
    "    1. pecah jadi kalimat (pecah_kalimat dari GIVEN);",
    "    2. kumpulkan sampai menambahi melewati `ukuran` → simpan chunk;",
    "    3. mulai chunk berikutnya dari kalimat ekor (total <= `overlap`);",
    "    4. sisa jadi chunk terakhir. Return list[str].",
    "    \"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 1.1 — hasil identik dengan referensi GIVEN",
    "from ragkit.chunking import potong_teks as potong_ref",
    "",
    "teks_doc = DOKUMEN[0]['halaman'][0]",
    "punya = potong_teks(teks_doc)",
    "ref = potong_ref(teks_doc)",
    "print('jumlah chunk:', len(punya))",
    "for i, p in enumerate(punya):",
    "    print('  [%d] len=%d | %s' % (i, len(p), p[:60]))",
    "assert punya == ref, 'hasil harus identik dengan referensi'",
    "print('✅ Latihan 1.1 benar! Chunkingmu deterministik & sama dengan pipeline.')",
))
cells.append(code(
    "# ✅ Cek latihan 1.2 — overlap & teks utuh",
    "teks_panjang = (' '.join(['Karyawan tetap mendapatkan cuti tahunan dua belas hari.'] * 12)",
    "                + ' Sisa cuti tidak dibawa ke tahun berikutnya dan hangus.')",
    "potongan = potong_teks(teks_panjang, ukuran=120, overlap=40)",
    "print('jumlah chunk :', len(potongan))",
    "for i, p in enumerate(potongan):",
    "    print('  [%d] len=%d | mulai: %s' % (i, len(p), p[:50]))",
    "assert all(len(p) <= 120 for p in potongan), 'ada chunk yang melompati batas'",
    "assert len(potongan) >= 3, 'teks panjang harus terpotong beberapa kali'",
    "gabung = ' '.join(potongan)",
    "assert 'hangus' in gabung, 'isi teks tidak boleh hilang'",
    "print('✅ Latihan 1.2 benar! Semua chunk di bawah batas, isi teks tetap utuh.')",
))

# ============================ BAGIAN 2 ============================
cells.append(md(
    "## Bagian 2 — Embedding & Cosine Similarity",
    "",
    "Encoder GIVEN memetakan teks → vektor 256 dim **ternormalisasi L2**. Konsekuensi:",
    "cosine similarity = dot product. Dua teks yang berbagi kata punya skor tinggi.",
    "",
    "Kamu mengisi `cosine_similarity` (dengan pembagi norm — jangan asumsikan",
    "ternormalisasi) dan mengeksplorasi matriks kemiripan.",
))
cells.append(code(
    "def cosine_similarity(a, b):",
    "    \"\"\"Cosine similarity dua vektor (list). Vektor nol → 0.0 (jangan ZeroDivision).\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 2.1 — sifat cosine",
    "assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0",
    "assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0",
    "assert abs(cosine_similarity([2.0, 0.0], [5.0, 0.0]) - 1.0) < 1e-9   # arah sama",
    "assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0              # nol aman",
    "print('✅ Latihan 2.1 benar! Cosine mengukur ARAH, bukan panjang vektor.')",
))
cells.append(code(
    "# ✅ Cek latihan 2.2 — semantik encoder GIVEN",
    "enc = EncoderEmbedding()",
    "v_cuti = enc.embed_query('Berapa hari cuti tahunan karyawan tetap?')",
    "v_sama = enc.embed_query('cuti tahunan karyawan tetap berapa hari')",
    "v_beda = enc.embed_query('gaji dibayarkan setiap tanggal 25 via transfer bank')",
    "print('cos(kueri, parafrase) : %.3f' % cosine_similarity(v_cuti, v_sama))",
    "print('cos(kueri, topik lain): %.3f' % cosine_similarity(v_cuti, v_beda))",
    "assert cosine_similarity(v_cuti, v_sama) > cosine_similarity(v_cuti, v_beda)",
    "assert cosine_similarity(v_cuti, v_sama) > 0.5",
    "assert abs(sum(x * x for x in v_cuti) - 1.0) < 1e-6   # ternormalisasi L2",
    "print('✅ Latihan 2.2 benar! Teks berbagi kata → vektor mirip; L2-norm = 1.')",
    "print('   Keterbatasan: sinonim tanpa kata sama tetap rendah — itu kerjaan')",
    "print('   embedding model nyata + hybrid search (Bagian 6).')",
))

# ============================ BAGIAN 3 ============================
cells.append(md(
    "## Bagian 3 — Vector Store & Retrieval",
    "",
    "Vector store = daftar vektor + pencarian cosine brute-force. Kontrak:",
    "",
    "- hasil diurutkan skor TERTINGGI dulu, tie → **index kecil** (deterministik);",
    "- `top_k` > jumlah data → kembalikan semua; store kosong → `[]`.",
))
cells.append(code(
    "class VectorStoreLab:",
    "    \"\"\"Vector store in-memory (versi lab).\"\"\"",
    "",
    "    def __init__(self):",
    "        # TODO",
    "        raise NotImplementedError",
    "",
    "    def add(self, chunks, matriks):",
    "        # TODO",
    "        raise NotImplementedError",
    "",
    "    def cari(self, vektor_kueri, top_k=3):",
    "        \"\"\"Return list {'chunk', 'score', 'index'} urut skor turun.\"\"\"",
    "        # TODO",
    "        raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 3.1 — retrieval bekerja",
    "store = VectorStoreLab()",
    "store.add(chunks, enc.embed([c['text'] for c in chunks]))",
    "",
    "Q = KUERI_UJI[0]['teks']",
    "dapat = store.cari(enc.embed_query(Q), top_k=3)",
    "print('kueri:', Q)",
    "for r in dapat:",
    "    print('[%d] %.4f %s p%d' % (r['chunk']['chunk_index'], r['score'],",
    "                                r['chunk']['source'], r['chunk']['page']))",
    "assert [r['chunk']['chunk_index'] for r in dapat] == [5, 0, 4]",
    "assert abs(dapat[0]['score'] - 0.587) < 1e-3",
    "print('✅ Latihan 3.1 benar! (angka terkunci — deterministik)')",
))
cells.append(code(
    "# ✅ Cek latihan 3.2 — kontrak kasus tepi",
    "kosong = VectorStoreLab()",
    "assert kosong.cari([1.0, 2.0], top_k=3) == []",
    "",
    "s2 = VectorStoreLab()",
    "v = [1.0, 0.0, 0.0]",
    "s2.add([{'chunk_index': 0, 'text': 'a', 'source': 's', 'page': 1, 'doc_id': 'd'},",
    "        {'chunk_index': 1, 'text': 'b', 'source': 's', 'page': 1, 'doc_id': 'd'}],",
    "       [v, v])",
    "tie = s2.cari([1.0, 0.0, 0.0], top_k=2)",
    "assert [r['index'] for r in tie] == [0, 1], 'tie harus dipecah index kecil'",
    "banyak = store.cari(enc.embed_query(Q), top_k=99)",
    "assert len(banyak) == len(chunks), 'top_k melebihi data → kembalikan semua'",
    "print('✅ Latihan 3.2 benar! Determinisme itu fitur: tie yang berubah-ubah =')",
    "print('   hasil evaluasi yang tidak bisa dipercaya.')",
))

# ============================ BAGIAN 4 ============================
cells.append(md(
    "## Bagian 4 — Evaluasi Retrieval: hit@k",
    "",
    "Sebelum bicara kualitas JAWABAN, ukur dulu apakah dokumen yang benar ditemukan.",
    "hit@k: kueri kena bila `source_emas` ada di antara sumber top-k.",
    "",
    "Kontrak `evaluasi_retrieval`: return `{'per_kueri': [{'id','hit','posisi'}...],",
    "'hit_rate': float, 'kueri_gagal': [id]}` — posisi 1-based, None bila miss.",
))
cells.append(code(
    "def evaluasi_retrieval(retrieve, kueri_uji=None, top_k=3):",
    "    \"\"\"hit@k per kueri + agregat. retrieve(kueri, top_k) → list {'chunk', ...}.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "",
    "def buat_retriever_lab(store=None, encoder=None):",
    "    \"\"\"Index sekali → fungsi retrieve(kueri, top_k=3).\"\"\"",
    "    # TODO (hint: gunakan store & encoder yang sudah ada di variabel global",
    "    #  bila parameter None — atau buat sendiri dari buat_semua_chunks)",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 4.1 — angka terkunci hit@k",
    "retrieve = buat_retriever_lab()",
    "for k in (1, 2, 3):",
    "    ev = evaluasi_retrieval(retrieve, top_k=k)",
    "    print('hit@%d = %.2f | gagal: %s' % (k, ev['hit_rate'], ev['kueri_gagal']))",
    "ev1 = evaluasi_retrieval(retrieve, top_k=1)",
    "ev3 = evaluasi_retrieval(retrieve, top_k=3)",
    "assert abs(ev1['hit_rate'] - 0.8) < 1e-9",
    "assert ev1['kueri_gagal'] == ['q1']",
    "assert abs(ev3['hit_rate'] - 1.0) < 1e-9",
    "by_id = {p['id']: p for p in ev3['per_kueri']}",
    "assert by_id['q1']['posisi'] == 2 and by_id['q2']['posisi'] == 1",
    "print('✅ Latihan 4.1 benar! Angka terkunci: hit@1 0.8, hit@2/3 1.0.')",
))
cells.append(code(
    "# ✅ Cek latihan 4.2 — kenapa q1 gagal di top-1?",
    "dapat_q1 = retrieve('Berapa hari cuti tahunan untuk karyawan tetap?', 3)",
    "print('kueri :', KUERI_UJI[0]['teks'])",
    "print('emas  :', KUERI_UJI[0]['source_emas'])",
    "for r in dapat_q1:",
    "    tanda = '← EMAS' if r['chunk']['source'] == KUERI_UJI[0]['source_emas'] else ''",
    "    print('[%d] %.4f %-32s | %s %s' % (r['chunk']['chunk_index'], r['score'],",
    "                                       r['chunk']['source'],",
    "                                       r['chunk']['text'][:42], tanda))",
    "print()",
    "print('FAQ menang karena kueri & FAQ berbagi KATA PERSIS (mengajukan, cuti, tahunan).')",
    "print('Encoder hashing paham kecocokan leksikal, bukan status dokumen — dan itu')",
    "print('WAJAR: info yang sama memang ada di dua dokumen. Misi jangka panjang:')",
    "print('kurangi duplikasi konten, atau tambahkan bobot metadata (Bagian 6).')",
))

# ============================ BAGIAN 5 ============================
cells.append(md(
    "## Bagian 5 — Prompt RAG, Generator & Abstain Gate",
    "",
    "Prompt RAG terkunci (GIVEN) punya 4 aturan: hanya dari KONTEKS, jawab persis",
    "`tidak ada di dokumen` bila tidak ada, sebut sumber [n], jangan mengarang angka.",
    "",
    "Generator GIVEN ekstraktif: hanya mengutip kalimat konteks. Lalu kamu mengisi",
    "**abstain gate** — timpa jawaban bila skor retrieval terbaik di bawah ambang.",
))
cells.append(code(
    "# ✅ Cek latihan 5.1 — anatomi prompt RAG",
    "prompt, meta = prompt_rag('Berapa hari cuti tahunan untuk karyawan tetap?',",
    "                          [r['chunk'] for r in dapat_q1])",
    "print(prompt[:420])",
    "print('...')",
    "print('metadata sitasi:', meta)",
    "assert 'KONTEKS' in prompt and 'tidak ada di dokumen' in prompt",
    "assert '[1]' in prompt and '[2]' in prompt",
    "assert meta[0]['source'] == 'faq-karyawan.pdf'",
    "print('✅ Latihan 5.1 benar! Prompt = aturan + konteks bernomor + pertanyaan.')",
))
cells.append(code(
    "# ✅ Cek latihan 5.2 — generator ekstraktif",
    "gen = GeneratorEkstraktif()",
    "chunks_q1 = [r['chunk'] for r in dapat_q1]",
    "h = jawab(gen, 'Berapa hari cuti tahunan untuk karyawan tetap?', chunks_q1)",
    "print('jawaban :', h['jawaban'])",
    "print('sitasi  :', h['sitasi'], '| sumber:', h['sumber'])",
    "assert '12' in h['jawaban']",
    "assert h['sumber'][0]['page'] == 1",
    "",
    "h6 = jawab(gen, KUERI_LUAR_CORPUS['teks'], chunks_q1)",
    "print('q6 (dengan konteks cuti):', h6['jawaban'][:60])",
    "print('✅ Latihan 5.2 benar! Generator HANYA mengutip — model nyata bisa mengarang,')",
    "print('   karena itu prompt ber-aturan tetap wajib.')",
))
cells.append(code(
    "def abstaikan_lab(hasil, ambang=0.30):",
    "    \"\"\"Timpa jawaban jadi abstain bila skor terbaik < ambang.",
    "",
    "    - return dict BARU (input tidak diubah);",
    "    - abstain yang sudah ada → dibiarkan;",
    "    - bila menimpa: kosongkan jawaban/sitasi/sumber, set abstain=True,",
    "      tambah kunci 'abstaikan_oleh' = round(skor, 4).",
    "    \"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 5.3 — abstain gate",
    "skor_q6 = retrieve(KUERI_LUAR_CORPUS['teks'], 1)[0]['score']",
    "print('skor terbaik q6:', round(skor_q6, 4), '(terkunci 0.0652)')",
    "",
    "h = {'jawaban': '12 hari', 'sitasi': [0], 'sumber': [{'n': 1}],",
    "     'abstain': False, 'skor_terbaik': skor_q6}",
    "out = abstaikan_lab(h, ambang=0.30)",
    "print('hasil gate:', out['jawaban'], '| oleh:', out.get('abstaikan_oleh'))",
    "assert out['abstain'] and out['jawaban'] == SEP_DETEKSI_TIDAK_ADA",
    "assert out['sitasi'] == [] and abs(out['abstaikan_oleh'] - 0.0652) < 1e-3",
    "assert h['jawaban'] == '12 hari'                       # input tak diubah",
    "",
    "h2 = dict(h, skor_terbaik=0.59)",
    "out2 = abstaikan_lab(h2)",
    "assert out2['jawaban'] == '12 hari' and 'abstaikan_oleh' not in out2",
    "",
    "h3 = dict(h, abstain=True, jawaban=SEP_DETEKSI_TIDAK_ADA)",
    "out3 = abstaikan_lab(h3)",
    "assert 'abstaikan_oleh' not in out3                   # abstain dibiarkan",
    "print('✅ Latihan 5.3 benar! Gate = jaring pengaman kedua di sisi retrieval.')",
))

# ============================ BAGIAN 6 ============================
cells.append(md(
    "## Bagian 6 — Hybrid Search (BM25 + RRF) & Reranker",
    "",
    "Vector search menangkap makna; BM25 menangkap kata kunci eksak. Reciprocal Rank",
    "Fusion menggabungkan **peringkat** (bukan skor mentah): `skor = Σ 1/(60+rank+1)`",
    "dengan rank 0-based. Reranker lalu mengurutkan ulang kandidat dengan skor murah",
    "(angka ×3) dan menyimpan hanya top-k terbaik untuk konteks LLM.",
))
cells.append(code(
    "def cari_bm25_lab(kueri, chunks, top_k=3):",
    "    \"\"\"BM25 mini: k1=1.2, b=0.75, idf = ln(1 + (N - df + 0.5)/(df + 0.5)).",
    "    Return list {'chunk', 'score', 'index'} — skor BM25 mentah.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "",
    "def gabung_rrf_lab(daftar_dapat, k=60):",
    "    \"\"\"skor[chunk_index] = Σ 1/(k + peringkat + 1); peringkat 0-based.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "",
    "def cari_hybrid_lab(retrieve_vektor, chunks, kueri, top_k=3, rrf_k=60):",
    "    \"\"\"Vektor top-k + BM25 top-k → RRF → top_k dict {'chunk','score','index'}.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 6.1 — BM25 menyukai kecocokan eksak",
    "d_bm25 = cari_bm25_lab('surat dokter cuti sakit', chunks, top_k=3)",
    "print('BM25 top-3:', [(d['chunk']['chunk_index'], round(d['score'], 2)) for d in d_bm25])",
    "assert d_bm25[0]['chunk']['source'] == 'kebijakan-cuti-kesehatan.pdf'",
    "",
    "d_vek = retrieve('surat dokter cuti sakit', 3)",
    "print('vektor    :', [(d['chunk']['chunk_index'], round(d['score'], 2)) for d in d_vek])",
    "print('✅ Latihan 6.1 benar! Untuk kata kunci spesifik, BM25 tajam.')",
))
cells.append(code(
    "# ✅ Cek latihan 6.2 — RRF & hybrid",
    "assert gabung_rrf_lab([[{'chunk': {'chunk_index': 1}}]], k=10)[1] == 1 / 11",
    "",
    "h3 = cari_hybrid_lab(retrieve, chunks, KUERI_UJI[0]['teks'], top_k=3)",
    "print('hybrid q1 top-3:', [d['chunk']['chunk_index'] for d in h3])",
    "assert [d['chunk']['chunk_index'] for d in h3] == [5, 0, 4]",
    "",
    "hit = sum(1 for q in KUERI_UJI",
    "          if q['source_emas'] in [d['chunk']['source'] for d in",
    "                                  cari_hybrid_lab(retrieve, chunks, q['teks'], top_k=3)])",
    "print('hit@3 hybrid:', hit, 'dari', len(KUERI_UJI))",
    "assert hit == 5",
    "print('✅ Latihan 6.2 benar! Hybrid = makna + kata kunci, stabil di kedua mode.')",
))
cells.append(code(
    "def rerank_lab(kueri, hasil, top_k=2, skorer=None):",
    "    \"\"\"Urutkan ulang dengan skorer murah (angka ×3, kata lain ×1, stopword dibuang),",
    "    tie → chunk_index kecil; ambil top_k. skorer None → skorer default.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
    "",
    "",
    "def skorer_kata_lab(kueri, teks):",
    "    \"\"\"Σ bobot kata kueri yang muncul di teks: angka 3.0, kata lain 1.0.\"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 6.3 — reranker memangkas konteks",
    "Q2 = KUERI_UJI[1]['teks']",
    "dapat2 = retrieve(Q2, 3)",
    "hasil_rr = rerank_lab(Q2, dapat2, top_k=2)",
    "print('kueri           :', Q2)",
    "print('urutan retrieval:', [d['chunk']['chunk_index'] for d in dapat2])",
    "print('sesudah rerank  :', [d['chunk']['chunk_index'] for d in hasil_rr])",
    "assert [d['chunk']['chunk_index'] for d in hasil_rr] == [3, 4]",
    "assert skorer_kata_lab('cuti 12 hari', '12 hari cuti tahunan') == 4.0",
    "print('✅ Latihan 6.3 benar! Konteks 3 → 2 chunk: LLM fokus, token turun.')",
    "print('   Produksi: cross-encoder (bge-reranker) menggantikan skor kata —')",
    "print('   polanya sama: kandidat sedikit, skor mahal per kandidat.')",
))

# ============================ BAGIAN 7 ============================
cells.append(md(
    "## Bagian 7 — Pipeline End-to-End + Biaya",
    "",
    "Gabungan semua: retrieve → prompt → generate → gate → biaya + latensi per tahap.",
    "Latensi pipeline DETERMINISTIK: `retrieve_ms = 8 + 2×jumlah_chunk + 25`,",
    "`generate_ms = 2×25`. Biaya memakai tarif mini (USD/1 juta token): input 0.15,",
    "output 0.60 — angka ILUSTRATIF, cara hitungnya yang dipelajari.",
))
cells.append(code(
    "def jalankan_rag_lab(retrieve, kueri, generator=None, top_k=3, ambang=0.30,",
    "                     generator_latency_ms=25):",
    "    \"\"\"Satu permintaan RAG end-to-end.",
    "",
    "    Return dict: kueri, chunks, jawaban, sitasi, sumber, abstain,",
    "    skor_terbaik, skema (prompt), biaya_usd, tahap {'retrieve_ms','generate_ms'},",
    "    + 'abstaikan_oleh' HANYA bila gate yang menimpa.",
    "    token = ceil(len(teks)/4); biaya = token_in/1e6*0.15 + token_out/1e6*0.60.",
    "    \"\"\"",
    "    # TODO",
    "    raise NotImplementedError",
))
cells.append(code(
    "# ✅ Cek latihan 7.1 — pipeline + angka terkunci",
    "h1 = jalankan_rag_lab(retrieve, KUERI_UJI[0]['teks'])",
    "print('jawaban :', h1['jawaban'][:80])",
    "print('sumber  :', h1['sumber'])",
    "print('biaya   : $%.8f' % h1['biaya_usd'])",
    "print('tahap   :', h1['tahap'])",
    "assert '12' in h1['jawaban'] and not h1['abstain']",
    "assert abs(h1['biaya_usd'] - 5.91e-05) < 1e-9",
    "assert h1['tahap'] == {'retrieve_ms': 47, 'generate_ms': 50}",
    "print('✅ Latihan 7.1 benar! Biaya & latensi per tahap terlapor — bukan \"jalan kok\".')",
))
cells.append(code(
    "# ✅ Cek latihan 7.2 — dua lapis pertahanan untuk kueri luar korpus",
    "h6 = jalankan_rag_lab(retrieve, KUERI_LUAR_CORPUS['teks'])",
    "print('q6:', h6['jawaban'], '| skor', round(h6['skor_terbaik'], 4),",
    "      '| abstaikan_oleh:', h6.get('abstaikan_oleh'))",
    "assert h6['abstain'] and h6['jawaban'] == SEP_DETEKSI_TIDAK_ADA",
    "# Generator abstain LEBIH DULU (skor kalimat < SKOR_MINIMUM) → gate HORMATI:",
    "# tidak menimpa abstain yang sudah ada — kontrak abstaikan (Latihan 5.3).",
    "assert 'abstaikan_oleh' not in h6",
    "",
    "# Lapis 1 sendirian: ambang dimatikan (ambang=-1.0) → generator tetap menahan",
    "h6b = jalankan_rag_lab(retrieve, KUERI_LUAR_CORPUS['teks'], ambang=-1.0)",
    "print('q6 tanpa gate:', h6b['jawaban'], '| abstain:', h6b['abstain'])",
    "assert h6b['abstain'] and h6b['jawaban'] == SEP_DETEKSI_TIDAK_ADA",
    "# Lapis 2 didemokan di Latihan 7.3: skor rendah → gate yang menimpa.",
    "print('✅ Latihan 7.2 benar! Dua lapis: generator (dulu) + gate (yang hormati).')",
))
cells.append(code(
    "# ✅ Cek latihan 7.3 — kegagalan retrieval tertutup gate",
    "asli = retrieve",
    "dapat_top1 = retrieve(KUERI_UJI[0]['teks'], 1)",
    "def retrieve_skor_rendah(kueri, top_k=3):",
    "    return [{'chunk': dapat_top1[0]['chunk'], 'score': 0.2, 'index': 0}]",
    "",
    "hs = jalankan_rag_lab(retrieve_skor_rendah, KUERI_UJI[0]['teks'])",
    "print('skor 0.2 →', hs['jawaban'], '| oleh:', hs.get('abstaikan_oleh'))",
    "assert hs['abstain'] and abs(hs['abstaikan_oleh'] - 0.2) < 1e-4",
    "print('✅ Latihan 7.3 benar! Konteks \"relevan\" + skor rendah = tetap abstain.')",
    "print('   Skor retrieval adalah suara sistem tentang keyakinannya — dengarkan.')",
))

# ============================ BAGIAN 8 ============================
cells.append(md(
    "## Bagian 8 — Ringkasan: Satu Pipeline, Lima Angka",
    "",
    "Angka dari bagian sebelumnya disatukan; inilah yang dilaporkan saat sistem ini",
    "berubah — bentuk paling sederhana dari evaluasi (Bab 15).",
))
cells.append(code(
    "# ✅ Cek latihan 8.1 — ringkasan terkunci",
    "print('=' * 64)",
    "print('RINGKASAN: pipeline RAG tanpa framework')",
    "print('=' * 64)",
    "print('1. chunking   : %d chunk (ukuran %d / overlap %d karakter)'",
    "      % (len(chunks), UKURAN_CHUNK, OVERLAP_CHUNK))",
    "print('2. retrieval  : hit@1 %.1f -> hit@2 %.1f (q1 gagal top-1)'",
    "      % (evaluasi_retrieval(retrieve, top_k=1)['hit_rate'],",
    "         evaluasi_retrieval(retrieve, top_k=2)['hit_rate']))",
    "print('3. hybrid     : hit@3 = 1.0 (vektor + BM25 + RRF)')",
    "print('4. pipeline   : $%.8f per permintaan | %d ms'",
    "      % (h1['biaya_usd'], h1['tahap']['retrieve_ms'] + h1['tahap']['generate_ms']))",
    "print('5. abstain    : q6 dijawab %r (dua lapis; skor %.4f < 0.30)'",
    "      % (h6['jawaban'], h6['skor_terbaik']))",
    "print()",
    "print('Semua angka terkunci karena embedding & generator palsu-deterministik.')",
    "print('Itulah inti bab ini: RAG yang serius diukur di titik kegagalannya —')",
    "print('retrieval miss, konteks kosong, pertanyaan di luar korpus.')",
))
cells.append(md(
    "### Koneksi ke bab lain",
    "",
    "- **Bab 11:** kunci cache wajib memuat konteks retrieval; anggaran & fallback",
    "  melindungi panggilan generator.",
    "- **Bab 13 (Fine-tuning):** embedding model sendiri = tuning retrieval ke",
    "  bahasa/domainmu; biaya per token turun, uptime jadi tanggung jawabmu.",
    "- **Bab 14 (Agent):** `cari_dokumen` adalah tool paling umum di tangan agent.",
    "- **Bab 15 (Evaluasi):** hit@k di sini = retrieval metrics; faithfulness &",
    "  answer relevancy (RAGAS) dilanjutkan di sana.",
    "- **Bab 16 (Deployment):** index yang di-refresh berkala + cache + kuota.",
    "",
    "**Pertanyaan analisis** (jawab di sini, benar-benar dijawab):",
    "",
    "1. hit@1 = 0.8 tapi hit@2 = 1.0 — kueri mana yang gagal, kenapa FAQ menang,",
    "   dan apa konsekuensinya untuk pilihan `top_k`?",
    "2. Ambang abstain 0.30: apa yang terjadi kalau terlalu rendah? Terlalu tinggi?",
    "   Ulangi Bagian 5 dengan 0.15 dan 0.45, lihat jumlah abstain vs jawaban salah.",
    "3. Generator ekstraktif tidak bisa mengarang. Lapisan mana yang tetap wajib",
    "   saat diganti LLM nyata? (prompt ber-aturan? gate? evaluasi?)",
    "4. Hybrid menambah pencarian kedua tiap permintaan. Kapan layak, kapan vector",
    "   saja cukup?",
    "5. Rerank memangkas konteks 3 → 2 chunk. Informasi apa yang bisa hilang?",
    "6. Kalau dokumen kebijakan berubah tiap bulan, bagian mana dari pipeline yang",
    "   harus jalan ulang, dan bagaimana cache retrieval ikut basi?",
    "",
    "Lanjut ke: **kuis** (`02_kuis_rag.ipynb`) lalu **project starter**",
    "(`project-ragkit/`) — paket `ragkit` end-to-end dengan **66 test**.",
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
