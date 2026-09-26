"""Generate starter.ipynb & solusi.ipynb untuk project Bab 12 (RAG)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines]}


def tulis(nama, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = BASE / nama
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out, f"({len(cells)} cells)")


SETUP = [
    "import sys",
    "from pathlib import Path",
    "",
    "# Tambahkan root project ke path (ragkit ada di ./ragkit)",
    "_akar = Path.cwd()",
    "while _akar != _akar.parent and not (_akar / 'ragkit' / 'data.py').is_file():",
    "    _akar = _akar.parent",
    "sys.path.insert(0, str(_akar))",
    "",
    "from ragkit.data import DOKUMEN, KUERI_UJI, KUERI_LUAR_CORPUS, TOLOK_UKUR",
    "from ragkit.chunking import buat_semua_chunks, UKURAN_CHUNK, OVERLAP_CHUNK",
    "from ragkit.embed import EncoderEmbedding, cosine",
    "from ragkit.generator import GeneratorEkstraktif, jawab, prompt_rag",
    "",
    "print('SEED korpus terkunci. Ukuran chunk:', UKURAN_CHUNK, 'karakter | overlap:', OVERLAP_CHUNK)",
    "print('dokumen:', [d['source'] for d in DOKUMEN])",
    "print('kueri uji:', [q['id'] for q in KUERI_UJI], '+ 1 kueri luar korpus')",
]

STARTER_HEADER = [
    "# 🏗️ Starter — project-ragkit (Bab 12: RAG)",
    "",
    "> Notebook eksperimen. Isi semua cell **TODO** (kontrak lengkap ada di `tests/`),",
    "> jalankan blok eksperimen, lalu jawab **6 Pertanyaan Analisis** di bagian akhir.",
    "",
    "Mode kerja yang disarankan (TDD):",
    "",
    "```bash",
    "python -m unittest discover -s tests -v",
    "```",
    "",
    "Urutan: `vectorstore.py` → `retriever.py` → `abstain.py` → `cache.py` →",
    "`hybrid.py` → `rerank.py` → `pipeline.py` → `evaluate.py` sampai **66 test hijau**.",
    "",
    "| Blok | Topik |",
    "|---|---|",
    "| 0 | Setup & anatomi korpus |",
    "| 1 | Vector store & cosine similarity |",
    "| 2 | Retriever + hit@k (angka terkunci) |",
    "| 3 | Abstain gate: kapan sistem bilang 'tidak tahu' |",
    "| 4 | Cache konteks retrieval |",
    "| 5 | Hybrid search (BM25 + RRF) |",
    "| 6 | Reranker |",
    "| 7 | Pipeline end-to-end + biaya |",
    "| 8 | Evaluasi ujung-ke-ujung |",
    "| 9 | Pertanyaan analisis |",
]

BLOK1_MD = [
    "## Blok 1 — Vector Store & Cosine Similarity",
    "",
    "`VectorStore` = daftar vektor + pencarian cosine brute-force. Dengan 7 chunk ini",
    "memang cukup; di produksi (≥100 ribu vektor) pencarian diganti ANN (HNSW/IVF) —",
    "konsepnya sama: **index sekali, cari cepat**.",
    "",
    "Kontrak penting: tie skor → index kecil dulu (deterministik); store kosong → `[]`.",
]

BLOK1_CODE = [
    "from ragkit.vectorstore import VectorStore",
    "",
    "chunks = buat_semua_chunks()",
    "enc = EncoderEmbedding()",
    "M = enc.embed([c['text'] for c in chunks])",
    "",
    "store = VectorStore()",
    "store.add(chunks, M)",
    "",
    "qv = enc.embed_query('Berapa hari cuti tahunan untuk karyawan tetap?')",
    "hasil = store.cari(qv, top_k=3)",
    "for r in hasil:",
    "    print('[%d] %.4f  %s p%d' % (r['chunk']['chunk_index'], r['score'],",
    "                                  r['chunk']['source'], r['chunk']['page']))",
    "    print('     ', r['chunk']['text'][:70].replace('\\n', ' '))",
    "",
    "# TODO: verifikasi — store kosong tidak crash, tie → index kecil",
    "assert VectorStore().cari([1.0, 2.0], top_k=3) == []",
    "print('OK: store kosong → []')",
]

BLOK2_MD = [
    "## Blok 2 — Retriever + Evaluasi hit@k",
    "",
    "hit@k: kueri \"kena\" bila dokumen sumber emas muncul di top-k hasil retrieval.",
    "Ini metrik retrieval paling murah sekaligus paling jujur — **sebelum** membahas",
    "kualitas jawaban, pastikan dulu dokumennya ditemukan.",
    "",
    "Angka terkunci (SEED 12): hit@1 = 0.8 (q1 gagal: FAQ menang top-1), hit@2 = 1.0.",
]

BLOK2_CODE = [
    "from ragkit.retriever import buat_retriever, evaluasi_retrieval",
    "",
    "retrieve = buat_retriever()",
    "",
    "for k in (1, 2, 3):",
    "    ev = evaluasi_retrieval(retrieve, top_k=k)",
    "    print('hit@%d = %.2f | gagal: %s' % (k, ev['hit_rate'], ev['kueri_gagal']))",
    "",
    "ev1 = evaluasi_retrieval(retrieve, top_k=1)",
    "assert abs(ev1['hit_rate'] - 0.8) < 1e-9",
    "assert ev1['kueri_gagal'] == ['q1']",
    "print('angka terkunci cocok: hit@1 = 0.8, hit@2 = 1.0')",
    "",
    "# Micro-eksperimen: kenapa q1 gagal di top-1?",
    "dapat_q1 = retrieve('Berapa hari cuti tahunan untuk karyawan tetap?', 3)",
    "for r in dapat_q1:",
    "    print('[%d] %.4f %s | %s' % (r['chunk']['chunk_index'], r['score'],",
    "                                  r['chunk']['source'], r['chunk']['text'][:50]))",
]

BLOK3_MD = [
    "## Blok 3 — Abstain Gate: Kapan Sistem Bilang \"Tidak Tahu\"",
    "",
    "Retrieval memberi **skor kemiripan**. Skor rendah = \"saya tidak yakin konteksnya",
    "relevan\" — dan jawaban terbaik saat itu adalah abstain, bukan mengarang.",
    "",
    "Kontrak: skor terbaik < ambang → jawaban diganti `tidak ada di dokumen`, sitasi",
    "dikosongkan, kunci `abstaikan_oleh` mencatat skor pembuktinya.",
]

BLOK3_CODE = [
    "from ragkit.abstain import abstaikan",
    "from ragkit.data import SEP_DETEKSI_TIDAK_ADA",
    "",
    "# Skor kueri luar korpus (terkunci): 0.0652 — jauh di bawah ambang 0.30",
    "retrieve_q6 = retrieve(KUERI_LUAR_CORPUS['teks'], 1)",
    "skor_q6 = retrieve_q6[0]['score']",
    "print('skor q6 (luar korpus):', round(skor_q6, 4))",
    "",
    "h = {'jawaban': '12 hari', 'sitasi': [0], 'sumber': [{'n': 1}],",
    "     'abstain': False, 'skor_terbaik': skor_q6}",
    "out = abstaikan(h, ambang=0.30)",
    "print('hasil gate:', out['jawaban'], '| oleh:', out.get('abstaikan_oleh'))",
    "assert out['abstain'] and out['jawaban'] == SEP_DETEKSI_TIDAK_ADA",
    "",
    "# Skor tinggi → utuh, tanpa kunci baru",
    "h2 = dict(h, skor_terbaik=0.59)",
    "out2 = abstaikan(h2, ambang=0.30)",
    "assert out2['jawaban'] == '12 hari' and 'abstaikan_oleh' not in out2",
    "print('OK: gate menimpa bila skor rendah, membiarkan bila aman')",
    "",
    "# Micro-eksperimen: coba ambang 0.2 / 0.4 — trade-off apa yang muncul?",
]

BLOK4_MD = [
    "## Blok 4 — Cache Konteks Retrieval",
    "",
    "Cache **konteks** (bukan jawaban LLM seperti Bab 11): kueri sama → konteks sama →",
    "pencarian + embedding ulang tidak perlu. Hit-rate diukur dengan pola permintaan",
    "berulang — `miss, hit, hit, miss` untuk 4 permintaan (2 pencarian).",
]

BLOK4_CODE = [
    "from ragkit.cache import CacheRetrieval, retrieve_bercache",
    "",
    "cache = CacheRetrieval(ttl=10)",
    "panggilan = {'n': 0}",
    "",
    "def retrieve_hitung(kueri, top_k=3):",
    "    panggilan['n'] += 1",
    "    return retrieve(kueri, top_k)",
    "",
    "pola = []",
    "for i, q in enumerate(TOLOK_UKUR['aturan']):",
    "    r = retrieve_bercache(retrieve_hitung, q, cache, sekarang=i, top_k=3)",
    "    pola.append(r['cache'])",
    "print('pola cache :', pola)",
    "print('pencarian  :', panggilan['n'], 'dari', len(TOLOK_UKUR['aturan']), 'permintaan')",
    "print('statistik  :', cache.statistik())",
    "assert pola == ['miss', 'hit', 'hit', 'miss']",
    "assert panggilan['n'] == 2",
    "print('OK: kueri sama (huruf besar-kecil berbeda) dianggap permintaan sama')",
    "",
    "# Micro-eksperimen: kunci cache memakai strip().lower() — kasus apa yang",
    "# masih dianggap 'beda' padahal semantiknya sama? (petunjuk: sinonim)",
]

BLOK5_MD = [
    "## Blok 5 — Hybrid Search (BM25 + RRF)",
    "",
    "Vector search menangkap makna; BM25 menangkap kata kunci eksak. Reciprocal Rank",
    "Fusion menggabungkan PERINGKAT (bukan skor mentah) dari keduanya:",
    "`skor = Σ 1/(60 + peringkat)`.",
    "",
    "Angka terkunci: hybrid q1 top-3 = chunk [5, 0, 4]; hit@3 hybrid = 1.0.",
]

BLOK5_CODE = [
    "from ragkit.hybrid import cari_bm25, gabung_rrf, cari_hybrid",
    "",
    "Q = KUERI_UJI[0]['teks']",
    "d_bm25 = cari_bm25(Q, chunks, top_k=3)",
    "print('BM25 top-3 :', [d['chunk']['chunk_index'] for d in d_bm25])",
    "",
    "h3 = cari_hybrid(retrieve, chunks, Q, top_k=3)",
    "print('hybrid top-3:', [d['chunk']['chunk_index'] for d in h3])",
    "assert [d['chunk']['chunk_index'] for d in h3] == [5, 0, 4]",
    "",
    "hit = sum(1 for q in KUERI_UJI",
    "          if q['source_emas'] in [d['chunk']['source']",
    "                                  for d in cari_hybrid(retrieve, chunks, q['teks'], top_k=3)])",
    "print('hit@3 hybrid:', hit, 'dari', len(KUERI_UJI))",
    "assert hit == 5",
    "print('OK: gabungan vektor + kata kunci paling stabil')",
]

BLOK6_MD = [
    "## Blok 6 — Reranker: Skor Mahal-per-Item untuk Sedikit Kandidat",
    "",
    "Pola two-stage: retrieval murah mengambil top-3, reranker mengurutkan ulang dan",
    "hanya menyimpan 2 terbaik untuk konteks LLM. Skor kata di sini menggantikan",
    "cross-encoder produksi (bge-reranker, Cohere Rerank) — polanya identik.",
]

BLOK6_CODE = [
    "from ragkit.rerank import rerank, skorer_kata",
    "",
    "Q2 = KUERI_UJI[1]['teks']",
    "dapat = retrieve(Q2, 3)",
    "print('urutan retrieval :', [d['chunk']['chunk_index'] for d in dapat])",
    "",
    "dapat_rerank = rerank(Q2, dapat, top_k=2)",
    "print('sesudah rerank   :', [d['chunk']['chunk_index'] for d in dapat_rerank])",
    "assert [d['chunk']['chunk_index'] for d in dapat_rerank] == [3, 4]",
    "",
    "# Lihat bobot skor kata: angka × 3",
    "print(\"skor 'Berapa lama maksimal cuti sakit tanpa surat dokter?' vs\")",
    "for c in (chunks[3], chunks[5]):",
    "    print('   [%d] %.1f | %s' % (c['chunk_index'], skorer_kata(Q2, c['text']),",
    "                                  c['text'][:55]))",
]

BLOK7_MD = [
    "## Blok 7 — Pipeline End-to-End + Biaya",
    "",
    "Satu permintaan RAG lengkap: retrieve → prompt → generate → gate → biaya +",
    "latensi per tahap. Latensi dibuat deterministik supaya bisa di-assert —",
    "pola yang sama dengan Bab 11 (provider palsu, angka terkunci).",
    "",
    "Angka terkunci q1: biaya $5.91e-05, retrieve_ms 47, generate_ms 50.",
]

BLOK7_CODE = [
    "from ragkit.pipeline import jalankan_rag",
    "",
    "h = jalankan_rag(retrieve, Q)",
    "print('jawaban   :', h['jawaban'])",
    "print('sumber    :', h['sumber'])",
    "print('skor      :', round(h['skor_terbaik'], 4))",
    "print('biaya     : $%.8f' % h['biaya_usd'])",
    "print('tahap     :', h['tahap'])",
    "assert abs(h['biaya_usd'] - 5.91e-05) < 1e-9",
    "assert h['tahap'] == {'retrieve_ms': 47, 'generate_ms': 50}",
    "print('OK: biaya & latensi per tahap terlapor — bukan \"jalan kok\"')",
    "",
    "# Kueri luar korpus lewat pipeline penuh",
    "h6 = jalankan_rag(retrieve, KUERI_LUAR_CORPUS['teks'])",
    "print('q6        :', h6['jawaban'], '| skor', round(h6['skor_terbaik'], 4))",
    "assert h6['abstain']",
    "print('OK: dua lapis pertahanan — generator + gate')",
]

BLOK8_MD = [
    "## Blok 8 — Evaluasi Ujung-ke-Ujung",
    "",
    "Akurasi = jawaban emas muncul di jawaban (untuk 5 kueri korpus) + kewajiban",
    "abstain untuk 1 kueri luar korpus. Inilah bentuk paling sederhana dari",
    "**faithfulness** yang dijabarkan di Bab 15.",
    "",
    "Angka terkunci: akurasi 1.0, abstain_benar True.",
]

BLOK8_CODE = [
    "from ragkit.evaluate import evaluasi_ujung_ke_ujung",
    "",
    "ev = evaluasi_ujung_ke_ujung(lambda q: jalankan_rag(retrieve, q))",
    "for p in ev['per_kueri']:",
    "    print('%s benar=%-5s abstain=%-5s %s' % (p['id'], p['benar'], p['abstain'], p['jawaban'][:40]))",
    "print('akurasi       :', ev['akurasi'])",
    "print('abstain_benar :', ev['abstain_benar'], '| jawaban luar:', ev['jawaban_luar'])",
    "assert abs(ev['akurasi'] - 1.0) < 1e-9 and ev['abstain_benar']",
    "print('OK: 5/5 kueri emas + 1 abstain untuk kueri di luar korpus')",
]

BLOK9_MD = [
    "## 🧠 6 Pertanyaan Analisis (bagian penilaian!)",
    "",
    "1. hit@1 = 0.8 tetapi hit@2 = 1.0: kueri mana yang gagal di top-1, dan mengapa",
    "   FAQ bisa \"mencuri\" posisi teratas? Konsekuensinya ke pilihan `top_k`?",
    "2. Ambang abstain 0.30 itu pilihan. Apa yang terjadi pada jawaban pengguna kalau",
    "   ambang terlalu rendah? Terlalu tinggi? Ulangi Blok 3 dengan 0.15 / 0.45.",
    "3. Cache retrieval di-key `strip().lower()`. Kueri \"cara ambil cuti\" vs",
    "   \"cara mengajukan cuti\" — kena cache atau tidak? Apa risiko kunci terlalu",
    "   agresif menyatukan kueri yang mirip?",
    "4. Hybrid menambah pencarian BM25 di setiap permintaan. Kapan tambahan latensi",
    "   ini layak, dan kapan vector search saja cukup?",
    "5. Rerank mengurangi konteks dari 3 chunk jadi 2. Informasi apa yang bisa hilang,",
    "   dan kapan itu lebih baik daripada memasukkan semua chunk ke LLM?",
    "6. Generator ekstraktif tidak bisa mengarang, LLM nyata bisa. Lapisan mana saja",
    "   di pipeline ini yang masih wajib ada saat generator diganti model nyata?",
    "   (petunjuk: skor retrieval, prompt ber-aturan, evaluasi — Bab 15)",
]

SOLUSI_HEADER = [
    "# ✅ Solusi — project-ragkit (Bab 12: RAG)",
    "",
    "> Notebook ini memuat versi jadi dari seluruh modul TODO (identik dengan",
    "> `solusi/ragkit_ref.py`), lalu seluruh blok eksperimen dijalankan.",
    "> Bandingkan dengan implementasimu — fokus pada **kenapa**, bukan hanya cocok.",
]

SOLUSI_IMPL = [
    "### Implementasi lengkap (referensi)",
    "",
    "Kontrak sama dengan `tests/`; angka terkunci berasal dari sini.",
]

REF_CODE = [
    "import sys",
    "from pathlib import Path",
    "",
    "_akar = Path.cwd()",
    "while _akar != _akar.parent and not (_akar / 'ragkit' / 'data.py').is_file():",
    "    _akar = _akar.parent",
    "sys.path.insert(0, str(_akar))",
    "sys.path.insert(0, str(_akar / 'solusi'))",
    "",
    "import ragkit_ref as ref",
    "from ragkit import abstain as _am, cache as _cm, evaluate as _em, hybrid as _hm, \\",
    "    pipeline as _pm, rerank as _rm, retriever as _rem, vectorstore as _vm",
    "",
    "_vm.VectorStore = ref.VectorStore",
    "_rem.buat_retriever = ref.buat_retriever",
    "_rem.evaluasi_retrieval = ref.evaluasi_retrieval",
    "_am.abstaikan = ref.abstaikan",
    "_cm.CacheRetrieval = ref.CacheRetrieval",
    "_cm.retrieve_bercache = ref.retrieve_bercache",
    "_cm.tolok_ukur = ref.tolok_ukur",
    "_hm.cari_bm25 = ref.cari_bm25",
    "_hm.gabung_rrf = ref.gabung_rrf",
    "_hm.cari_hybrid = ref.cari_hybrid",
    "_rm.skorer_kata = ref.skorer_kata",
    "_rm.rerank = ref.rerank",
    "_pm.jalankan_rag = ref.jalankan_rag",
    "_em.evaluasi_ujung_ke_ujung = ref.evaluasi_ujung_ke_ujung",
    "",
    "print('implementasi referensi dimuat (solusi/ragkit_ref.py)')",
]

# ============================== STARTER ==============================
starter = [md(*STARTER_HEADER)]
starter += [md(
    "## Blok 0 — Setup & Anatomi Korpus",
    "",
    "Korpus terkunci: 3 dokumen kebijakan HR (6 halaman) → 7 chunk. Lima kueri uji",
    "punya jawaban emas; satu kueri SENGAJA di luar korpus untuk menguji abstain.",
)]
starter += [code(*SETUP)]
starter += [code(
    "chunks = buat_semua_chunks()",
    "for c in chunks:",
    "    print('[%d] %-32s p%d len=%d' % (c['chunk_index'], c['source'], c['page'], len(c['text'])))",
    "print()",
    "print('chunk pertama:')",
    "print(chunks[0]['text'][:200])",
)]
starter += [md(*BLOK1_MD), code(*BLOK1_CODE)]
starter += [md(*BLOK2_MD), code(*BLOK2_CODE)]
starter += [md(*BLOK3_MD), code(*BLOK3_CODE)]
starter += [md(*BLOK4_MD), code(*BLOK4_CODE)]
starter += [md(*BLOK5_MD), code(*BLOK5_CODE)]
starter += [md(*BLOK6_MD), code(*BLOK6_CODE)]
starter += [md(*BLOK7_MD), code(*BLOK7_CODE)]
starter += [md(*BLOK8_MD), code(*BLOK8_CODE)]
starter += [md(*BLOK9_MD)]
starter += [md(
    "---",
    "",
    "Setelah semua test hijau + analisis terjawab: isi `RUBRIK.md`, baru bandingkan",
    "dengan `solusi.ipynb` / `solusi/ragkit_ref.py`.",
)]
tulis("starter.ipynb", starter)

# ============================== SOLUSI ==============================
solusi = [md(*SOLUSI_HEADER)]
solusi += [code(*SETUP)]
solusi += [md(*SOLUSI_IMPL), code(*REF_CODE)]

SOL_BLOK = [
    ("Blok 1 — Vector Store", BLOK1_MD, [
        line for line in BLOK1_CODE
        if not line.startswith("from ragkit.vectorstore")
    ]),
    ("Blok 2 — Retriever & hit@k", BLOK2_MD, [
        line for line in BLOK2_CODE
        if not line.startswith("from ragkit.retriever")
    ]),
    ("Blok 3 — Abstain Gate", BLOK3_MD, [
        line for line in BLOK3_CODE
        if not line.startswith("from ragkit.abstain")
    ]),
    ("Blok 4 — Cache Konteks", BLOK4_MD, [
        line for line in BLOK4_CODE
        if not line.startswith("from ragkit.cache")
    ]),
    ("Blok 5 — Hybrid Search", BLOK5_MD, [
        line for line in BLOK5_CODE
        if not line.startswith("from ragkit.hybrid")
    ]),
    ("Blok 6 — Reranker", BLOK6_MD, [
        line for line in BLOK6_CODE
        if not line.startswith("from ragkit.rerank")
    ]),
    ("Blok 7 — Pipeline", BLOK7_MD, [
        line for line in BLOK7_CODE
        if not line.startswith("from ragkit.pipeline")
    ]),
    ("Blok 8 — Evaluasi", BLOK8_MD, [
        line for line in BLOK8_CODE
        if not line.startswith("from ragkit.evaluate")
    ]),
]

# variabel global yang dibutuhkan blok solusi (chunks, retrieve, TOLOK_UKUR, KUERI_*)
solusi += [code(
    "from ragkit.chunking import buat_semua_chunks",
    "from ragkit.embed import EncoderEmbedding",
    "VectorStore = _vm.VectorStore       # versi referensi (sudah di-patch)",
    "buat_retriever = _rem.buat_retriever",
    "evaluasi_retrieval = _rem.evaluasi_retrieval",
    "abstaikan = _am.abstaikan",
    "CacheRetrieval = _cm.CacheRetrieval",
    "retrieve_bercache = _cm.retrieve_bercache",
    "cari_bm25, gabung_rrf, cari_hybrid = _hm.cari_bm25, _hm.gabung_rrf, _hm.cari_hybrid",
    "skorer_kata, rerank = _rm.skorer_kata, _rm.rerank",
    "jalankan_rag = _pm.jalankan_rag",
    "evaluasi_ujung_ke_ujung = _em.evaluasi_ujung_ke_ujung",
    "chunks = buat_semua_chunks()",
    "retrieve = buat_retriever()",
    "print('korpus:', len(chunks), 'chunk — siap diuji')",
)]
for judul, md_lines, kode_lines in SOL_BLOK:
    solusi += [md("## " + judul)]
    solusi += [code(*kode_lines)]

solusi += [md(
    "## Ringkasan Angka Terkunci",
    "",
    "```",
    "chunking      : 3 dokumen → 7 chunk (ukuran 320 / overlap 64 karakter)",
    "hit@1         : 0.8  (q1 gagal: FAQ menang top-1)",
    "hit@2, hit@3  : 1.0",
    "hybrid q1     : top-3 = [5, 0, 4] | hit@3 = 1.0",
    "rerank q2     : [3, 5, 4] → [3, 4]",
    "pipeline q1   : biaya $5.91e-05 | retrieve 47 ms | generate 50 ms",
    "cache 4 req   : miss, hit, hit, miss → 2 pencarian, hit_rate 0.5",
    "e2e           : akurasi 1.0 | abstain benar (skor luar korpus 0.0652)",
    "```",
)]
tulis("solusi.ipynb", solusi)

print("selesai.")
