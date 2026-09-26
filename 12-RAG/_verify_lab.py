"""Verify 01_lab_rag.ipynb: exec every cell; TODO cells get reference impls.

Usage (dari folder ini):
    python _verify_lab.py
"""
import contextlib  # noqa: E402
import io  # noqa: E402
import json
import math  # noqa: E402
import os
import sys
import time  # noqa: E402
from pathlib import Path

BASE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.chdir(BASE)  # notebook memakai Path.cwd() untuk menemukan project-ragkit

NB = BASE / "01_lab_rag.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "potong_teks": (
        "def potong_teks(teks, ukuran=320, overlap=64):\n"
        "    kalimat = pecah_kalimat(teks)\n"
        "    chunks, current, current_len = [], [], 0\n"
        "    for s in kalimat:\n"
        "        tambah = len(s) + (1 if current else 0)\n"
        "        if current and current_len + tambah > ukuran:\n"
        "            chunks.append(' '.join(current))\n"
        "            tail, total = [], 0\n"
        "            for prev in reversed(current):\n"
        "                biaya = len(prev) + (1 if tail else 0)\n"
        "                if total + biaya > overlap:\n"
        "                    break\n"
        "                tail.insert(0, prev)\n"
        "                total += biaya\n"
        "            current, current_len = list(tail), total\n"
        "        current.append(s)\n"
        "        current_len += len(s) + (1 if len(current) > 1 else 0)\n"
        "    if current:\n"
        "        chunks.append(' '.join(current))\n"
        "    return chunks\n"
    ),
    "cosine_similarity": (
        "def cosine_similarity(a, b):\n"
        "    dot = sum(x * y for x, y in zip(a, b))\n"
        "    na = math.sqrt(sum(x * x for x in a))\n"
        "    nb = math.sqrt(sum(y * y for y in b))\n"
        "    if na == 0.0 or nb == 0.0:\n"
        "        return 0.0\n"
        "    return dot / (na * nb)\n"
    ),
    "VectorStoreLab": (
        "class VectorStoreLab:\n"
        "    def __init__(self):\n"
        "        self.chunks = []\n"
        "        self.matriks = []\n"
        "\n"
        "    def add(self, chunks, matriks):\n"
        "        self.chunks = list(chunks)\n"
        "        self.matriks = [list(v) for v in matriks]\n"
        "\n"
        "    def cari(self, vektor_kueri, top_k=3):\n"
        "        if not self.chunks:\n"
        "            return []\n"
        "        skor = [cosine_similarity(vektor_kueri, v) for v in self.matriks]\n"
        "        urut = sorted(range(len(skor)), key=lambda i: (-skor[i], i))[:top_k]\n"
        "        return [{'chunk': self.chunks[i], 'score': skor[i], 'index': i}\n"
        "                for i in urut]\n"
    ),
    "evaluasi_retrieval": (
        "def evaluasi_retrieval(retrieve, kueri_uji=None, top_k=3):\n"
        "    kueri_uji = KUERI_UJI if kueri_uji is None else kueri_uji\n"
        "    per_kueri, hit, kueri_gagal = [], 0, []\n"
        "    for q in kueri_uji:\n"
        "        hasil = retrieve(q['teks'], top_k)\n"
        "        sumber = [r['chunk']['source'] for r in hasil]\n"
        "        ok = q['source_emas'] in sumber\n"
        "        per_kueri.append({'id': q['id'], 'hit': ok,\n"
        "                          'posisi': (sumber.index(q['source_emas']) + 1) if ok else None})\n"
        "        if ok:\n"
        "            hit += 1\n"
        "        else:\n"
        "            kueri_gagal.append(q['id'])\n"
        "    return {'per_kueri': per_kueri,\n"
        "            'hit_rate': hit / len(kueri_uji) if kueri_uji else 0.0,\n"
        "            'kueri_gagal': kueri_gagal}\n"
    ),
    "buat_retriever_lab": (
        "def buat_retriever_lab(store=None, encoder=None):\n"
        "    encoder = enc if encoder is None else encoder\n"
        "    if store is None:\n"
        "        store = VectorStoreLab()\n"
        "        store.add(chunks, encoder.embed([c['text'] for c in chunks]))\n"
        "\n"
        "    def retrieve(kueri, top_k=3):\n"
        "        return store.cari(encoder.embed_query(kueri), top_k)\n"
        "\n"
        "    return retrieve\n"
    ),
    "abstaikan_lab": (
        "def abstaikan_lab(hasil, ambang=0.30):\n"
        "    skor_terbaik = hasil.get('skor_terbaik', 0.0)\n"
        "    if hasil.get('abstain') or skor_terbaik >= ambang:\n"
        "        return dict(hasil)\n"
        "    keluar = dict(hasil)\n"
        "    keluar['jawaban'] = SEP_DETEKSI_TIDAK_ADA\n"
        "    keluar['sitasi'] = []\n"
        "    keluar['sumber'] = []\n"
        "    keluar['abstain'] = True\n"
        "    keluar['abstaikan_oleh'] = round(float(skor_terbaik), 4)\n"
        "    return keluar\n"
    ),
    "cari_bm25_lab": (
        "def cari_bm25_lab(kueri, chunks, top_k=3):\n"
        "    k1, b = 1.2, 0.75\n"
        "    docs = [tokenisasi(c['text']) for c in chunks]\n"
        "    n = len(docs)\n"
        "    if n == 0:\n"
        "        return []\n"
        "    rata_len = sum(len(d) for d in docs) / n\n"
        "    df = {}\n"
        "    for d in docs:\n"
        "        for t in set(d):\n"
        "            df[t] = df.get(t, 0) + 1\n"
        "    skor = [0.0] * n\n"
        "    for t in tokenisasi(kueri):\n"
        "        if t not in df:\n"
        "            continue\n"
        "        idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))\n"
        "        for i, d in enumerate(docs):\n"
        "            tf = d.count(t)\n"
        "            if tf:\n"
        "                skor[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(d) / rata_len))\n"
        "    urut = sorted(range(n), key=lambda i: (-skor[i], i))[:top_k]\n"
        "    return [{'chunk': chunks[i], 'score': skor[i], 'index': i} for i in urut]\n"
    ),
    "gabung_rrf_lab": (
        "def gabung_rrf_lab(daftar_dapat, k=60):\n"
        "    skor = {}\n"
        "    for dapat in daftar_dapat:\n"
        "        for rank, item in enumerate(dapat):\n"
        "            idx = item['chunk']['chunk_index']\n"
        "            skor[idx] = skor.get(idx, 0.0) + 1.0 / (k + rank + 1)\n"
        "    return skor\n"
    ),
    "cari_hybrid_lab": (
        "def cari_hybrid_lab(retrieve_vektor, chunks, kueri, top_k=3, rrf_k=60):\n"
        "    d_vek = retrieve_vektor(kueri, top_k)\n"
        "    d_bm25 = cari_bm25_lab(kueri, chunks, top_k)\n"
        "    skor = gabung_rrf_lab([d_vek, d_bm25], k=rrf_k)\n"
        "    by_index = {c['chunk_index']: c for c in chunks}\n"
        "    urut = sorted(skor.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]\n"
        "    return [{'chunk': by_index[idx], 'score': s, 'index': idx} for idx, s in urut]\n"
    ),
    "rerank_lab": (
        "def rerank_lab(kueri, hasil, top_k=2, skorer=None):\n"
        "    skorer = skorer_kata_lab if skorer is None else skorer\n"
        "    diurut = sorted(hasil,\n"
        "                    key=lambda r: (-skorer(kueri, r['chunk']['text']),\n"
        "                                   r['chunk']['chunk_index']))\n"
        "    return diurut[:top_k]\n"
    ),
    "skorer_kata_lab": (
        "def skorer_kata_lab(kueri, teks):\n"
        "    stopword = {'berapa', 'bagaimana', 'apa', 'yang', 'untuk', 'dengan',\n"
        "                'adalah', 'tanpa', 'dari', 'ke', 'di', 'dan', 'atau', 'hari'}\n"
        "    kata = [t for t in tokenisasi(kueri) if t not in stopword]\n"
        "    teks_low = str(teks).lower()\n"
        "    skor = 0.0\n"
        "    for k in kata:\n"
        "        bobot = 3.0 if k.isdigit() else 1.0\n"
        "        if k in teks_low:\n"
        "            skor += bobot\n"
        "    return skor\n"
    ),
    "jalankan_rag_lab": (
        "def jalankan_rag_lab(retrieve, kueri, generator=None, top_k=3, ambang=0.30,\n"
        "                     generator_latency_ms=25):\n"
        "    generator = GeneratorEkstraktif() if generator is None else generator\n"
        "    dapat = retrieve(kueri, top_k)\n"
        "    chunks_pilihan = [d['chunk'] for d in dapat]\n"
        "    skor_terbaik = dapat[0]['score'] if dapat else 0.0\n"
        "    retrieve_ms = 8 + 2 * len(chunks) + generator_latency_ms\n"
        "\n"
        "    hasil = jawab(generator, kueri, chunks_pilihan)\n"
        "    hasil['skor_terbaik'] = skor_terbaik\n"
        "    hasil = abstaikan_lab(hasil, ambang)\n"
        "    generate_ms = 2 * generator_latency_ms\n"
        "\n"
        "    token_in = math.ceil(len(hasil.get('prompt', '')) / 4)\n"
        "    token_out = math.ceil(len(hasil['jawaban']) / 4)\n"
        "    biaya = token_in / 1e6 * 0.15 + token_out / 1e6 * 0.60\n"
        "\n"
        "    keluar = {'kueri': kueri, 'chunks': chunks_pilihan,\n"
        "              'jawaban': hasil['jawaban'], 'sitasi': hasil['sitasi'],\n"
        "              'sumber': hasil['sumber'], 'abstain': hasil['abstain'],\n"
        "              'skor_terbaik': skor_terbaik,\n"
        "              'skema': hasil.get('prompt', ''), 'biaya_usd': biaya,\n"
        "              'tahap': {'retrieve_ms': retrieve_ms,\n"
        "                        'generate_ms': generate_ms}}\n"
        "    if 'abstaikan_oleh' in hasil:\n"
        "        keluar['abstaikan_oleh'] = hasil['abstaikan_oleh']\n"
        "    return keluar\n"
    ),
}


def nama_ref(src):
    """Nama REF yang didefinisikan di cell ini (fungsi ATAU kelas)."""
    return [n for n in REF
            if f"def {n}(" in src or f"class {n}(" in src or f"class {n}:" in src]


ctx = {"__name__": "__main__", "__builtins__": __builtins__}

t0 = time.time()
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src:
        names = nama_ref(src)
        if not names:
            print(f"[cell {i:>2}] SKIPPED (TODO tanpa fungsi/kelas dikenal)")
            continue
        for n in names:
            exec(REF[n], ctx)
        print(f"[cell {i:>2}] injected: {', '.join(names)}")
        continue
    cap = io.StringIO()
    try:
        with contextlib.redirect_stdout(cap):
            exec(compile(src, f"<lab cell {i}>", "exec"), ctx)
    except Exception as e:
        print(f"FAILED di cell {i}: {type(e).__name__}: {e}")
        print(cap.getvalue()[-1500:])
        sys.exit(1)
    out = cap.getvalue()
    if "\u2705" in out:
        baris = [l for l in out.splitlines() if "\u2705" in l]
        print(f"[cell {i:>2}] OK  {baris[0][:88] if baris else ''}")
    else:
        print(f"[cell {i:>2}] OK ({len(out)} chars)")

print(f"\nSemua cell lab dieksekusi dalam {time.time() - t0:.1f} detik.")
print("LAB VERIFY OK")
