"""Implementasi REFERENSI ragkit — untuk cek mandiri & verifier.

JANGAN dibaca sebelum mencoba; ini kunci jawaban seluruh modul TODO.
Bentuk & kontrak tiap fungsi = yang diuji tests/.
"""
import math

from ragkit.chunking import buat_semua_chunks, pecah_kalimat
from ragkit.data import (KUERI_LUAR_CORPUS, KUERI_UJI, LATENSI_CACHE_MS,
                         SEP_DETEKSI_TIDAK_ADA, TOLOK_UKUR)
from ragkit.embed import EncoderEmbedding, cosine, tokenisasi
from ragkit.generator import GeneratorEkstraktif, jawab, prompt_rag


# ============================ 1. vectorstore ============================

class VectorStore:
    """Vector store in-memory: cosine similarity (vektor ternormalisasi L2)."""

    def __init__(self):
        self.chunks = []
        self.matriks = []

    def add(self, chunks, matriks):
        self.chunks = list(chunks)
        self.matriks = [list(v) for v in matriks]

    def cari(self, vektor_kueri, top_k=3):
        if not self.chunks:
            return []
        skor = [cosine(vektor_kueri, v) for v in self.matriks]
        urut = sorted(range(len(skor)), key=lambda i: (-skor[i], i))[:top_k]
        return [{"chunk": self.chunks[i], "score": skor[i], "index": i}
                for i in urut]


# ============================ 2. retriever ============================

def buat_retriever(encoder=None, chunks=None):
    """Index korpus sekali → fungsi retrieve(kueri, top_k) siap pakai."""
    encoder = EncoderEmbedding() if encoder is None else encoder
    chunks = buat_semua_chunks() if chunks is None else chunks
    store = VectorStore()
    store.add(chunks, encoder.embed([c["text"] for c in chunks]))

    def retrieve(kueri, top_k=3):
        return store.cari(encoder.embed_query(kueri), top_k)

    return retrieve


def evaluasi_retrieval(retrieve, kueri_uji=None, top_k=3):
    """hit@k per kueri + agregat; kueri terluar (q6) tidak ikut rata-rata.

    Return: {per_kueri: [...], hit_rate: float, kueri_gagal: [id]}
    """
    kueri_uji = KUERI_UJI if kueri_uji is None else kueri_uji
    per_kueri, hit, kueri_gagal = [], 0, []
    for q in kueri_uji:
        hasil = retrieve(q["teks"], top_k)
        sumber = [r["chunk"]["source"] for r in hasil]
        ok = q["source_emas"] in sumber
        per_kueri.append({"id": q["id"], "hit": ok,
                          "posisi": (sumber.index(q["source_emas"]) + 1) if ok else None})
        if ok:
            hit += 1
        else:
            kueri_gagal.append(q["id"])
    return {"per_kueri": per_kueri,
            "hit_rate": hit / len(kueri_uji) if kueri_uji else 0.0,
            "kueri_gagal": kueri_gagal}


# ============================ 3. abstain gate ============================

def abstaikan(hasil, ambang=0.30):
    """Timpa jawaban jadi abstain bila skor retrieval terbaik < ambang.

    Jangan mengganti jawaban yang sudah abstain, dan jangan menurunkan
    abstain → jawaban. Return dict BARU dengan kunci 'abstaikan_oleh'.
    """
    skor_terbaik = hasil.get("skor_terbaik", 0.0)
    sudah_abstain = bool(hasil.get("abstain"))
    if sudah_abstain or skor_terbaik >= ambang:
        return dict(hasil)
    keluar = dict(hasil)
    keluar["jawaban"] = SEP_DETEKSI_TIDAK_ADA
    keluar["sitasi"] = []
    keluar["sumber"] = []
    keluar["abstain"] = True
    keluar["abstaikan_oleh"] = round(float(skor_terbaik), 4)
    return keluar


def _kueri_semirip(a, b):
    """Cukup untuk cache retrieval: 60% kata penting sama."""
    ka = {t for t in tokenisasi(a) if len(t) > 2}
    kb = {t for t in tokenisasi(b) if len(t) > 2}
    if not ka or not kb:
        return False
    irisan = len(ka & kb) / max(len(ka), len(kb))
    return irisan >= 0.6


# ============================ 4. cache retrieval ============================

class CacheRetrieval:
    """Cache konteks retrieval: kunci = kueri; TTL tick logis.

    Bedanya dengan cache LLM (Bab 11): yang di-cache KONTEKS (hasil retrieval),
    bukan jawaban — biaya yang dihindari adalah pencarian + embedding ulang,
    dan jawaban tetap digenerate segar dari konteks yang sama.
    """

    def __init__(self, ttl=10):
        self.ttl = ttl
        self._isi = {}
        self.hit = 0
        self.miss = 0
        self.expired = 0

    def kunci(self, kueri):
        return kueri.strip().lower()

    def ambil(self, kunci, sekarang):
        if kunci not in self._isi:
            self.miss += 1
            return None
        hasil, tick = self._isi[kunci]
        if sekarang - tick >= self.ttl:
            self.expired += 1
            del self._isi[kunci]
            return None
        self.hit += 1
        return hasil

    def simpan(self, kunci, hasil, sekarang):
        self._isi[kunci] = (hasil, sekarang)

    def statistik(self):
        total = self.hit + self.miss + self.expired
        return {"hit": self.hit, "miss": self.miss, "expired": self.expired,
                "ukuran": len(self._isi),
                "hit_rate": (self.hit / total) if total else 0.0}


def retrieve_bercache(retrieve, kueri, cache, sekarang=0, top_k=3,
                      latensi_ms=LATENSI_CACHE_MS):
    """Retrieve lewat cache: hit → tanpa embedding/pencarian (latensi ~2 ms)."""
    kunci = cache.kunci(kueri)
    tersimpan = cache.ambil(kunci, sekarang)
    if tersimpan is not None:
        return {"hasil": tersimpan, "cache": "hit", "latensi_ms": latensi_ms}
    mulai = [r["chunk"] for r in retrieve(kueri, top_k)]
    cache.simpan(kunci, mulai, sekarang)
    return {"hasil": mulai, "cache": "miss", "latensi_ms": latensi_ms * 8}


# ============================ 5. hybrid (RRF) ============================

def cari_bm25(kueri, chunks, top_k=3):
    """BM25 mini (stdlib): k=1.2, b=0.75 — tanpa library eksternal."""
    k1, b = 1.2, 0.75
    docs = [tokenisasi(c["text"]) for c in chunks]
    n = len(docs)
    if n == 0:
        return []
    rata_len = sum(len(d) for d in docs) / n
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    skor = [0.0] * n
    for t in tokenisasi(kueri):
        if t not in df:
            continue
        idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(t)
            if tf:
                skor[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(d) / rata_len))
    urut = sorted(range(n), key=lambda i: (-skor[i], i))[:top_k]
    return [{"chunk": chunks[i], "score": skor[i], "index": i} for i in urut]


def gabung_rrf(daftar_dapat, k=60):
    """Reciprocal Rank Fusion: skor = Σ 1/(k + peringkat).

    daftar_dapat: list of list of chunk-dict (hasil tiap pencarian).
    """
    skor = {}
    for dapat in daftar_dapat:
        for rank, item in enumerate(dapat):
            idx = item["chunk"]["chunk_index"]
            skor[idx] = skor.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return skor


def cari_hybrid(retrieve_vektor, chunks, kueri, top_k=3, rrf_k=60):
    """Vektor + BM25 → gabung RRF → urutkan → ambil top_k."""
    d_vek = retrieve_vektor(kueri, top_k)
    d_bm25 = cari_bm25(kueri, chunks, top_k)
    skor = gabung_rrf([d_vek, d_bm25], k=rrf_k)
    by_index = {c["chunk_index"]: c for c in chunks}
    urut = sorted(skor.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [{"chunk": by_index[idx], "score": s, "index": idx} for idx, s in urut]


# ============================ 6. reranker ============================

def skorer_kata(kueri, teks):
    """Skor leksikal murah: Σ bobot kata kueri yang ada di teks.

    Bobot: angka 3.0 (paling menentukan untuk 'berapa hari'), kata lain 1.0.
    Stopword kecil dibuang. Deterministik & cepat — di produksi digantikan
    cross-encoder (bge-reranker, Cohere Rerank), tapi POLANYA sama:
    kandidat banyak → skor mahal-per-item hanya untuk sedikit kandidat.
    """
    kata = [t for t in tokenisasi(kueri) if t not in
            {"berapa", "bagaimana", "apa", "yang", "untuk", "dengan", "adalah",
             "tanpa", "dari", "ke", "di", "dan", "atau", "hari"}]
    teks_low = str(teks).lower()
    skor = 0.0
    for k in kata:
        bobot = 3.0 if k.isdigit() else 1.0
        if k in teks_low:
            skor += bobot
    return skor


def rerank(kueri, hasil, top_k=2, skorer=None):
    """Urutkan ulang kandidat dengan skorer murah, ambil top_k terbaik."""
    skorer = skorer_kata if skorer is None else skorer
    diurut = sorted(hasil,
                    key=lambda r: (-skorer(kueri, r["chunk"]["text"]),
                                   r["chunk"]["chunk_index"]))
    return diurut[:top_k]


# ============================ 7. pipeline end-to-end ============================

def _latensi_index(index):
    """Latensi pencarian deterministik dari jumlah vektor di index (ms)."""
    return 8 + 2 * int(index)


def jalankan_rag(retrieve, kueri, generator=None, top_k=3, ambang=0.30,
                 generator_latency_ms=25):
    """Satu permintaan RAG end-to-end + pelaporan per tahap.

    Return: kueri, chunks, jawaban, sitasi, sumber, abstain, skor_terbaik,
    skema (prompt), biaya_usd, tahap {retrieve_ms, generate_ms}.
    """
    generator = GeneratorEkstraktif() if generator is None else generator
    chunks_all = buat_semua_chunks()
    n_index = len(chunks_all)

    t0 = _latensi_index(n_index)
    dapat = retrieve(kueri, top_k)
    chunks = [d["chunk"] for d in dapat]
    skor_terbaik = dapat[0]["score"] if dapat else 0.0
    retrieve_ms = t0 + generator_latency_ms   # embedding kueri + pencarian

    hasil = jawab(generator, kueri, chunks)
    hasil["skor_terbaik"] = skor_terbaik
    hasil = abstaikan(hasil, ambang)
    generate_ms = generator_latency_ms * 2    # sengaja deterministik

    token_in = math.ceil(len(hasil.get("prompt", "")) / 4)
    token_out = math.ceil(len(hasil["jawaban"]) / 4)
    biaya = token_in / 1e6 * 0.15 + token_out / 1e6 * 0.60   # tarif "mini"

    keluar = {"kueri": kueri, "chunks": chunks, "jawaban": hasil["jawaban"],
              "sitasi": hasil["sitasi"], "sumber": hasil["sumber"],
              "abstain": hasil["abstain"], "skor_terbaik": skor_terbaik,
              "skema": hasil.get("prompt", ""), "biaya_usd": biaya,
              "tahap": {"retrieve_ms": retrieve_ms, "generate_ms": generate_ms}}
    if "abstaikan_oleh" in hasil:          # bukti gate: skor yang memicunya
        keluar["abstaikan_oleh"] = hasil["abstaikan_oleh"]
    return keluar


def tolok_ukur(retrieve, n_permintaan=None, aturan=None, ambang=0.30):
    """Ukur manfaat cache retrieval di atas pipeline.

    Aturan: kueri per permintaan (index 0 & 1 identik → uji cache).
    Return: {latensi_ms: [..], hit_rate: float, panggilan_pencarian: int,
             total_biaya_usd, jawaban: [...], cache: [...]}
    """
    aturan = TOLOK_UKUR["aturan"] if aturan is None else aturan
    n_permintaan = TOLOK_UKUR["n_permintaan"] if n_permintaan is None else n_permintaan
    cache = CacheRetrieval(ttl=10)
    generator = GeneratorEkstraktif()
    latensi, jawaban, cache_status = [], [], []
    panggilan = 0
    total_biaya = 0.0
    for i, teks in enumerate(aturan[:n_permintaan]):
        r = retrieve_bercache(retrieve, teks, cache, sekarang=i, top_k=3)
        panggilan += 0 if r["cache"] == "hit" else 1
        cache_status.append(r["cache"])
        chunks = r["hasil"]
        hasil = jawab(generator, teks, chunks)
        hasil["skor_terbaik"] = dapat_skor(retrieve, teks)
        hasil = abstaikan(hasil, ambang)
        latensi.append(r["latensi_ms"] + 50)   # + generate (deterministik)
        jawaban.append(hasil["jawaban"])
        total_biaya += 5e-06
    return {"latensi_ms": latensi, "cache": cache_status,
            "hit_rate": cache.statistik()["hit_rate"],
            "panggilan_pencarian": panggilan, "total_biaya_usd": total_biaya,
            "jawaban": jawaban}


def dapat_skor(retrieve, kueri):
    dapat = retrieve(kueri, 1)
    return dapat[0]["score"] if dapat else 0.0


# ============================ 8. evaluasi end-to-end ============================

def evaluasi_ujung_ke_ujung(jalankan, kueri_uji=None, ambang=0.30):
    """Jawab semua kueri uji (+1 di luar korpus) → faithfulness & abstain.

    - faithfulness/akurasi: jawaban_emas ada di teks jawaban;
    - abstain benar: kueri luar korpus harus dijawab 'tidak ada di dokumen'.
    """
    kueri_uji = KUERI_UJI if kueri_uji is None else kueri_uji
    per_kueri, benar = [], 0
    for q in kueri_uji:
        h = jalankan(q["teks"])
        ok = q["jawaban_emas"].lower() in str(h["jawaban"]).lower()
        benar += 1 if ok else 0
        per_kueri.append({"id": q["id"], "benar": ok, "abstain": h["abstain"],
                          "jawaban": h["jawaban"][:60]})
    h6 = jalankan(KUERI_LUAR_CORPUS["teks"])
    abstain_ok = h6["abstain"] and SEP_DETEKSI_TIDAK_ADA in h6["jawaban"]
    total = len(kueri_uji)
    return {"per_kueri": per_kueri, "akurasi": (benar / total) if total else 0.0,
            "abstain_benar": abstain_ok, "jawaban_luar": h6["jawaban"]}


# dipakai verifier untuk memastikan chunking ref = chunking GIVEN
def potong_teks_ref(teks, ukuran=320, overlap=64):
    """Referensi potong_teks: identik dengan ragkit.chunking.potong_teks."""
    kalimat = pecah_kalimat(teks)
    chunks, current, current_len = [], [], 0
    for s in kalimat:
        tambah = len(s) + (1 if current else 0)
        if current and current_len + tambah > ukuran:
            chunks.append(" ".join(current))
            tail, total = [], 0
            for prev in reversed(current):
                biaya = len(prev) + (1 if tail else 0)
                if total + biaya > overlap:
                    break
                tail.insert(0, prev)
                total += biaya
            current, current_len = list(tail), total
        current.append(s)
        current_len += len(s) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks
