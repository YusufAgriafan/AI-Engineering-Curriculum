"""Potongan: chunking deterministik — GIVEN, JANGAN DIUBAH.

Fungsi & aturannya SATU untuk seluruh project (lab, kuis, pipeline) supaya
angka evaluasi bisa dibandingkan antar konfigurasi. Ukuran dihitung per KARAKTER
(320/64) — di dunia nyata hitung per token, tapi prinsip dan trade-off-nya sama.

Batas potong SENGAJA di batas kalimat (terakhir yang muat), bukan tengah kata:
kualitas retrieval tergantung chunk yang "utuh maknanya".

Kamu di lab nanti menulis `potong_teks` SENDIRI di notebook (dengan hasil yang
harus identik); versi di sini adalah referensi GIVEN untuk pipeline & tests.
"""
import re

from .data import DOKUMEN

UKURAN_CHUNK = 320      # karakter (~80 token; produksi: 300-800 token)
OVERLAP_CHUNK = 64      # karakter (~16 token; produksi: 10-20% ukuran)

_PAT_KALIMAT = re.compile(r"(?<=[.!?])\s+")


def pecah_kalimat(teks):
    """Pecah teks jadi kalimat; baris baru diratakan jadi spasi."""
    rata = " ".join(str(teks).split())
    return [s.strip() for s in _PAT_KALIMAT.split(rata) if s.strip()]


def potong_teks(teks, ukuran=UKURAN_CHUNK, overlap=OVERLAP_CHUNK):
    """Potong teks per kalimat: maksimal `ukuran` karakter, overlap `overlap`.

    Algoritma (deterministik — versi di lab HARUS menghasilkan chunk yang sama):
      1. pecah jadi kalimat;
      2. kumpulkan kalimat sampai menambahi melewati `ukuran` → simpan chunk;
      3. mulai chunk berikutnya dari kalimat TAIL (overlap) — kalimat utuh dari
         akhir chunk sebelumnya sepanjang totalnya masih <= `overlap`;
      4. sisa < `ukuran` jadi chunk terakhir.
    """
    kalimat = pecah_kalimat(teks)
    chunks = []
    current = []
    current_len = 0
    for s in kalimat:
        tambah = len(s) + (1 if current else 0)
        if current and current_len + tambah > ukuran:
            chunks.append(" ".join(current))
            # overlap: ambil kalimat ekor yang masih muat
            tail = []
            total = 0
            for prev in reversed(current):
                biaya = len(prev) + (1 if tail else 0)
                if total + biaya > overlap:
                    break
                tail.insert(0, prev)
                total += biaya
            current = list(tail)
            current_len = total
        current.append(s)
        current_len += len(s) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def buat_semua_chunks(dokumen=None, ukuran=UKURAN_CHUNK, overlap=OVERLAP_CHUNK):
    """Dokumen → daftar chunk dict: text, doc_id, source, page, chunk_index.

    chunk_index GLOBAL dan deterministik (urut dokumen → urut halaman → urut
    chunk). Metadata source+page inilah bahan sitasi jawaban.
    """
    dokumen = DOKUMEN if dokumen is None else dokumen
    chunks = []
    for doc in dokumen:
        for no, hal in enumerate(doc["halaman"], start=1):
            for potong in potong_teks(hal, ukuran, overlap):
                chunks.append({
                    "text": potong,
                    "doc_id": doc["id"],
                    "source": doc["source"],
                    "page": no,
                    "chunk_index": len(chunks),
                })
    return chunks
