"""TODO Bagian 5 — Agent end-to-end + planner + evaluasi trace.

Kontrak lengkap ada di tests/test_agent_eval.py. Isi fungsi di bawah
sampai semua test hijau.
"""
from .data import AMBAT_SKOR_RAG
from .nlu import klasifikasi_intent, rencana_untuk


def jawab(teks, maks_iter=6, skor_rag_min=AMBAT_SKOR_RAG):
    """teks user → dict jawaban agent end-to-end (tanpa state antar panggilan).

    Return {'intent', 'final', 'langkah', 'jumlah_langkah', 'berhenti_dini'}.
    Alur: klasifikasi intent → rencana → jalankan_rencana → tambah 'intent'.
    """
    # TODO
    raise NotImplementedError


def planner_bertahap(intent, teks):
    """Planner 'kalau-perlu' (bandingkan dengan rencana statis nlu.rencana_untuk):

    - 'status_pesanan' TANPA id invoice di teks → rencana tanya_balik (1 langkah);
    - 'kebijakan' → cari_dokumen lalu final 'jawab' (2 langkah);
    - 'refund' DENGAN id → cek_pesanan lalu final 'minta_konfirmasi' (2 langkah);
    - 'refund' TANPA id → rencana tanya_balik (1 langkah);
    - selain itu → rencana tanya_balik (1 langkah).

    Bentuk langkah SAMA dengan rencana_untuk — planner hanya menentukan
    KAPAN langkah tambah diperlukan, bukan mengubah kontrak langkah.
    """
    # TODO
    raise NotImplementedError


def evaluasi_trace(hasil):
    """Hasil jalankan_rencana/jawab → skor kualitas + temuan.

    Return {'skor': float 0..1, 'temuan': list[str]} dengan aturan:
    - tool dipanggil tanpa error, tanpa langkah 'tanya_balik', tanpa
      berhenti dini: skor 1.0;
    - ada 'tanya_balik' (dan tidak ada masalah lain): skor 0.5;
    - ada error tool ATAU berhenti_dini: skor 0.0;
    - temuan: pesan masalah berurutan (error tool → tanya_balik → berhenti
      dini) — ['bersih'] bila tak ada.
    """
    # TODO
    raise NotImplementedError


def verifikasi_tools():
    """Health check registry: setiap tool dipanggil dengan args valid →
    True bila SEMUA hasil tak punya kunci 'error', selain itu False."""
    # TODO
    raise NotImplementedError
