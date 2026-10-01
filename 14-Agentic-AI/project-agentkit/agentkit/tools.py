"""TODO Bagian 1 — Registry tools + validasi skema + eksekusi.

Kontrak lengkap ada di tests/test_tools.py. Isi fungsi-fungsi di bawah
sampai semua test hijau.
"""
from .data import PESANAN, TOOLS_SCHEMA


def validasi_skema(nama_tool, args):
    """Validasi args terhadap skema TOOLS_SCHEMA[nama_tool].

    Return (ok, pesan_error): ok=True bila valid; ok=False + pesan bila tidak.
    Yang dicek: argumen required ada; type string; min_panjang; pola (regex
    penuh via re.search). Tool tak dikenal → (False, pesan).
    """
    # TODO
    raise NotImplementedError


def tool_cari_dokumen(kueri):
    """Pencarian kata kunci di DOKUMEN['kb'] → list hasil JSON-ready.

    Skor = kata kunci unik yang muncul di (judul + teks), dibagi jumlah kata
    kunci unik; ambil dokumen skor > 0, urut skor menurun, tie-break id.
    Tiap hasil: {'id', 'judul', 'teks', 'skor'} (skor dibulatkan 4 desimal).
    """
    # TODO
    raise NotImplementedError


def tool_cek_pesanan(invoice_id):
    """INV-### → dict pesanan (SALINAN, tanpa kunci 'user' — PII!) atau
    {'error': 'pesanan tidak ditemukan'} bila ID tak terdaftar."""
    # TODO
    raise NotImplementedError


def tool_refund(invoice_id, alasan):
    """Refund TIDAK dieksekusi sungguhan: return dict rencana konfirmasi
    {'butuh_konfirmasi': True, 'invoice_id', 'alasan', 'pesan'} — guardrail
    human-in-the-loop selalu menyala untuk aksi berisiko."""
    # TODO
    raise NotImplementedError


REGISTRY = {
    "cari_dokumen": tool_cari_dokumen,
    "cek_pesanan": tool_cek_pesanan,
    "refund": tool_refund,
}


def eksekusi_tool(nama_tool, args):
    """Registry dispatch: validasi skema dulu, lalu panggil tool.

    Return hasil tool (JSON-ready). Tool tak dikenal / argumen tak valid →
    {'error': pesan} (TIDAK raise — error adalah observasi bagi agent).
    """
    # TODO
    raise NotImplementedError
