"""Policy deterministik "LLM mini" agent — GIVEN, JANGAN DIUBAH.

Peran ganda:

1. **Klasifier intent** (`klasifikasi_intent`): kata kunci → intent. Meniru
   "otak" LLM yang memilih tool, tapi deterministik sehingga trace & angka
   evaluasi terkunci SEED 14.

2. **Generator rencana** (`rencana_untuk`): intent → daftar langkah
   (Thought/Action) ala ReAct. Loop agent-nya (yang kamu bangun di
   `loop_react.py`) mengeksekusi rencana ini langkah demi langkah, memanggil
   tool sungguhan lewat registry, lalu menyusun jawaban final.

3. **Tokeniser mini** (`token_estimasi`): proyeksi biaya (ceil(len/4)) — sama
   dengan Bab 11 & 13.

Di produksi, `klasifikasi_intent` diganti panggilan LLM dan `rencana_untuk`
diganti output tool-calling model — kontrak loop, guardrail, dan memory
TETAP SAMA. Itulah alasan mesinnya dipisah dari otaknya.
"""
from .data import INTENT_TOOL

KATA_KUNCI_INTENT = (
    # (tuple kata kunci, intent) — urutan PENTING: refund dicek sebelum kebijakan
    (("refund", "uang kembali", "pengembalian dana"), "refund"),
    (("status", "pesanan", "invoice", "inv-", "kirim ke mana", "dikirim"), "status_pesanan"),
    (("kebijakan", "pengembalian", "retur", "garansi", "cara kirim", "ongkir",
      "cod", "cashback", "pembayaran", "berapa lama"), "kebijakan"),
)


def klasifikasi_intent(teks):
    """Teks user → intent ('kebijakan' | 'status_pesanan' | 'refund' | 'lainnya').

    Aturan: intent pertama yang kata kuncinya muncul (substring, huruf kecil).
    Tanpa kecocokan → 'lainnya' (agent harus bertanya balik, bukan menebak).
    """
    t = str(teks).lower()
    for kata_kunci, intent in KATA_KUNCI_INTENT:
        if any(k in t for k in kata_kunci):
            return intent
    return "lainnya"


def ekstrak_invoice(teks):
    """Ambil ID invoice pertama (pola INV-###) dari teks; None bila tak ada."""
    import re
    hasil = re.search(r"INV-\d{3}", str(teks).upper())
    return hasil.group(0) if hasil else None


def rencana_untuk(intent, teks):
    """Intent → rencana ReAct deterministik: list langkah.

    Tiap langkah: {'thought': str, 'tool': nama|None, 'args': dict|None}.
    'tool' None berarti langkah FINAL (jawab user dari observasi terkumpul).

    - status_pesanan TANPA invoice id → langkah final 'tanya_balik'
      (agent jujur bertanya, bukan mengarang ID — pola abstain Bab 12);
    - refund SELALU berakhir langkah 'minta_konfirmasi' (human-in-the-loop).
    """
    if intent == "status_pesanan":
        inv = ekstrak_invoice(teks)
        if inv is None:
            return [{"thought": "User tanya status tapi tidak menyebut ID invoice.",
                     "tool": None, "args": {"aksi_final": "tanya_balik",
                                            "jawaban": "Boleh sebutkan nomor invoice Anda (format INV-XXX)?"}}]
        return [{"thought": f"Saya perlu cek status pesanan {inv} di sistem.",
                 "tool": "cek_pesanan", "args": {"invoice_id": inv}},
                {"thought": "Status sudah didapat — jawab user.",
                 "tool": None, "args": {"aksi_final": "jawab"}}]

    if intent == "refund":
        inv = ekstrak_invoice(teks)
        if inv is None:
            return [{"thought": "Refund butuh nomor invoice yang valid.",
                     "tool": None, "args": {"aksi_final": "tanya_balik",
                                            "jawaban": "Untuk refund, sebutkan nomor invoice (format INV-XXX) dan alasannya."}}]
        return [{"thought": f"Cek dulu status {inv} supaya refund kontekstual.",
                 "tool": "cek_pesanan", "args": {"invoice_id": inv}},
                {"thought": "Refund adalah aksi berisiko — minta konfirmasi manusia.",
                 "tool": None, "args": {"aksi_final": "minta_konfirmasi"}}]

    if intent == "kebijakan":
        return [{"thought": "Saya carikan kebijakan relevan di KB.",
                 "tool": "cari_dokumen", "args": {"kueri": str(teks)}},
                {"thought": "Kebijakan ditemukan — rangkum untuk user.",
                 "tool": None, "args": {"aksi_final": "jawab"}}]

    # 'lainnya': satu langkah klarifikasi — agent TIDAK menebak
    return [{"thought": "Saya tidak yakin maksud user — lebih jujur bertanya balik.",
             "tool": None, "args": {"aksi_final": "tanya_balik",
                                    "jawaban": "Maaf, saya belum paham. Bisa dijelaskan terkait status pesanan, kebijakan, atau refund?"}}]


def tool_untuk(intent):
    """Intent → nama tool utamanya (None bila 'lainnya')."""
    return INTENT_TOOL.get(intent)


def token_estimasi(teks):
    """Proyeksi token: ceil(len/4) — cukup untuk proyeksi biaya, bukan tagihan."""
    import math
    return math.ceil(len(str(teks)) / 4.0)
