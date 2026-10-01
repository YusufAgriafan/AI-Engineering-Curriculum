"""TODO Bagian 2 — Loop ReAct: reason → act → observe → final.

Kontrak lengkap ada di tests/test_loop_react.py. Isi fungsi-fungsi di bawah
sampai semua test hijau.
"""
from .data import MAKS_ITER


def format_langkah(thought, action=None, observation=None):
    """Satu langkah → teks blok ReAct terkunci:

    'Thought: <thought>'
    + 'Action: <tool>(<args>)' bila action bukan None   (args: dict → repr singkat)
    + 'Observation: <observation>' bila observation bukan None
    """
    # TODO
    raise NotImplementedError


def format_trace(langkah_list):
    """List langkah → satu string trace: blok tiap langkah dipisah baris kosong."""
    # TODO
    raise NotImplementedError


def jalankan_langkah(langkah):
    """Eksekusi satu langkah rencana.

    - tool None → observasi None (langkah final; aksi final di args);
    - tool ada → eksekusi_tool(nama, args) → observasi (hasil/error).
    Return (observasi, nama_tool_yang_dipakai atau None).
    """
    # TODO
    raise NotImplementedError


def jalankan_rencana(rencana, maks_iter=MAKS_ITER):
    """Eksekusi rencana ReAct dengan guardrail iterasi.

    - Eksekusi berurutan; kumpulkan (langkah, observasi) ke trace;
    - Langkah final (tool None) → selesai, return hasil final;
    - Bila rencana habis tanpa langkah final ATAU jumlah langkah mencapai
      maks_iter → paksa berhenti: final {'aksi_final': 'jawab',
      'jawaban': pesan escallate} dan tanda 'berhenti_dini' = True.

    Return dict {'final', 'langkah': [...], 'berhenti_dini': bool,
    'jumlah_langkah': int} — identik bentuk dengan agent.jawab().
    """
    # TODO
    raise NotImplementedError
