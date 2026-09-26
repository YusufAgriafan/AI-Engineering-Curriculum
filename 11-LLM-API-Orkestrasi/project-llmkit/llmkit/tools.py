"""Bagian 6 — Function calling: skema, validasi argumen, loop.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - `skema_parameter` menurunkan JSON-schema dari contoh argumen di
    `data.TOOLS[nama]['argumen']` (str->string, int->integer, float->number,
    bool->boolean); semua kunci = required
  - Error validasi stabil: "missing:<k>", "type:<k>", "tak_dikenal:<k>",
    "tak_dikenal:<tool>", "type:<argumen>"
  - `loop_tool` berhenti saat model tidak lagi meminta tool ATAU `maks_iterasi`
    tercapai (maka hasilnya punya `berhenti="maks_iterasi"` dan `jawaban=None`)
  - Hasil tool (atau pesan galatnya) DIKIRIM BALIK ke model sebagai pesan "tool"
"""
import json

from .data import TOOLS


def skema_parameter(nama, tools=None):
    """JSON-schema mini dari contoh argumen."""
    # TODO
    raise NotImplementedError


def skema_tools(tools=None):
    """Bentuk tools untuk API function calling."""
    # TODO
    raise NotImplementedError


def validasi_argumen(nama, argumen, tools=None):
    """Daftar error argumen (kosong = valid)."""
    # TODO
    raise NotImplementedError


def jalankan_tool(nama, argumen, tools=None):
    """Validasi lalu eksekusi handler. Return {'hasil', 'galat', 'errors'}."""
    # TODO
    raise NotImplementedError


def loop_tool(provider, model, pesan_awal, *, tools=None, maks_iterasi=4,
              timeout_ms=5000, temperature=0.0, maks_token=200):
    """Loop function calling.

    Return: jawaban, iterasi, tool_dipakai, riwayat, tokens_in, tokens_out, total_ms
    (+ berhenti="maks_iterasi" bila mentok).
    """
    # TODO
    raise NotImplementedError
