"""Bagian 2 — Template {{placeholder}} & escape delimiter.  TODO: kamu yang isi.

Konvensi wajib (dipakai tests/):
  - Placeholder = `{{nama}}`, nama = identifier Python ([a-zA-Z_][a-zA-Z0-9_]*)
  - `render` BUKAN `str.format`: kurung kurawal JSON harus lolos apa adanya
  - Substitusi SINGLE-PASS: isi variabel tidak pernah di-render ulang
"""
import re

PAT_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def placeholder(template):
    """Nama placeholder unik, terurut (kanonik)."""
    # TODO
    raise NotImplementedError


def render(template, **nilai):
    """Isi placeholder {{nama}}. Placeholder tanpa nilai -> ValueError."""
    # TODO
    raise NotImplementedError


def render_aman(template, **nilai):
    """Versi yang tidak melempar: return (teks, daftar placeholder kurang)."""
    # TODO
    raise NotImplementedError


def escape_delimiter(teks, tag="dokumen"):
    """Buang tag pembuka & penutup `tag` dari data (cegah breakout delimiter)."""
    # TODO
    raise NotImplementedError
