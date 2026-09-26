"""plib — paket stdlib murni untuk project Bab 10 (Prompt Engineering).

Modul:
    data.py       : dataset & konstanta terkunci (JANGAN DIUBAH)  — GIVEN
    sections.py   : anatomi prompt + bagian wajib + anggaran token
    template.py   : render {{placeholder}} + escape delimiter
    fewshot.py    : pemilihan & pengurutan contoh few-shot
    schema.py     : validasi JSON-schema mini
    parse.py      : ekstraksi JSON dari output berisik + perbaikan
    guard.py      : deteksi prompt injection + pembungkusan data
    mock_llm.py   : simulator model deterministik                   — GIVEN
    evaluate.py   : harness evaluasi prompt (metrik + perbandingan)

Prasyarat: stdlib saja. Semua kontrak ada di tests/.
"""
__all__ = ["data", "sections", "template", "fewshot", "schema", "parse",
           "guard", "mock_llm", "evaluate"]
