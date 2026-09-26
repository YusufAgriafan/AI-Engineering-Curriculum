"""llmkit — paket stdlib murni untuk project Bab 11 (LLM API & Orkestrasi).

Modul:
    data.py        : tarif, rencana kegagalan, tools, pipeline (JANGAN DIUBAH)
    transport.py   : provider LLM palsu & deterministik               — GIVEN
    cost.py        : estimasi token, biaya, penjaga anggaran
    retry.py       : retry + backoff eksponensial, galat transien vs fatal
    fallback.py    : rantai fallback multi-provider
    cache.py       : cache konten-address + TTL, hit-rate
    stream.py      : merakit chunk streaming & mengukur TTFT
    tools.py       : function calling: skema, validasi argumen, loop
    orchestrate.py : prompt chaining (pipeline) + laporan per langkah

Prasyarat: stdlib saja. Semua kontrak ada di tests/.
"""
__all__ = ["data", "transport", "cost", "retry", "fallback", "cache",
           "stream", "tools", "orchestrate"]
