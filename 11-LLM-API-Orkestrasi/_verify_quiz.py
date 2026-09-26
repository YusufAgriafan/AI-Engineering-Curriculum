"""Verify quiz notebook: inject perfect answers, run grader, expect full score.

Usage (dari folder ini):
    python _verify_quiz.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_kuis_llm_api.ipynb"
EXPECTED = 22

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "c", 2: "b", 3: "a", 4: "b", 5: "d", 6: "c"},
    "backoff_ms": (
        "def backoff_ms(percobaan, basis_ms=1000, faktor=2, maks_ms=8000):\n"
        "    return min(basis_ms * (faktor ** (percobaan - 1)), maks_ms)\n"
    ),
    "panggil_dengan_retry": (
        "def panggil_dengan_retry(provider, messages, maks_percobaan=4, basis_ms=1000,\n"
        "                         faktor=2, maks_ms=8000, sleep_fn=None):\n"
        "    riwayat, tunggu = [], []\n"
        "    for percobaan in range(1, maks_percobaan + 1):\n"
        "        try:\n"
        "            hasil = provider.panggil(messages)\n"
        "        except GalatAPI as e:\n"
        "            riwayat.append((percobaan, e.jenis))\n"
        "            transien = e.jenis in JENIS_TRANSIEN\n"
        "            if not transien:\n"
        "                raise\n"
        "            if percobaan == maks_percobaan:\n"
        "                raise GalatSemuaPercobaan(riwayat) from e\n"
        "            jeda = backoff_ms(percobaan, basis_ms, faktor, maks_ms)\n"
        "            tunggu.append(jeda)\n"
        "            if sleep_fn is not None:\n"
        "                sleep_fn(jeda)\n"
        "            continue\n"
        "        hasil = dict(hasil)\n"
        "        hasil['attempts'] = percobaan\n"
        "        hasil['timpenungguan_ms'] = list(tunggu)\n"
        "        hasil['total_ms'] = hasil['latency_ms'] + sum(tunggu)\n"
        "        hasil['riwayat'] = list(riwayat)\n"
        "        return hasil\n"
        "    raise GalatSemuaPercobaan(riwayat)\n"
    ),
    "validasi_argumen": (
        "TIPE_JSON = {str: 'string', int: 'integer', float: 'number', bool: 'boolean'}\n"
        "\n"
        "\n"
        "def _tipe_ok(nilai, tipe):\n"
        "    if tipe == 'string':\n"
        "        return isinstance(nilai, str)\n"
        "    if tipe == 'integer':\n"
        "        return isinstance(nilai, int) and not isinstance(nilai, bool)\n"
        "    if tipe == 'number':\n"
        "        return isinstance(nilai, (int, float)) and not isinstance(nilai, bool)\n"
        "    if tipe == 'boolean':\n"
        "        return isinstance(nilai, bool)\n"
        "    return False\n"
        "\n"
        "\n"
        "def _skema(nama, tools):\n"
        "    argumen = tools[nama]['argumen']\n"
        "    props = {k: TIPE_JSON.get(type(v), 'string') for k, v in argumen.items()}\n"
        "    return props, list(argumen.keys())\n"
        "\n"
        "\n"
        "def validasi_argumen(nama, argumen, tools=TOOLS):\n"
        "    if nama not in tools:\n"
        "        return [f'tak_dikenal:{nama}']\n"
        "    if not isinstance(argumen, dict):\n"
        "        return ['type:<argumen>']\n"
        "    props, wajib = _skema(nama, tools)\n"
        "    errors = []\n"
        "    for k in wajib:\n"
        "        if k not in argumen:\n"
        "            errors.append(f'missing:{k}')\n"
        "    for k, v in argumen.items():\n"
        "        if k not in props:\n"
        "            errors.append(f'tak_dikenal:{k}')\n"
        "            continue\n"
        "        if not _tipe_ok(v, props[k]):\n"
        "            errors.append(f'type:{k}')\n"
        "    return errors\n"
    ),
    "konsumsi_stream": (
        "def konsumsi_stream(chunks):\n"
        "    potongan, jumlah, ttft, terakhir = [], 0, None, 0\n"
        "    for c in chunks:\n"
        "        terakhir = c['t_ms']\n"
        "        if c['delta']:\n"
        "            jumlah += 1\n"
        "            if ttft is None:\n"
        "                ttft = c['t_ms']\n"
        "            potongan.append(c['delta'])\n"
        "    return {'teks': ''.join(potongan), 'jumlah_chunk': jumlah,\n"
        "            'ttft_ms': ttft, 'total_ms': terakhir}\n"
    ),
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ctx = {"__name__": "__main__"}

for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "JAWABAN = {" in src and "None" in src:
        exec("JAWABAN = " + repr(ANSWERS["JAWABAN"]), ctx)
        print(f"[cell {i}] injected JAWABAN")
        continue
    if "TODO" in src:
        names = [n for n in ANSWERS if n != "JAWABAN" and f"def {n}(" in src]
        if not names:
            print(f"[cell {i}] SKIPPED (TODO tanpa fungsi dikenal)")
            continue
        for n in names:
            exec(ANSWERS[n], ctx)
            print(f"[cell {i}] injected {n}")
        continue
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, ctx)
    out = buf.getvalue()
    print(f"[cell {i}] executed ({len(out)} chars)")
    if "SKOR AKHIR" in out:
        print(out)
        m = re.search(r"SKOR AKHIR:\s*(\d+)/(\d+)", out)
        skor, total = int(m.group(1)), int(m.group(2))
        status = "PASS" if (skor == total == EXPECTED) else "FAIL"
        print(f">>> {status}: {skor}/{total} (expected {EXPECTED})")
        sys.exit(0 if status == "PASS" else 1)
