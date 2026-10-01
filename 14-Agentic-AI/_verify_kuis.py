"""Verify quiz notebook Bab 14: inject perfect answers, run grader, expect full score.

Usage (dari folder ini):
    python _verify_kuis.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_kuis_agentic.ipynb"
EXPECTED = 22

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "b", 2: "b", 3: "c", 4: "c", 5: "b", 6: "b"},
    "format_langkah": (
        "def format_langkah(thought, action=None, observation=None):\n"
        "    baris = [f'Thought: {thought}']\n"
        "    if action is not None:\n"
        "        nama, args = action\n"
        "        baris.append(f'Action: {nama}({json.dumps(args, ensure_ascii=False)})')\n"
        "    if observation is not None:\n"
        "        baris.append(f'Observation: {observation}')\n"
        "    return '\\n'.join(baris)\n"
    ),
    "tool_cek_pesanan": (
        "def tool_cek_pesanan(invoice_id):\n"
        "    pesanan = PESANAN_KB.get(str(invoice_id).upper())\n"
        "    if pesanan is None:\n"
        "        return {'error': 'pesanan tidak ditemukan'}\n"
        "    keluar = dict(pesanan)\n"
        "    keluar.pop('user', None)\n"
        "    return keluar\n"
    ),
    "sanitasi": (
        "def sanitasi(teks):\n"
        "    bersih = str(teks)\n"
        "    for frasa in POLA_INJECTION:\n"
        "        pola = re.compile(re.escape(frasa), re.IGNORECASE)\n"
        "        bersih = pola.sub('[dihapus]', bersih)\n"
        "    bersih = bersih.replace('\\n', ' ')\n"
        "    return bersih.strip()\n"
    ),
    "MemoriSesi": (
        "class MemoriSesi:\n"
        "    def __init__(self, maks_giliran=4):\n"
        "        self.maks_giliran = maks_giliran\n"
        "        self.riwayat = []\n"
        "        self.cache = {}\n"
        "        self.cache_hit = 0\n"
        "        self.cache_miss = 0\n"
        "\n"
        "    def tambah(self, peran, teks):\n"
        "        giliran = {'peran': peran, 'teks': teks, 'token': token_estimasi(teks)}\n"
        "        self.riwayat.append(giliran)\n"
        "        return giliran\n"
        "\n"
        "    def konteks(self):\n"
        "        return list(self.riwayat[-self.maks_giliran:])\n"
        "\n"
        "    def cache_get(self, kueri):\n"
        "        nilai = self.cache.get(str(kueri).strip().lower())\n"
        "        if nilai is None:\n"
        "            self.cache_miss += 1\n"
        "        else:\n"
        "            self.cache_hit += 1\n"
        "        return nilai\n"
        "\n"
        "    def cache_put(self, kueri, nilai):\n"
        "        self.cache[str(kueri).strip().lower()] = nilai\n"
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
        names = [n for n in ANSWERS if n != "JAWABAN"
                 and (f"def {n}(" in src or f"class {n}" in src)]
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
