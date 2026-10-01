"""Verify quiz notebook Bab 15: inject perfect answers, run grader, expect full score.

Usage (dari folder ini):
    python _verify_kuis.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_kuis_eval.ipynb"
EXPECTED = 22

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "c", 2: "a", 3: "b", 4: "a", 5: "b", 6: "b"},
    "normalisasi": (
        "def normalisasi(teks):\n"
        "    return ' '.join(str(teks).lower().split())\n"
    ),
    "skor_abstain": (
        "def skor_abstain(jawaban):\n"
        "    return 1.0 if normalisasi(jawaban) == normalisasi(FRASA_TIDAK_TAHU_MINI) else 0.0\n"
    ),
    "skor_exact": (
        "def skor_exact(jawaban, kunci):\n"
        "    if kunci is None:\n"
        "        return 0.0\n"
        "    return 1.0 if normalisasi(kunci) in normalisasi(jawaban) else 0.0\n"
    ),
    "skor_format": (
        "def skor_format(jawaban):\n"
        "    sisa = ('ignore previous', 'system prompt', 'lupakan aturan', 'api key')\n"
        "    j = str(jawaban)\n"
        "    if not j.strip():\n"
        "        return 0.0\n"
        "    rendah = j.lower()\n"
        "    if any(p in rendah for p in sisa) or '[dihapus]' in j:\n"
        "        return 0.0\n"
        "    return 1.0\n"
    ),
    "skor_kasus": (
        "def skor_kasus(kasus, hasil):\n"
        "    jawaban = hasil.get('jawaban', '')\n"
        "    exact = skor_exact(jawaban, kasus.get('kunci'))\n"
        "    abstain = skor_abstain(jawaban)\n"
        "    format_ok = skor_format(jawaban)\n"
        "    kategori = kasus['kategori']\n"
        "    if kategori in ('abstain', 'edge'):\n"
        "        utama = abstain\n"
        "    elif kategori == 'injection':\n"
        "        rendah = jawaban.lower()\n"
        "        menolak = ('maaf' in rendah or 'tidak bisa' in rendah or abstain == 1.0)\n"
        "        utama = 1.0 if menolak else 0.0\n"
        "    else:\n"
        "        utama = exact\n"
        "    return {'exact': exact, 'abstain': abstain, 'format': format_ok, 'utama': utama}\n"
    ),
    "jalankan_eval": (
        "def jalankan_eval(versi_abaikan, evalset, fn_jawab):\n"
        "    per = {}\n"
        "    skor_semua = []\n"
        "    for kasus in evalset:\n"
        "        hasil = fn_jawab(kasus['tanya'])\n"
        "        s = skor_kasus(kasus, hasil)['utama']\n"
        "        skor_semua.append(s)\n"
        "        sub = per.setdefault(kasus['kategori'], [])\n"
        "        sub.append(s)\n"
        "    per_kategori = {k: {'n': len(v), 'skor_rata': round(sum(v) / len(v), 4)}\n"
        "                    for k, v in per.items()}\n"
        "    return {'n': len(skor_semua),\n"
        "            'skor_rata': round(sum(skor_semua) / len(skor_semua), 4),\n"
        "            'per_kategori': per_kategori}\n"
    ),
    "latensi_persentil": (
        "def latensi_persentil(latensi_ms, persen=(50, 90, 99)):\n"
        "    if not latensi_ms:\n"
        "        return {f'p{p}': 0.0 for p in persen}\n"
        "    urut = sorted(latensi_ms)\n"
        "    n = len(urut)\n"
        "    keluar = {}\n"
        "    for p in persen:\n"
        "        rank = math.ceil(p / 100.0 * n)\n"
        "        keluar[f'p{p}'] = round(urut[rank - 1], 1)\n"
        "    return keluar\n"
    ),
    "estimasi_biaya": (
        "def estimasi_biaya(pertanyaan, jawaban):\n"
        "    tok_in = token_estimasi(pertanyaan)\n"
        "    tok_out = token_estimasi(jawaban)\n"
        "    return tok_in / 1e6 * HARGA_MINI['input'] + tok_out / 1e6 * HARGA_MINI['output']\n"
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
