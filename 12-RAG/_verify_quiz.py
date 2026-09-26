"""Verify quiz notebook Bab 12: inject perfect answers, run grader, expect full score.

Usage (dari folder ini):
    python _verify_quiz.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_kuis_rag.ipynb"
EXPECTED = 22

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "b", 2: "c", 3: "b", 4: "c", 5: "a", 6: "b"},
    "cosine": (
        "def cosine(a, b):\n"
        "    dot = sum(x * y for x, y in zip(a, b))\n"
        "    na = math.sqrt(sum(x * x for x in a))\n"
        "    nb = math.sqrt(sum(y * y for y in b))\n"
        "    if na == 0.0 or nb == 0.0:\n"
        "        return 0.0\n"
        "    return dot / (na * nb)\n"
    ),
    "cari_vektor": (
        "def cari_vektor(kueri, top_k=2):\n"
        "    skor = []\n"
        "    for i, c in enumerate(CHUNKS):\n"
        "        skor.append((cosine(vektor(kueri), vektor(c['text'])), i))\n"
        "    urut = sorted(skor, key=lambda t: (-t[0], t[1]))[:top_k]\n"
        "    return [{'chunk': CHUNKS[i], 'score': s, 'index': i} for s, i in urut]\n"
    ),
    "gabung_rrf": (
        "def gabung_rrf(daftar_dapat, k=60):\n"
        "    skor = {}\n"
        "    for dapat in daftar_dapat:\n"
        "        for rank, item in enumerate(dapat):\n"
        "            idx = item['chunk']['chunk_index']\n"
        "            skor[idx] = skor.get(idx, 0.0) + 1.0 / (k + rank + 1)\n"
        "    return skor\n"
    ),
    "abstaikan": (
        "def abstaikan(hasil, ambang=0.30):\n"
        "    skor = hasil.get('skor_terbaik', 0.0)\n"
        "    if hasil.get('abstain') or skor >= ambang:\n"
        "        return dict(hasil)\n"
        "    keluar = dict(hasil)\n"
        "    keluar['jawaban'] = SEP_TIDAK_ADA\n"
        "    keluar['abstain'] = True\n"
        "    keluar['abstaikan_oleh'] = skor\n"
        "    return keluar\n"
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
