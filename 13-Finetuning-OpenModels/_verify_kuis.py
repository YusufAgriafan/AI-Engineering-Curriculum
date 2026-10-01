"""Verify quiz notebook Bab 13: inject perfect answers, run grader, expect full score.

Usage (dari folder ini):
    python _verify_kuis.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_kuis_finetuning.ipynb"
EXPECTED = 22

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "b", 2: "c", 3: "c", 4: "d", 5: "c", 6: "b"},
    "ke_contoh_sft": (
        "def ke_contoh_sft(baris):\n"
        "    return (f\"{baris['instruksi']}\\n\\nInput: {baris['input']}\\n\\nJawaban:\",\n"
        "            baris[\"respons\"])\n"
    ),
    "oversample": (
        "def oversample(baris_list, per_label=4):\n"
        "    hasil = list(baris_list)\n"
        "    per_kelas = {}\n"
        "    for b in baris_list:\n"
        "        per_kelas.setdefault(b[\"respons\"], []).append(b)\n"
        "    for label, kumpulan in per_kelas.items():\n"
        "        i = 0\n"
        "        while sum(1 for b in hasil if b[\"respons\"] == label) < per_label:\n"
        "            hasil.append(kumpulan[i % len(kumpulan)])\n"
        "            i += 1\n"
        "    return hasil\n"
    ),
    "kuantisasi_simulasi": (
        "def kuantisasi_simulasi(bobot, bit):\n"
        "    if bit >= 32:\n"
        "        return list(bobot)\n"
        "    levels = 2 ** (bit - 1) - 1\n"
        "    skala = max(abs(w) for w in bobot) / levels if levels else 0.0\n"
        "    if skala == 0.0:\n"
        "        return list(bobot)\n"
        "    keluar = []\n"
        "    for w in bobot:\n"
        "        q = int(round(w / skala))\n"
        "        q = max(-levels, min(levels, q))\n"
        "        keluar.append(q * skala)\n"
        "    return keluar\n"
    ),
    "laju_kompresi": (
        "def laju_kompresi(bit):\n"
        "    return 32.0 / bit\n"
    ),
    "ukuran_model_gb": (
        "def ukuran_model_gb(param_miliar, bit):\n"
        "    return param_miliar * bit / 8.0\n"
    ),
    "kebijakan_ood": (
        "def kebijakan_ood(hasil_pred, skor_min=0.6, jawaban_aman='kurang yakin, mohon periksa'):\n"
        "    if hasil_pred['yakin'] < skor_min:\n"
        "        return {**hasil_pred, 'tampilkan': False, 'aksi': jawaban_aman}\n"
        "    return {**hasil_pred, 'tampilkan': True, 'aksi': 'tampilkan label'}\n"
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
