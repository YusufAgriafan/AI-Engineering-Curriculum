"""Verify project: patch evalkit with reference impls, run test suite, run notebooks.

Usage (dari folder project-evalkit):
    python _verify_project.py
"""
import io
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import evalkit_ref as ref  # noqa: E402
import evalkit.evaluasi as eval_mod  # noqa: E402
import evalkit.evalset as es_mod  # noqa: E402
import evalkit.judge as judge_mod  # noqa: E402
import evalkit.monitor as mon_mod  # noqa: E402
import evalkit.regression as reg_mod  # noqa: E402
import evalkit.runner as run_mod  # noqa: E402
import evalkit.scorers as scorer_mod  # noqa: E402

for mod, names in [
    (es_mod, ["validasi_evalset", "hitung_kategori", "split_dev_test"]),
    (scorer_mod, ["normalisasi", "skor_exact", "skor_abstain", "skor_format",
                  "skor_kasus"]),
    (run_mod, ["jalankan_satu", "jalankan_eval", "agregat"]),
    (reg_mod, ["bandingkan", "gate_rilis"]),
    (mon_mod, ["estimasi_biaya", "biaya_run", "latensi_persentil"]),
    (judge_mod, ["nilai_jawaban", "sinyal_kasus", "skor_sinyal", "skor_total",
                 "bias_panjang", "kalibrasi"]),
    (eval_mod, ["evaluasi_versi", "bandingkan_versi", "analisis_error",
                "antrean_feedback", "laporan_lengkap"]),
]:
    for name in names:
        setattr(mod, name, getattr(ref, name))

print("evalkit di-patch dengan implementasi referensi.")

loader = unittest.TestLoader()
suite = loader.discover(str(BASE / "tests"))
buf = io.StringIO()
result = unittest.TextTestRunner(stream=buf, verbosity=1).run(suite)
print(buf.getvalue())

total = result.testsRun
failed = len(result.failures) + len(result.errors)
if failed:
    print(f">>> FAIL: {failed} test gagal dari {total}")
    sys.exit(1)
print(f">>> ALL {total} TESTS OK")

# ---- eksekusi notebook (starter & solusi) cell-per-cell ----
import contextlib  # noqa: E402
import json  # noqa: E402

for nb_name in ("starter.ipynb", "solusi.ipynb"):
    nb_path = BASE / nb_name
    if not nb_path.exists():
        print(f"(lewati {nb_name} — belum ada)")
        continue
    cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
    ctx = {"__name__": "__main__", "__builtins__": __builtins__}
    print(f"\n--- eksekusi {nb_name} ---")
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(src, f"<{nb_name} cell {i}>", "exec"), ctx)
        except Exception as e:
            print(f"FAILED di {nb_name} cell {i}: {type(e).__name__}: {e}")
            print(out.getvalue()[-600:])
            sys.exit(1)
        tail = out.getvalue().rstrip().splitlines()
        if tail:
            print(f"[cell {i:>2}] {tail[-1][:100]}")
    print(f"{nb_name}: SEMUA CELL OK")

print("\nPROJECT VERIFY OK")
