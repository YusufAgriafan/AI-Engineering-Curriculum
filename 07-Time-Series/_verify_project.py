"""Verify project-starter-forecasting-mini: inject reference impls, run all tests + notebooks.

Usage (dari folder ini):
    python _verify_project.py

Student stubs di ts_mini/{windowing,baseline,ar,holtwinters}.py di-patch dengan
solusi/ts_mini_ref.py supaya seluruh test suite bisa dijalankan end-to-end,
lalu starter.ipynb & solusi.ipynb dieksekusi cell-per-cell (backend Agg).
"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent / "project-starter-forecasting-mini"

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))

import ts_mini_ref as ref  # noqa: E402

# ---- patch stub module namespaces dengan implementasi referensi ----
import ts_mini.ar as ar_mod  # noqa: E402
import ts_mini.baseline as base_mod  # noqa: E402
import ts_mini.holtwinters as hw_mod  # noqa: E402
import ts_mini.windowing as win_mod  # noqa: E402

for mod, names in [
    (win_mod, ["buat_window", "split_waktu", "buat_window_multi"]),
    (base_mod, ["mae", "rmse", "smape", "forecast_naive",
                "forecast_seasonal_naive", "forecast_ham"]),
    (ar_mod, ["fit_ar", "ar_forecast"]),
    (hw_mod, ["holt_winters_add", "holt_winters_mult"]),
]:
    for nama in names:
        setattr(mod, nama, getattr(ref, nama))

# ---- jalankan test suite ----
loader = unittest.TestLoader()
suite = loader.discover(str(BASE / "tests"))
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)

print("\n" + "=" * 60)
total = result.testsRun
failed = len(result.failures) + len(result.errors)
if failed:
    print(f"FAILED: {failed} test gagal dari {total}")
    sys.exit(1)
print(f"ALL {total} TESTS OK")

# ---- eksekusi notebook (starter & solusi) cell-per-cell ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")

for nb_name in ("starter.ipynb", "solusi.ipynb"):
    nb_path = BASE / nb_name
    cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
    ctx = {"__name__": "__main__", "__builtins__": __builtins__}
    print(f"\n--- eksekusi {nb_name} ---")
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, f"<{nb_name} cell {i}>", "exec"), ctx)
        except Exception as e:
            print(f"FAILED di {nb_name} cell {i}: {type(e).__name__}: {e}")
            print(buf.getvalue()[-500:])
            sys.exit(1)
        tail = buf.getvalue().rstrip().splitlines()
        if tail:
            print(f"[cell {i:>2}] {tail[-1][:100]}")
    print(f"{nb_name}: SEMUA CELL OK")

print("\nPROJECT VERIFY OK")
