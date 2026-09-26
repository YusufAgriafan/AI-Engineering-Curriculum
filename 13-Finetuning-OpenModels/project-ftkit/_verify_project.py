"""Verify project: patch ftkit with reference impls, run test suite, run notebooks.

Usage (dari folder project-ftkit):
    python _verify_project.py
"""
import io
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))
sys.path.insert(0, str(BASE.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ftkit_ref as ref  # noqa: E402
import ftkit.api as api_mod  # noqa: E402
import ftkit.evaluasi as eval_mod  # noqa: E402
import ftkit.lora as lora_mod  # noqa: E402
import ftkit.quantize as quant_mod  # noqa: E402
import ftkit.sft as sft_mod  # noqa: E402
import ftkit.trainer as trainer_mod  # noqa: E402

for mod, names in [
    (sft_mod, ["ke_contoh_sft", "muat_jsonl", "tulis_jsonl", "oversample", "split_sft"]),
    (lora_mod, ["buat_lora", "delta_w", "hitung_param", "rasio_param",
                "kontribusi_adapter", "gradient_check"]),
    (quant_mod, ["kuantisasi_simulasi", "laju_kompresi", "ukuran_model_gb",
                 "bandingkan_varian"]),
    (trainer_mod, ["epoch_satu", "latih", "dataset_ftkit"]),
    (api_mod, ["biaya_api", "biaya_bulanan", "token_estimasi", "keputusan_deployment"]),
    (eval_mod, ["prediksi", "evaluasi_sebelum_sesudah", "kebijakan_ood", "laporan_eval"]),
]:
    for name in names:
        setattr(mod, name, getattr(ref, name))

print("ftkit di-patch dengan implementasi referensi.")

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
