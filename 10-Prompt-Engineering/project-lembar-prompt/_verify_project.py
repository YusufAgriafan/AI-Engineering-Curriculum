"""Verify project: patch plib with reference impls, run test suite, run notebooks.

Usage (dari folder project-lembar-prompt):
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

import plib_ref as ref  # noqa: E402
import plib.evaluate as eval_mod  # noqa: E402
import plib.fewshot as few_mod  # noqa: E402
import plib.guard as guard_mod  # noqa: E402
import plib.parse as parse_mod  # noqa: E402
import plib.schema as schema_mod  # noqa: E402
import plib.sections as sections_mod  # noqa: E402
import plib.template as tmpl_mod  # noqa: E402

for mod, names in [
    (sections_mod, ["bagian_ditemukan", "bagian_hilang", "prompt_lengkap",
                    "build_prompt", "estimasi_token", "potong_budget"]),
    (tmpl_mod, ["placeholder", "render", "render_aman", "escape_delimiter"]),
    (few_mod, ["format_contoh", "pilih_contoh", "urutkan_contoh", "bangun_fewshot"]),
    (schema_mod, ["validate", "valid"]),
    (parse_mod, ["perbaiki_json", "ekstrak_json", "parse_output", "naive_json_ok"]),
    (guard_mod, ["deteksi_injection", "bungkus_data", "langgar_leash"]),
    (eval_mod, ["jalankan_eval", "bandingkan"]),
]:
    for name in names:
        setattr(mod, name, getattr(ref, name))

print("plib di-patch dengan implementasi referensi.")

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

import matplotlib  # noqa: E402

matplotlib.use("Agg")

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
