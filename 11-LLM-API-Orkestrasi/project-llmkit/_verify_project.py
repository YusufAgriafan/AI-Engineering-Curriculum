"""Verify project: patch llmkit with reference impls, run test suite, run notebooks.

Usage (dari folder project-llmkit):
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

import llmkit_ref as ref  # noqa: E402
import llmkit.cache as cache_mod  # noqa: E402
import llmkit.cost as cost_mod  # noqa: E402
import llmkit.fallback as fb_mod  # noqa: E402
import llmkit.orchestrate as orch_mod  # noqa: E402
import llmkit.retry as retry_mod  # noqa: E402
import llmkit.stream as stream_mod  # noqa: E402
import llmkit.tools as tools_mod  # noqa: E402

for mod, names in [
    (cost_mod, ["token_estimasi", "hitung_biaya", "biaya_hasil", "jalankan_anggaran"]),
    (retry_mod, ["backoff_ms", "panggil_dengan_retry"]),
    (fb_mod, ["panggil_dengan_fallback"]),
    (cache_mod, ["panggil_bercache"]),
    (stream_mod, ["konsumsi_stream"]),
    (tools_mod, ["skema_parameter", "skema_tools", "validasi_argumen",
                 "jalankan_tool", "loop_tool"]),
    (orch_mod, ["jalankan_pipeline"]),
]:
    for name in names:
        setattr(mod, name, getattr(ref, name))

# kelas (bukan sekadar fungsi): CachePrompt
cache_mod.CachePrompt = ref.CachePrompt
retry_mod.GalatSemuaPercobaan = ref.GalatSemuaPercobaan

print("llmkit di-patch dengan implementasi referensi.")

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
