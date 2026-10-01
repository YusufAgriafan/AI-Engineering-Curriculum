"""Verify project: patch agentkit with reference impls, run test suite, run notebooks.

Usage (dari folder project-agentkit):
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

import agentkit_ref as ref  # noqa: E402
import agentkit.agent as agent_mod  # noqa: E402
import agentkit.guardrail as guard_mod  # noqa: E402
import agentkit.loop_react as loop_mod  # noqa: E402
import agentkit.memory as mem_mod  # noqa: E402
import agentkit.tools as tools_mod  # noqa: E402

for mod, names in [
    (tools_mod, ["validasi_skema", "tool_cari_dokumen", "tool_cek_pesanan",
                 "tool_refund", "REGISTRY", "eksekusi_tool"]),
    (loop_mod, ["format_langkah", "format_trace", "jalankan_langkah", "jalankan_rencana"]),
    (guard_mod, ["butuh_konfirmasi", "sanitasi", "amankan_observasi"]),
    (mem_mod, ["MemoriSesi", "hitung_biaya"]),
    (agent_mod, ["jawab", "planner_bertahap", "evaluasi_trace", "verifikasi_tools"]),
]:
    for name in names:
        setattr(mod, name, getattr(ref, name))

print("agentkit di-patch dengan implementasi referensi.")

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
