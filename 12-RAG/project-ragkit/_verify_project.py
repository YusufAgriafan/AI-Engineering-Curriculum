"""Verify project: patch ragkit with reference impls, run test suite, run notebooks.

Usage (dari folder project-ragkit):
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

import ragkit_ref as ref  # noqa: E402
import ragkit.abstain as abstain_mod  # noqa: E402
import ragkit.cache as cache_mod  # noqa: E402
import ragkit.evaluate as eval_mod  # noqa: E402
import ragkit.hybrid as hybrid_mod  # noqa: E402
import ragkit.pipeline as pipe_mod  # noqa: E402
import ragkit.rerank as rerank_mod  # noqa: E402
import ragkit.retriever as retr_mod  # noqa: E402
import ragkit.vectorstore as store_mod  # noqa: E402

for mod, names in [
    (store_mod, ["VectorStore"]),
    (retr_mod, ["buat_retriever", "evaluasi_retrieval"]),
    (abstain_mod, ["abstaikan"]),
    (cache_mod, ["CacheRetrieval", "retrieve_bercache", "tolok_ukur"]),
    (hybrid_mod, ["cari_bm25", "gabung_rrf", "cari_hybrid"]),
    (rerank_mod, ["skorer_kata", "rerank"]),
    (pipe_mod, ["_latensi_index", "jalankan_rag"]),
    (eval_mod, ["evaluasi_ujung_ke_ujung"]),
]:
    for name in names:
        setattr(mod, name, getattr(ref, name))

print("ragkit di-patch dengan implementasi referensi.")

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
