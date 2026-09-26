"""Verify project: patch tok_mini with reference impls, then run the test suite.

Usage (dari folder project-starter-tokenizer-mini):
    python _verify_project.py
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))
sys.path.insert(0, str(BASE.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import tok_mini_ref as ref  # noqa: E402
import tok_mini.attention as att_mod  # noqa: E402
import tok_mini.bpe as bpe_mod  # noqa: E402
import tok_mini.embedding as emb_mod  # noqa: E402
import tok_mini.lm as lm_mod  # noqa: E402
import tok_mini.position as pos_mod  # noqa: E402
import tok_mini.sampling as smp_mod  # noqa: E402
import tok_mini.tokenizer as tok_mod  # noqa: E402

for mod, names in [
    (bpe_mod, ["hitung_pasangan", "best_pair", "merge_pasangan", "latih_bpe", "apply_merges"]),
    (tok_mod, ["buat_vocab", "encode", "decode", "encode_teks", "decode_teks"]),
    (emb_mod, ["cos_sim", "cari_mirip", "analogi", "latih_embedding"]),
    (pos_mod, ["positional_encoding", "rpe"]),
    (att_mod, ["softmax_stabil", "self_attention"]),
    (lm_mod, ["loss_fn", "train_lm"]),
    (smp_mod, ["sample_greedy", "sample_topk", "sample_topp"]),
]:
    for name in names:
        setattr(mod, name, getattr(ref, name))

# fungsi tokenizer ref bergantung pada bpe ref — jaga konsistensi silang
import tok_mini.tokenizer  # noqa: E402,F811  (sudah di-patch di atas)

print("tok_mini di-patch dengan implementasi referensi.")

loader = unittest.TestLoader()
suite = loader.discover(str(BASE / "tests"))
buf = io.StringIO()
runner = unittest.TextTestRunner(stream=buf, verbosity=1)
result = runner.run(suite)
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
            print(out.getvalue()[-500:])
            sys.exit(1)
        tail = out.getvalue().rstrip().splitlines()
        if tail:
            print(f"[cell {i:>2}] {tail[-1][:100]}")
    print(f"{nb_name}: SEMUA CELL OK")

print("\nPROJECT VERIFY OK")
