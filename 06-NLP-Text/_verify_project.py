"""Verify project-starter-lm-mini: inject reference impls, run all tests + notebooks.

Usage (dari root repo):
    python AI-Engineering-Curriculum/06-NLP-Text/_verify_project.py

Student stubs di lm_mini/{text,embeddings,languagemodel,rnn}.py di-patch dengan
solusi/lm_mini_ref.py supaya seluruh test suite bisa dijalankan end-to-end,
lalu starter.ipynb & solusi.ipynb dieksekusi cell-per-cell (backend Agg).
"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent / "project-starter-lm-mini"

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))

import lm_mini_ref as ref  # noqa: E402

# ---- patch stub module namespaces dengan implementasi referensi ----
import lm_mini.text as text_mod  # noqa: E402
import lm_mini.embeddings as emb_mod  # noqa: E402
import lm_mini.languagemodel as lm_mod  # noqa: E402
import lm_mini.rnn as rnn_mod  # noqa: E402

text_mod.PAD_ID = ref.PAD_ID
text_mod.OOV_ID = ref.OOV_ID
text_mod.tokenize_kata = ref.tokenize_kata
text_mod.bangun_vocab = ref.bangun_vocab
text_mod.encode = ref.encode
text_mod.encode_aman = ref.encode_aman
text_mod.pad_sequences = ref.pad_sequences

emb_mod.sigmoid = ref.sigmoid
emb_mod.pasangan_skipgram = ref.pasangan_skipgram
emb_mod.frekuensi_unigram = ref.frekuensi_unigram
emb_mod.train_skipgram = ref.train_skipgram

lm_mod.bigram_matrix = ref.bigram_matrix
lm_mod.ee_bigram = ref.ee_bigram
lm_mod.prob_baris_dengan_temp = ref.prob_baris_dengan_temp
lm_mod.sample_bigram = ref.sample_bigram

rnn_mod.onehot_batch = ref.onehot_batch
rnn_mod.softmax = ref.softmax
rnn_mod.rnn_step_forward = ref.rnn_step_forward
rnn_mod.rnn_step_backward = ref.rnn_step_backward
rnn_mod.rnn_forward = ref.rnn_forward
rnn_mod.rnn_backward = ref.rnn_backward

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
