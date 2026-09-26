"""Verify quiz notebook: inject perfect answers, run grader, expect full score.

Usage (dari folder ini):
    python _verify_quiz.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_kuis_llm_fondasi.ipynb"
EXPECTED = 22

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "c", 2: "b", 3: "b", 4: "b", 5: "b", 6: "b"},
    "softmax_stabil": (
        "def softmax_stabil(x, axis=-1):\n"
        "    x = np.asarray(x, dtype=float)\n"
        "    z = x - np.max(x, axis=axis, keepdims=True)\n"
        "    e = np.exp(z)\n"
        "    return e / np.sum(e, axis=axis, keepdims=True)\n"
    ),
    "attention": (
        "def attention(Q, K, V):\n"
        "    d_k = Q.shape[-1]\n"
        "    scores = Q @ K.T / np.sqrt(d_k)\n"
        "    attn = softmax_stabil(scores, axis=-1)\n"
        "    return attn @ V, attn\n"
    ),
    "apply_causal_mask": (
        "def apply_causal_mask(scores):\n"
        "    scores = np.asarray(scores, dtype=float)\n"
        "    n = scores.shape[0]\n"
        "    mask = np.triu(np.ones((n, n)), k=1)\n"
        "    return scores + mask * -1e9\n"
    ),
    "sample_topk": (
        "def sample_topk(logits, k, rng):\n"
        "    logits = np.asarray(logits, dtype=float)\n"
        "    idx = np.argsort(logits)[::-1][:k]\n"
        "    p = softmax_stabil(logits[idx])\n"
        "    return int(rng.choice(idx, p=p))\n"
    ),
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ctx = {"__name__": "__main__"}
import numpy as np  # noqa: E402  -- implementasi referensi memakai np.*

ctx["np"] = np

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
        # cell TODO asli TIDAK dieksekusi; timpa dengan implementasi referensi
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
