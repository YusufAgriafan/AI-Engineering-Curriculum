"""Verify quiz notebook: inject perfect answers, run grader, expect full score.

Usage: python _verify_quiz.py <nb.ipynb> <total_score>
"""
import json
import sys
from pathlib import Path

NB = Path(sys.argv[1])
EXPECTED = int(sys.argv[2])
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "b", 2: "b", 3: "c", 4: "b"},
    "conv_valid": (
        "def conv_valid(img, K):\n"
        "    img = np.asarray(img, dtype=float); K = np.asarray(K, dtype=float)\n"
        "    H, W = img.shape; kh, kw = K.shape\n"
        "    out = np.zeros((H - kh + 1, W - kw + 1))\n"
        "    for i in range(out.shape[0]):\n"
        "        for j in range(out.shape[1]):\n"
        "            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * K)\n"
        "    return out\n"
    ),
    "ukuran_output": (
        "def ukuran_output(H, k, pad=0, stride=1):\n"
        "    return (H + 2 * pad - k) // stride + 1\n"
    ),
    "maxpool2x2": (
        "def maxpool2x2(x):\n"
        "    x = np.asarray(x, dtype=float)\n"
        "    H, W = x.shape\n"
        "    return x.reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))\n"
    ),
    "maxpool2x2_backward": (
        "def maxpool2x2_backward(dOut, x):\n"
        "    x = np.asarray(x, dtype=float)\n"
        "    H, W = x.shape\n"
        "    dX = np.zeros_like(x)\n"
        "    for i in range(H // 2):\n"
        "        for j in range(W // 2):\n"
        "            win = x[2 * i:2 * i + 2, 2 * j:2 * j + 2]\n"
        "            idx = np.unravel_index(np.argmax(win), win.shape)\n"
        "            dX[2 * i + idx[0], 2 * j + idx[1]] += dOut[i, j]\n"
        "    return dX\n"
    ),
    "shift_zero": (
        "def shift_zero(img, dy, dx):\n"
        "    img = np.asarray(img, dtype=float)\n"
        "    H, W = img.shape\n"
        "    out = np.zeros_like(img)\n"
        "    ys, xs = slice(max(0, -dy), min(H, H - dy)), slice(max(0, -dx), min(W, W - dx))\n"
        "    yd, xd = slice(max(0, dy), min(H, H + dy)), slice(max(0, dx), min(W, W + dx))\n"
        "    out[yd, xd] = img[ys, xs]\n"
        "    return out\n"
    ),
    "filter_tepi_v": (
        "def filter_tepi_v():\n"
        "    return np.array([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])\n"
    ),
    "tepi_v_score": (
        "def tepi_v_score(img):\n"
        "    return conv_valid(img, filter_tepi_v())\n"
    ),
}

ctx = {"__name__": "__main__"}
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
        for n in names:
            exec(ANSWERS[n], ctx)
        print(f"[cell {i}] injected: {names}")
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ctx)
    print(f"[cell {i}] ok")

print("\nEXPECTED SCORE:", EXPECTED)
