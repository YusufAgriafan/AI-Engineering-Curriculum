"""Verify 01_lab_computer_vision.ipynb: exec every cell; TODO cells get reference impls."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

NB = Path(sys.argv[1])
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "konvolusi_manual": (
        "def konvolusi_manual(img, kernel):\n"
        "    ih, iw = img.shape\n"
        "    kh, kw = kernel.shape\n"
        "    out = np.zeros((ih - kh + 1, iw - kw + 1))\n"
        "    for i in range(out.shape[0]):\n"
        "        for j in range(out.shape[1]):\n"
        "            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)\n"
        "    return out\n"
    ),
    "pad_zero": (
        "def pad_zero(img, pad):\n"
        "    if pad == 0:\n"
        "        return np.asarray(img, dtype=float).copy()\n"
        "    return np.pad(np.asarray(img, dtype=float), pad, mode='constant')\n"
    ),
    "ukuran_output": (
        "def ukuran_output(H, k, pad=0, stride=1):\n"
        "    return (H + 2 * pad - k) // stride + 1\n"
    ),
    "conv2d_forward": (
        "def conv2d_forward(img, K, pad=0, stride=1):\n"
        "    xp = pad_zero(img, pad)\n"
        "    win = sliding_window_view(xp, K.shape)[::stride, ::stride]\n"
        "    return np.tensordot(win, K, axes=([2, 3], [0, 1]))\n"
    ),
    "conv2d_backward": (
        "def conv2d_backward(dOut, img, K, pad=0, stride=1):\n"
        "    img = np.asarray(img, dtype=float)\n"
        "    H, W = img.shape\n"
        "    Ho, Wo = dOut.shape\n"
        "    # --- dK: tiap posisi jendela berkontribusi ke elemen K yang sama ---\n"
        "    xp = pad_zero(img, pad)\n"
        "    winK = sliding_window_view(xp, K.shape)[::stride, ::stride]\n"
        "    dK = np.tensordot(dOut, winK, axes=([0, 1], [0, 1]))\n"
        "    # --- dX: sebar dOut ke kanvas (dilasi stride), lalu full-conv dgn K dibalik ---\n"
        "    dil = np.zeros(((Ho - 1) * stride + 1, (Wo - 1) * stride + 1))\n"
        "    dil[::stride, ::stride] = dOut\n"
        "    Kf = K[::-1, ::-1]\n"
        "    d = np.pad(dil, [(K.shape[0] - 1, K.shape[0] - 1),\n"
        "                     (K.shape[1] - 1, K.shape[1] - 1)])\n"
        "    dX_full = np.tensordot(sliding_window_view(d, K.shape), Kf,\n"
        "                           axes=([2, 3], [0, 1]))\n"
        "    dX = dX_full[pad:pad + H, pad:pad + W]\n"
        "    return dK, dX\n"
    ),
    "maxpool2x2_forward": (
        "def maxpool2x2_forward(x):\n"
        "    H, W = x.shape\n"
        "    return x.reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))\n"
    ),
    "maxpool2x2_backward": (
        "def maxpool2x2_backward(dOut, x):\n"
        "    H, W = x.shape\n"
        "    dX = np.zeros_like(x, dtype=float)\n"
        "    for i in range(H // 2):\n"
        "        for j in range(W // 2):\n"
        "            win = x[2 * i:2 * i + 2, 2 * j:2 * j + 2]\n"
        "            idx = np.unravel_index(np.argmax(win), win.shape)\n"
        "            dX[2 * i + idx[0], 2 * j + idx[1]] += dOut[i, j]\n"
        "    return dX\n"
    ),
    "flip_horizontal": "def flip_horizontal(img):\n    return np.asarray(img)[:, ::-1].copy()\n",
    "flip_vertical": "def flip_vertical(img):\n    return np.asarray(img)[::-1, :].copy()\n",
    "random_noise": (
        "def random_noise(img, rng, std=0.1):\n"
        "    out = np.asarray(img, dtype=float).copy()\n"
        "    out = out + rng.normal(0, std, out.shape)\n"
        "    return out\n"
    ),
    "random_shift": (
        "def random_shift(img, rng, max_shift=1):\n"
        "    img = np.asarray(img, dtype=float)\n"
        "    H, W = img.shape\n"
        "    dy = int(rng.integers(-max_shift, max_shift + 1))\n"
        "    dx = int(rng.integers(-max_shift, max_shift + 1))\n"
        "    out = np.zeros_like(img)\n"
        "    ys = slice(max(0, -dy), min(H, H - dy))\n"
        "    xs = slice(max(0, -dx), min(W, W - dx))\n"
        "    yd = slice(max(0, dy), min(H, H + dy))\n"
        "    xd = slice(max(0, dx), min(W, W + dx))\n"
        "    out[yd, xd] = img[ys, xs]\n"
        "    return out\n"
    ),
    "ReLU": (
        "class ReLU:\n"
        "    def forward(self, x):\n"
        "        self.mask = x > 0\n"
        "        return np.maximum(x, 0)\n"
        "    def backward(self, dout):\n"
        "        return dout * self.mask\n"
        "    def step(self):\n"
        "        pass\n"
    ),
    "Pool2x2": (
        "class Pool2x2:\n"
        "    def forward(self, X):\n"
        "        self.X = X\n"
        "        n, H, W = X.shape\n"
        "        return X.reshape(n, H // 2, 2, W // 2, 2).max(axis=(2, 4))\n"
        "    def backward(self, dOut):\n"
        "        dX = np.zeros_like(self.X, dtype=float)\n"
        "        for s in range(self.X.shape[0]):\n"
        "            dX[s] = maxpool2x2_backward(dOut[s], self.X[s])\n"
        "        return dX\n"
        "    def step(self):\n"
        "        pass\n"
    ),
    "Flatten": (
        "class Flatten:\n"
        "    def forward(self, x):\n"
        "        self.shape = x.shape\n"
        "        return x.reshape(x.shape[0], -1)\n"
        "    def backward(self, dout):\n"
        "        return dout.reshape(self.shape)\n"
        "    def step(self):\n"
        "        pass\n"
    ),
    "ConvLayer": (
        "class ConvLayer:\n"
        "    def __init__(self, ksize, rng, lr=0.05, pad=0):\n"
        "        self.K = rng.normal(0, np.sqrt(2.0 / (ksize * ksize)), (ksize, ksize))\n"
        "        self.lr = lr; self.pad = pad\n"
        "    def forward(self, X):\n"
        "        self.X = X\n"
        "        return np.stack([conv2d_forward(x, self.K, pad=self.pad) for x in X])\n"
        "    def backward(self, dOut):\n"
        "        self.dK = np.zeros_like(self.K)\n"
        "        dX = np.empty_like(self.X)\n"
        "        for s in range(self.X.shape[0]):\n"
        "            dKs, dXs = conv2d_backward(dOut[s], self.X[s], self.K, pad=self.pad)\n"
        "            self.dK += dKs\n"
        "            dX[s] = dXs\n"
        "        return dX\n"
        "    def step(self):\n"
        "        self.K -= self.lr * self.dK\n"
    ),
    "cek_early_stop": (
        "def cek_early_stop(val_loss, patience):\n"
        "    best = np.inf; best_epoch = 0; wait = 0; stop = len(val_loss) - 1\n"
        "    for i, v in enumerate(val_loss):\n"
        "        if v < best - 1e-6:\n"
        "            best = v; best_epoch = i; wait = 0\n"
        "        else:\n"
        "            wait += 1\n"
        "            if wait >= patience:\n"
        "                stop = i\n"
        "                break\n"
        "    return (best_epoch, stop)\n"
    ),
}

ctx = {"__name__": "__main__"}
# Sel TODO yang di-skip bisa memuat import penting — sediakan di ctx sejak awal.
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

ctx["np"] = np
ctx["sliding_window_view"] = sliding_window_view
errors = []
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src and "raise NotImplementedError" in src:
        injected = False
        for name, refsrc in REF.items():
            if f"def {name}(" in src or f"class {name}" in src:
                try:
                    exec(refsrc, ctx)
                    print(f"[cell {i}] injected ref: {name}")
                except Exception:
                    errors.append((i, f"inject {name}", traceback.format_exc()))
                injected = True
        if not injected:
            errors.append((i, "todo-cell-without-ref", src[:80]))
        continue
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ctx)
        print(f"[cell {i}] ok")
    except Exception:
        errors.append((i, "exec", traceback.format_exc()))

print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for i, kind, tb in errors:
        print(f"\n--- cell {i} ({kind}) ---\n{tb}")
    sys.exit(1)
print("ALL CELLS OK")
