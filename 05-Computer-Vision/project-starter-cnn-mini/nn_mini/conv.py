"""Konvolusi, max-pool, dan flatten dari nol — forward + backward.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_conv.py.

Matematika (satu filter 2D, satu kanal — cukup untuk dataset "bars"):

    forward konvolusi (valid, stride s, input sudah di-pad):
        out[i, r, c] = Σ_{u,v} Xp[i, r*s+u, c*s+v] * K[u, v]

    backward konvolusi:
        dK[u, v]    = Σ_{i,r,c} grad[i, r, c] * Xp[i, r*s+u, c*s+v]
        dXp[i, rs+u, cs+v] += grad[i, r, c] * K[u, v]      (akumulasi, BUKAN assign)

    forward max-pool 2x2 stride 2:
        out[i, r, c] = max(X[i, 2r:2r+2, 2c:2c+2])
        backward: gradien hanya mengalir ke posisi yang jadi max (argmax).

Aturan emas: bentuk yang keluar dari forward harus bisa diterima backward
dengan gradien BENTUK SAMA seperti output forward-nya.
"""

import numpy as np

__all__ = ["pad_zero", "conv2d_forward", "conv2d_backward",
           "maxpool2x2_forward", "maxpool2x2_backward", "Flatten", "Conv2D"]


# ------------------------------------------------------------------ padding
def pad_zero(X, pad):
    """Zero-padding simetris: (n, h, w) dengan pad=p → (n, h+2p, w+2p).

    pad = 0 → return salinan X (jangan referensi yang sama).
    pad < 0 → ValueError.
    """
    if pad < 0:
        raise ValueError("pad tidak boleh negatif")
    if pad == 0:
        return np.asarray(X, dtype=float).copy()
    # TODO: np.pad X dengan ((0,0),(pad,pad),(pad,pad)) mode 'constant' nilai 0
    raise NotImplementedError


# --------------------------------------------------------------- konvolusi
def conv2d_forward(X, K, stride=1):
    """Konvolusi 2D valid (input dianggap sudah di-pad oleh pemanggil).

    X: (n, h, w) ; K: (kh, kw) ; stride: int ≥ 1
    → (n, (h-kh)//stride + 1, (w-kw)//stride + 1)
    """
    X = np.asarray(X, dtype=float)
    K = np.asarray(K, dtype=float)
    if X.ndim != 3:
        raise ValueError("X harus (n, h, w)")
    if K.ndim != 2:
        raise ValueError("K harus (kh, kw)")
    if stride < 1:
        raise ValueError("stride minimal 1")
    n, h, w = X.shape
    kh, kw = K.shape
    if kh > h or kw > w:
        raise ValueError("kernel lebih besar dari input")
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((n, oh, ow))
    for u in range(kh):
        for v in range(kw):
            # TODO: out[:, r, c] += X[:, r*s+u, c*s+v] * K[u, v]
            #       (loop r, c di dalam — atau slicing dengan langkah stride)
            raise NotImplementedError
    return out


def conv2d_backward(Xp, K, grad, stride=1):
    """Backward konvolusi valid.

    Xp  : (n, h, w) input ke forward (SUDAH di-pad)
    K   : (kh, kw)
    grad: (n, oh, ow) dL/dout — bentuk output forward
    → (dXp, dK) dengan dXp bentuk sama seperti Xp.
    """
    Xp = np.asarray(Xp, dtype=float)
    K = np.asarray(K, dtype=float)
    grad = np.asarray(grad, dtype=float)
    n, h, w = Xp.shape
    kh, kw = K.shape
    oh, ow = grad.shape[1], grad.shape[2]
    dXp = np.zeros_like(Xp)
    dK = np.zeros_like(K)
    for r in range(oh):
        for c in range(ow):
            g = grad[:, r, c]                      # (n,)
            potong = Xp[:, r * stride:r * stride + kh, c * stride:c * stride + kw]
            # TODO 1: dK += rata-rata tertimbang g per sampel:
            #         dK += (g[:, None, None] * potong).sum(axis=0)
            # TODO 2: dXp[:, rs:rs+kh, cs:cs+kw] += g[:, None, None] * K[None, :, :]
            raise NotImplementedError
    return dXp, dK


# ---------------------------------------------------------------- max-pool
def maxpool2x2_forward(X):
    """Max-pool 2x2 stride 2: (n, h, w) → (n, h//2, w//2).

    h/w ganil? ValueError (input CNN mini selalu kelipatan 2).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X harus (n, h, w)")
    n, h, w = X.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("h dan w harus genap untuk max-pool 2x2")
    out = np.zeros((n, h // 2, w // 2))
    # TODO: isi out — cara apa pun yang benar; dua opsi:
    #   (a) loop kecil atas r, c + np.max(X[:, 2r:2r+2, 2c:2c+2], axis=(1,2))
    #   (b) reshape (n, h//2, 2, w//2, 2) → max(axis=(2, 4))
    raise NotImplementedError


def maxpool2x2_backward(X, grad):
    """Backward max-pool: gradien hanya ke posisi argmax tiap jendela 2x2.

    X   : (n, h, w) input forward
    grad: (n, h//2, w//2)
    → dX bentuk (n, h, w), nol kecuali posisi max tiap jendela.
    """
    X = np.asarray(X, dtype=float)
    grad = np.asarray(grad, dtype=float)
    n, h, w = X.shape
    dX = np.zeros_like(X)
    for r in range(h // 2):
        for c in range(w // 2):
            jendela = X[:, 2 * r:2 * r + 2, 2 * c:2 * c + 2]        # (n, 2, 2)
            # TODO: cari argmax tiap sampel (reshape (n, 4), argmax axis=1),
            #       dX[:, 2r+ar, 2c+ac] = grad[:, r, c]
            raise NotImplementedError
    return dX


# ----------------------------------------------------------------- flatten
class Flatten:
    """(n, h, w) → (n, h*w). Backward mengembalikan bentuk asli."""

    def __init__(self):
        self._bentuk = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("Flatten butuh (n, h, w)")
        self._bentuk = X.shape
        # TODO: return X.reshape(n, -1)
        raise NotImplementedError

    def backward(self, grad):
        # TODO: return grad.reshape(self._bentuk)
        raise NotImplementedError


# ---------------------------------------------------- Conv2D (rakit sendiri)
class Conv2D:
    """Layer konvolusi: menyimpan filter K, pakai pad_zero + conv2d_forward/backward.

    Lazy init seperti Dense Bab 4: K dibuat saat forward pertama
    (rng.normal(0, sqrt(2/(kh*kw)), size=K.shape) — varian He untuk konvolusi).
    """

    def __init__(self, kh, kw, pad=1, stride=1, seed=None):
        if kh < 1 or kw < 1:
            raise ValueError("ukuran kernel minimal 1x1")
        self.kh, self.kw = int(kh), int(kw)
        self.pad = int(pad)
        self.stride = int(stride)
        self._rng = np.random.default_rng(seed)
        self.K = None
        self.dK = None

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("input Conv2D harus (n, h, w)")
        if self.K is None:
            # TODO: self.K = self._rng.normal(0, np.sqrt(2.0 / (self.kh * self.kw)),
            #                                 size=(self.kh, self.kw))
            raise NotImplementedError
        self._Xp = pad_zero(X, self.pad)
        self._out = conv2d_forward(self._Xp, self.K, self.stride)
        return self._out

    def backward(self, grad):
        if self.K is None:
            raise RuntimeError("forward belum pernah dipanggil")
        # TODO: dXp, self.dK = conv2d_backward(self._Xp, self.K, grad, self.stride)
        #       lalu buang padding dari dXp sesuai self.pad (slicing), return hasilnya
        raise NotImplementedError
