"""RNN Elman char-level: langkah tunggal, forward batch, dan BPTT.

Tugasmu: lengkapi bagian bertanda TODO. Kontrak lengkap ada di docstring dan
di tests/test_rnn.py.

Matematika:

    forward : h_t    = tanh(x_t @ Wx + h_{t-1} @ Wh.T + bh)      h_pre disimpan
              logits = h_T @ Wy + by          -> softmax stabil
    backward: dlogits = (probs - onehot(y)) / n                  (softmax + CE)
              dWy = h_T.T @ dlogits ; dby = dlogits.sum(0) ; dh = dlogits @ Wy.T
              loop t MUNDUR:
                  da_t  = dh * (1 - tanh(h_pre_t)**2)
                  dWx  += x_t.T @ da_t ; dWh += da_t.T @ h_{t-1}
                  dbh  += da_t.sum(0) ;  dh = da_t @ Wh    <- gradien mengalir ke waktu

Aturan emas: dWx/dWh/dbh di-akumulasi (+=) sepanjang waktu — BUKAN assign.
"""

import numpy as np

__all__ = ["onehot_batch", "rnn_step_forward", "rnn_step_backward",
           "rnn_forward", "rnn_backward", "softmax"]


def onehot_batch(ids, width):
    """(n,) id -> matriks (n, width) one-hot."""
    z = np.zeros((len(ids), width))
    z[np.arange(len(ids)), ids] = 1.0
    return z


def softmax(logits):
    """Softmax stabil per baris: kurangi max dulu sebelum exp."""
    # TODO
    raise NotImplementedError


def rnn_step_forward(x_t, h_prev, Wx, Wh, bh):
    """Satu langkah waktu. x_t (n, in), h_prev (n, hidden).
    Return (h, cache) dengan cache = (h_prev, h_pre) dan h = tanh(h_pre).
    """
    # TODO: h_pre = x_t @ Wx + h_prev @ Wh.T + bh ; h = np.tanh(h_pre)
    #       return (h, (h_prev, h_pre))
    raise NotImplementedError


def rnn_step_backward(dh, cache, x_t, Wx, Wh):
    """Backward satu langkah. Return (dx_t, dh_prev, dWx, dWh, dbh).

    da = dh * (1 - tanh(h_pre)**2)
    dWx = x_t.T @ da ; dWh = da.T @ h_prev ; dbh = da.sum(0)
    dx_t = da @ Wx.T ; dh_prev = da @ Wh
    """
    # TODO
    raise NotImplementedError


def rnn_forward(X, params, ctx):
    """Forward penuh. X: (n, ctx) id token; params = (Wx, Wh, Wy, bh, by).
    Return (probs, cache) dengan cache = (hs, Hs):
        hs = [h_0, h_1, ..., h_ctx]  (h_0 = zeros)
        Hs = [h_pre_1, ..., h_pre_ctx]
    logits hanya dari state TERAKHIR: h_ctx @ Wy + by.
    """
    Wx, Wh, Wy, bh, by = params
    n = len(X)
    hidden = Wh.shape[0]
    hs = [np.zeros((n, hidden))]
    Hs = []
    # TODO: loop t in range(ctx): rnn_step_forward dengan x = onehot_batch(X[:, t], V)
    #       lalu logits = hs[-1] @ Wy + by ; probs = softmax(logits)
    raise NotImplementedError


def rnn_backward(probs, y, X, params, cache, ctx):
    """BPTT: return (dWx, dWh, dWy, dbh, dby) — rata-rata per batch.

    y: (n,) id target. Pembagian n terjadi di dlogits (softmax + CE rata-rata),
    lalu mengalir ke semua gradien. dWx/dWh/dbh di-akumulasi sepanjang waktu.
    """
    Wx, Wh, Wy, bh, by = params
    hs, Hs = cache
    n = len(X)
    vocab = Wy.shape[1]
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dbh = np.zeros_like(bh)
    # TODO 1: dlogits = (probs - onehot_batch(y, vocab)) / n
    # TODO 2: dWy = hs[-1].T @ dlogits ; dby = dlogits.sum(0) ; dh = dlogits @ Wy.T
    # TODO 3: loop t mundur (ctx-1 .. 0):
    #         x_t = onehot_batch(X[:, t], vocab)
    #         _, dh_prev, dWx_step, dWh_step, dbh_step = rnn_step_backward(
    #             dh, (hs[t], Hs[t]), x_t, Wx, Wh)
    #         dWx += dWx_step ; dWh += dWh_step ; dbh += dbh_step ; dh = dh_prev
    raise NotImplementedError
