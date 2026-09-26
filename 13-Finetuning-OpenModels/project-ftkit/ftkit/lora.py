"""TODO Bagian 2 — Matematika LoRA (A@B) + gradient check.

Kontrak lengkap ada di tests/test_lora.py. Analogi: adapter kecil di atas
bobot beku — di sini kamu menghitung ΔW = B @ A secara eksplisit.
"""
def buat_lora(dim_in, dim_out, rank, seed=13):
    """Adapter LoRA: A (rank × dim_in) = 0.01*((i+j)%5 - 2);
    B (dim_out × rank) = NOL → ΔW awal 0 (model mulai utuh!)."""
    raise NotImplementedError


def delta_w(adapter):
    """ΔW = B @ A (dim_out × dim_in)."""
    raise NotImplementedError


def hitung_param(adapter):
    """Jumlah param terlatih = rank*(dim_in + dim_out)."""
    raise NotImplementedError


def rasio_param(dim_in, dim_out, rank):
    """hitung_param / (dim_in*dim_out)."""
    raise NotImplementedError


def kontribusi_adapter(x, adapter):
    """Kontribusi adapter ke logit 1-output: (B @ (A @ x))[0]."""
    raise NotImplementedError


def gradient_check(tol=1e-6):
    """Selisih maksimum gradien analitik vs numerik (central diff h=1e-5).

    Pakai inisialisasi(), forward(), gradien(), loss_batch() dari ftkit.model.
    Return selisih absolut maksimum; wajar < 1e-6.
    """
    raise NotImplementedError
