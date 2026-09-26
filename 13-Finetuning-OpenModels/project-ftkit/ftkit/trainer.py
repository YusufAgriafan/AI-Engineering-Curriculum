"""TODO Bagian 4 — Loop SFT + early stopping.

Kontrak lengkap ada di tests/test_trainer.py. Konsep Bab 4 tetap berlaku:
loss, overfitting, val split, early stopping — hanya skalanya yang beda.
"""
from ftkit.model import inisialisasi


def epoch_satu(contoh, w, b, lr):
    """Satu epoch SGD (urutan dataset, tanpa shuffle): update per contoh.

    Return (w_baru, b_baru, loss_rata) — loss dicatat dari p SEBELUM update
    tiap contoh.
    """
    raise NotImplementedError


def latih(train, val, lr=0.5, epochs=30, sabar=4, seed=13):
    """Loop SFT + early stopping.

    - val loss & acc dihitung SETIAP epoch;
    - best = epoch dengan val loss minimum (1-based);
    - berhenti bila val tak membaik `sabar` epoch berturut-turut;
    - restore bobot TERBAIK (bukan bobot terakhir!).

    Return dict: w, b, riwayat {'train_loss','val_loss','val_acc'},
    best_epoch, berhenti_dini, epochs_dijalankan.
    """
    raise NotImplementedError


def dataset_ftkit(tok=None):
    """DATA_SENTIMEN → {'train': [(x17,y)...], 'val': [...]} urutan asli."""
    raise NotImplementedError
