"""CNN mini — paket numpy murni untuk project Bab 5 (Computer Vision).

Modul:
    data.py    : dataset "bars" terkunci + loader (JANGAN DIUBAH)
    conv.py    : zero-padding, konvolusi 2D, max-pool + backward (TODO kamu)
    augment.py : augmentasi data gambar (flip, noise, shift)   (TODO kamu)
    model.py   : CNNMini — conv → relu → maxpool → dense       (TODO kamu)
    train.py   : mini-batch SGD + early stopping               (TODO kamu)

Prasyarat: numpy saja. Semua kontrak ada di tests/.
"""

__all__ = ["data", "conv", "augment", "model", "train"]
