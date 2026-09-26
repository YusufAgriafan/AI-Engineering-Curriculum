"""tok_mini — paket numpy murni untuk project Bab 9 (Fondasi LLM).

Modul:
    bpe.py        : normalisasi, pair counting, merge, latih/encode/decode (TODO kamu)
    tokenizer.py  : vocab + tokenizer end-to-end (TODO kamu)
    embedding.py  : cosine, similaritas, analogi vektor (TODO kamu)
    position.py   : positional encoding sinusoidal (TODO kamu)
    attention.py  : softmax stabil + scaled dot-product attention (TODO kamu)
    lm.py         : mini language model + training (SEBAGIAN GIVEN)
    sampling.py   : greedy, top-k, top-p (TODO kamu)

Prasyarat: numpy saja. Semua kontrak ada di tests/.
"""
__all__ = ["bpe", "tokenizer", "embedding", "position", "attention", "lm", "sampling"]
