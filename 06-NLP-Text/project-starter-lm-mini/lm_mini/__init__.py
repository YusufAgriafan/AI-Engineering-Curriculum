"""LM mini — paket numpy murni untuk project Bab 6 (NLP & Text).

Modul:
    corpus.py       : korpus mini terkunci (JANGAN DIUBAH)
    text.py         : tokenize, vocab, encode OOV-aman, pad/truncate (TODO kamu)
    embeddings.py   : pasangan skip-gram + skip-gram negative sampling (TODO kamu)
    languagemodel.py: bigram smoothing, cross-entropy, sampling temperature (TODO kamu)
    rnn.py          : langkah RNN, forward batch, BPTT (TODO kamu)

Prasyarat: numpy saja. Semua kontrak ada di tests/.
"""

__all__ = ["corpus", "text", "embeddings", "languagemodel", "rnn"]
