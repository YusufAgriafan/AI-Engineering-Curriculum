"""Dataset & konstanta terkunci project Bab 9 — JANGAN DIUBAH.

Semua angka di tests/ dan notebook dikalibrasi terhadap konstanta di sini.
"""
SEED = 9

CORPUS_TEKS = "low low lower lowest newer newer wider wider new new new"

# mini-LM: vocab 4 token, pola A-B bergantian
LM_VOCAB = 4
LM_D = 16
LM_SEQ = [0, 1, 0, 1, 0, 1, 0, 1]

# sampling: logits 8 token terkunci
SAMPLE_LOGITS = [0.1, 2.2, 0.5, 1.3, 0.1, 0.9, 0.2, 0.4]
