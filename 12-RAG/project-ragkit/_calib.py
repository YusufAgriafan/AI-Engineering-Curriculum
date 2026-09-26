"""Kalibrasi angka project Bab 12 — dijalankan sekali untuk mengunci assert.

Usage: python _calib.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ragkit_ref as ref  # noqa: E402
from ragkit.chunking import buat_semua_chunks  # noqa: E402
from ragkit.data import KUERI_LUAR_CORPUS, KUERI_UJI  # noqa: E402

print("=" * 68)
print("CHUNKING (GIVEN)")
print("=" * 68)
chunks = buat_semua_chunks()
print("  total chunks:", len(chunks))
for c in chunks:
    print("   [%d] %-32s p%d len=%d" % (c["chunk_index"], c["source"],
                                        c["page"], len(c["text"])))

print()
print("=" * 68)
print("VECTORSTORE + RETRIEVER")
print("=" * 68)
retrieve = ref.buat_retriever()
dapat = retrieve(KUERI_UJI[0]["teks"], 3)
print("  q1 top3 :", [(d["chunk"]["chunk_index"], round(d["score"], 4))
                     for d in dapat])
for k in (1, 2, 3, 4, 5):
    print(f"  hit@{k} =", ref.evaluasi_retrieval(retrieve, top_k=k)["hit_rate"])
ev = ref.evaluasi_retrieval(retrieve, top_k=1)
print("  gagal hit@1:", ev["kueri_gagal"])

print()
print("=" * 68)
print("ABSTAIN GATE")
print("=" * 68)
h6 = ref.jalankan_rag(retrieve, KUERI_LUAR_CORPUS["teks"])
print("  q6 skor terbaik :", round(h6["skor_terbaik"], 4))
print("  q6 jawaban      :", h6["jawaban"])
print("  q6 abstaikan_oleh:", h6.get("abstaikan_oleh"))

print()
print("=" * 68)
print("PIPELINE (end-to-end)")
print("=" * 68)
h = ref.jalankan_rag(retrieve, KUERI_UJI[0]["teks"])
print("  jawaban :", h["jawaban"])
print("  sumber  :", h["sumber"])
print("  biaya   :", h["biaya_usd"])
print("  tahap   :", h["tahap"])
ev2 = ref.evaluasi_ujung_ke_ujung(lambda q: ref.jalankan_rag(retrieve, q))
print("  akurasi :", ev2["akurasi"], "| abstain_benar:", ev2["abstain_benar"])

print()
print("=" * 68)
print("HYBRID + RERANK")
print("=" * 68)
hasil_h = ref.cari_hybrid(retrieve, chunks, KUERI_UJI[0]["teks"], top_k=3)
print("  hybrid q1 top3:", [d["chunk"]["chunk_index"] for d in hasil_h])
dapat3 = retrieve(KUERI_UJI[1]["teks"], 3)
dapat3r = ref.rerank(KUERI_UJI[1]["teks"], dapat3, top_k=2)
print("  rerank q2:", [d["chunk"]["chunk_index"] for d in dapat3], "->",
      [d["chunk"]["chunk_index"] for d in dapat3r])

print()
print("=" * 68)
print("CACHE + TOLOK UKUR")
print("=" * 68)
tb = ref.tolok_ukur(retrieve)
print("  latensi_ms        :", tb["latensi_ms"])
print("  cache             :", tb["cache"])
print("  hit_rate          :", tb["hit_rate"])
print("  panggilan_pencarian:", tb["panggilan_pencarian"])
print("  total_biaya_usd   :", tb["total_biaya_usd"])
print("  harga tarif mini  : input 0.15 | output 0.60 (USD/1jt token)")
print("  kueri uji         :", [q["id"] for q in KUERI_UJI])
