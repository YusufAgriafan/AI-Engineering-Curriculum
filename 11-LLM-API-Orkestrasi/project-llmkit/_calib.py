"""Kalibrasi angka project Bab 11 — dijalankan sekali untuk mengunci assert.

Usage: python _calib.py
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import llmkit_ref as ref  # noqa: E402
from llmkit.data import (HARGA, LANGKAH_PIPELINE, RANTAI_FALLBACK, TIKET_CONTOH,  # noqa: E402
                         TOOLS)
from llmkit.transport import (GalatAPI, GalatPermintaanBuruk,  # noqa: E402
                              ProviderPalsu, provider_cadangan, provider_utama)

PESAN = [{"role": "user", "content": "Jelaskan apa itu machine learning dalam 2 kalimat."}]

print("=" * 68)
print("COST")
print("=" * 68)
print("  token_estimasi('') =", ref.token_estimasi(""))
print("  token_estimasi(1..4 char) =", [ref.token_estimasi("x" * n) for n in (1, 4, 5, 9)])
print("  biaya mini 1000/300 =", ref.hitung_biaya(1000, 300, "mini"))
print("  biaya besar 1000/300 =", ref.hitung_biaya(1000, 300, "besar"))
print("  biaya mini dasar (tokens 0) =", ref.hitung_biaya(0, 0, "mini"))

p = provider_utama()
ang = ref.jalankan_anggaran(provider_utama(), [PESAN] * 4, 0.00003, model="mini")
print("  anggaran 4 job, batas $0.00005 ->", {k: v for k, v in ang.items() if k != "hasil"})
print("  biaya per hasil:", [round(h["biaya_usd"], 8) for h in ang["hasil"]])

print()
print("=" * 68)
print("RETRY")
print("=" * 68)
p = provider_utama()
tunggu = []
h = ref.panggil_dengan_retry(p, "sedang", PESAN, sleep_fn=tunggu.append)
print("  sedang (rencana 1 gagal): attempts", h["attempts"], "tunggu", h["timpenungguan_ms"],
      "latency", h["latency_ms"], "total", h["total_ms"], "riwayat", h["riwayat"])
h2 = ref.panggil_dengan_retry(provider_utama(), "mini", PESAN)
print("  mini (langsung ok): attempts", h2["attempts"], "tunggu", h2["timpenungguan_ms"])
try:
    ref.panggil_dengan_retry(provider_utama(), "besar", PESAN, maks_percobaan=2,
                             sleep_fn=None)
except Exception as e:
    print("  besar maks=2 ->", type(e).__name__, "riwayat", getattr(e, "riwayat", None))
p3 = provider_utama()
try:
    ref.panggil_dengan_retry(p3, "besar", PESAN, maks_percobaan=3)
except Exception as e:
    print("  besar maks=3 ->", type(e).__name__, e)
try:
    ref.panggil_dengan_retry(provider_utama(), "model_ajaib", PESAN)
except Exception as e:
    print("  model tak dikenal ->", type(e).__name__, "| jenis:", getattr(e, "jenis", None))
print("  backoff:", [ref.backoff_ms(n) for n in (1, 2, 3, 4, 5)])
print("  backoff basis 100 maks 500:", [ref.backoff_ms(n, 100, 3, 500) for n in (1, 2, 3, 4)])

print()
print("=" * 68)
print("FALLBACK")
print("=" * 68)
provs = {"utama": provider_utama(), "cadangan": provider_cadangan()}
h = ref.panggil_dengan_fallback(provs, RANTAI_FALLBACK, PESAN, settings={"maks_percobaan": 2})
print("  model terpakai:", h["model"], "| lompatan:", h["lompatan"], "| dicoba:", h["dicoba"])
try:
    ref.panggil_dengan_fallback({"utama": provider_cadangan()}, RANTAI_FALLBACK, PESAN,
                                settings={"maks_percobaan": 1})
except Exception as e:
    print("  rantai gagal ->", type(e).__name__, getattr(e, "riwayat", None))

print()
print("=" * 68)
print("CACHE")
print("=" * 68)
c = ref.CachePrompt(ttl=3)
p = provider_utama()
r1 = ref.panggil_bercache(p, "mini", PESAN, c, sekarang=0)
r2 = ref.panggil_bercache(p, "mini", PESAN, c, sekarang=1)
r3 = ref.panggil_bercache(p, "mini", PESAN, c, sekarang=5)
r4 = ref.panggil_bercache(p, "mini", PESAN, c, sekarang=6)
print("  cache:", r1["cache"], r2["cache"], r3["cache"], r4["cache"])
print("  statistik:", c.statistik())
print("  kunci sama untuk params sama:",
      c.kunci("mini", PESAN, {"temperature": 0.0}) == c.kunci("mini", PESAN, {"temperature": 0.0}))
print("  kunci beda model:", c.kunci("mini", PESAN) != c.kunci("sedang", PESAN))
print("  kunci:", c.kunci("mini", PESAN))
c2 = ref.CachePrompt(ttl=2)
c2.simpan("a", {"x": 1}, 0)
print("  bersihkan(t=5) ->", c2.bersihkan(5), "| expired:", c2.statistik())

print()
print("=" * 68)
print("STREAM")
print("=" * 68)
s = ref.konsumsi_stream(provider_utama().stream("mini", PESAN))
print("  ", {k: (v if k != "teks" else v[:40] + "...") for k, v in s.items()})

print()
print("=" * 68)
print("TOOLS")
print("=" * 68)
print("  skema_tools:", json.dumps(ref.skema_tools()[:1], ensure_ascii=False)[:150])
print("  validasi ok   :", ref.validasi_argumen("get_cuaca", {"kota": "Bandung", "satuan": "celsius"}))
print("  validasi kosong:", ref.validasi_argumen("get_cuaca", {}))
print("  validasi tipe :", ref.validasi_argumen("get_cuaca", {"kota": 5, "satuan": "celsius"}))
print("  validasi extra:", ref.validasi_argumen("get_cuaca", {"kota": "a", "satuan": "c", "x": 1}))
print("  validasi tool tak ada:", ref.validasi_argumen("nuklir", {}))
print("  jalankan_tool ok:", ref.jalankan_tool("get_cuaca", {"kota": "Jakarta", "satuan": "celsius"}))
print("  jalankan_tool gagal:", ref.jalankan_tool("get_cuaca", {"kota": 5}))

loop1 = ref.loop_tool(provider_utama(), "mini", "Berapa suhu Jakarta hari ini?")
print("  loop 1 tool:", {k: v for k, v in loop1.items() if k not in ("riwayat",)})
loop2 = ref.loop_tool(provider_utama(), "mini",
                      "Cari catatan tentang machine learning, lalu hitung 15 * 8 + 20.")
print("  loop 2 tool: iterasi", loop2["iterasi"], "tool", loop2["tool_dipakai"],
      "jawaban", (loop2["jawaban"] or "")[:45])
loop3 = ref.loop_tool(provider_utama(), "mini", "Jelaskan singkat apa itu embedding.")
print("  loop tanpa tool: iterasi", loop3["iterasi"], "tool", loop3["tool_dipakai"])

print()
print("=" * 68)
print("PIPELINE")
print("=" * 68)
pl = ref.jalankan_pipeline(provider_utama(), LANGKAH_PIPELINE, TIKET_CONTOH)
print("  berhasil/gagal:", pl["berhasil"], "/", pl["gagal"])
print("  total biaya   :", pl["total_biaya_usd"])
print("  total_ms      :", pl["total_ms"])
for l in pl["langkah"]:
    print("   ", l["nama"], l["model"], "sukses", l["sukses"], "attempts", l["attempts"],
          "ms", l["total_ms"], "biaya", l["biaya_usd"])
print("  teks_akhir:", (pl["teks_akhir"] or "")[:50])
pl2 = ref.jalankan_pipeline(provider_utama(), LANGKAH_PIPELINE, TIKET_CONTOH,
                            maks_percobaan=1)
print("  maks=1 ->", [(l["nama"], l["sukses"], l["galat"]) for l in pl2["langkah"]])
print("  berhenti: ", pl2["berhasil"], "/", pl2["gagal"], "teks_akhir", pl2["teks_akhir"])
print("  harga:", HARGA)
print("  tools:", list(TOOLS))
