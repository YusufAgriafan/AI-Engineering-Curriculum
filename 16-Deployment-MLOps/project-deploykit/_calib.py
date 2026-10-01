"""Audit angka terkunci deploykit (bukan test) — cetak semua angka kalibrasi SEED 16.

Usage (dari folder project-deploykit):
    python _calib.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import deploykit_ref as ref  # noqa: E402
from deploykit.data import (CANARY_KASUS, ENV_PRODUKSI, LATENSI_P50,  # noqa: E402
                            PROFIL_TOKEN)
from deploykit.konfig import config_layanan  # noqa: E402 (via ref patch nanti)

# Patch modul TODO agar _calib jalan tanpa implementasi mahasiswa
from deploykit import api, canary, konfig, monitor, router, evaluasi  # noqa: E402

konfig.baca_env = ref.baca_env
konfig.config_layanan = ref.config_layanan
api.validasi_chat = ref.validasi_chat
api.RateLimiter = ref.RateLimiter
router.pilih_tier = ref.pilih_tier
router.jawab_router = ref.jawab_router
router.jawab_fallback = ref.jawab_fallback
router.biaya_permintaan = ref.biaya_permintaan
router.breakeven_req_per_hari = ref.breakeven_req_per_hari
canary.evaluasi_canary = ref.evaluasi_canary
canary.rencana_rollout = ref.rencana_rollout
monitor.health_score = ref.health_score
monitor.kueri_ke_kata = ref.kueri_ke_kata
monitor.deteksi_drift = ref.deteksi_drift
monitor.ringkas_permintaan = ref.ringkas_permintaan
evaluasi.serve = ref.serve
evaluasi.laporan_ops = ref.laporan_ops

print("KONFIGURASI (12-factor)")
print("  default  :", ref.baca_env({}))
print("  produksi :", ref.config_layanan(ENV_PRODUKSI))

print("\nVALIDASI & RATE LIMIT")
print("  validasi body valid      →", ref.validasi_chat(
    {"pesan": [{"peran": "user", "teks": "halo"}]}))
print("  validasi body 'halo'     →", ref.validasi_chat("halo"))
rl = ref.RateLimiter(60)
hasil = [rl.izinkan("u1", t) for t in range(60)]
print("  60 req detik 0..59       →", hasil[-1], "| ke-61 @59.5 →", rl.izinkan("u1", 59.5))
print("  jendela baru @60         →", rl.izinkan("u1", 60))
rl2 = ref.RateLimiter(60)
rl2.izinkan("u1", 0.0)
print("  reset_detik @59.2/30/60  →", rl2.reset_detik("u1", 59.2),
      rl2.reset_detik("u1", 30.0), rl2.reset_detik("u1", 60.0))

print("\nROUTER & BIAYA")
print("  'Berapa lama pengiriman reguler?' →", ref.pilih_tier("Berapa lama pengiriman reguler?"))
print("  'Kenapa pesanan dikembalikan?'    →", ref.pilih_tier("Kenapa pesanan saya dikembalikan?"))
h = ref.jawab_router("Berapa lama pengiriman reguler?")
print("  jawab_router mudah  :", h["meta"], "|", h["jawaban"][:56])
print("  jawab_router sulit  :", ref.jawab_router("Jelaskan kebijakan pengembalian.")["meta"])
meta = {"token_in": PROFIL_TOKEN["input"], "token_out": PROFIL_TOKEN["output"]}
print("  biaya api_mini      :", ref.biaya_permintaan(meta, "api_mini"), "USD/req")
print("  biaya api_besar     :", ref.biaya_permintaan(meta, "api_besar"), "USD/req")
print("  breakeven api_mini  :", ref.breakeven_req_per_hari(), "req/hari")
print("  breakeven api_besar :", ref.breakeven_req_per_hari(model="api_besar"), "req/hari")


def gagal(teks, ambang=None):
    return {"jawaban": "", "skor_retrieval": 0.0,
            "meta": {"model": "api_mini", "error": "timeout"}}


hf = ref.jawab_fallback("Berapa lama pengiriman reguler?", gagal_fn=gagal)
print("  fallback            :", hf["meta"], "|", hf["jawaban"][:44])

print("\nCANARY (5 kasus terkunci)")
for nama, (err, skor, p50, bb, bl) in CANARY_KASUS.items():
    print(f"  {nama:<11}:", ref.evaluasi_canary(nama, err, skor, p50, bb, bl))
print("  batas tepat (0.02, 0.75, 480) →",
      ref.evaluasi_canary("x", 0.02, 0.75, 480.0, 0.00033, 0.00033)["putuskan"])
print("  720ms = 1.5 × 480             →",
      ref.evaluasi_canary("x", 0.0, 0.9, 720.0, 0.00033, 0.00033)["putuskan"])

print("\nRENCANA ROLLOUT (config produksi, CANARY_PERSEN=20)")
cfg = ref.config_layanan(ENV_PRODUKSI)
for r in ref.rencana_rollout(cfg):
    print(f"  {r['persen']:>3}% penuh={r['penuh']} CANARY_PERSEN={r['dengan_canary']['CANARY_PERSEN']}")

print("\nMONITORING & DRIFT")
print("  health_score:", ref.health_score())
d = ref.deteksi_drift()
print("  kosakata lama:", d["kosakata_lama"], "kata")
print("  kata baru    :", d["kata_baru"])
print("  proporsi     :", d["proporsi_baru"], "| drift:", d["drift"])

print("\nSERVE END-TO-END (config produksi)")
print("  normal   :", ref.serve({"pesan": [{"peran": "user",
                                         "teks": "Berapa lama pengiriman reguler?"}]}, cfg))
kecil = dict(cfg, RATE_LIMIT_PER_MENIT=2)
print("  limit 2/menit →", ref.serve({"pesan": [{"peran": "user", "teks": "a"}]}, kecil, waktu=0.0)[0],
      ref.serve({"pesan": [{"peran": "user", "teks": "b"}]}, kecil, waktu=1.0)[0],
      ref.serve({"pesan": [{"peran": "user", "teks": "c"}]}, kecil, waktu=2.0))

print("\nLAPORAN OPS")
L = ref.laporan_ops(cfg)
print("  health ok :", L["health"]["ok"], "| skor:", L["health"]["skor"])
print("  drift     :", L["drift"]["drift"], "| breakeven (api_besar):", L["breakeven_req_per_hari"])
print("  biaya harian (2000 req):", L["biaya_harian"])
