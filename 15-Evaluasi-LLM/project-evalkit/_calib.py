"""Audit angka terkunci evalkit (bukan test) — cetak semua angka kalibrasi SEED 15.

Usage (dari folder project-evalkit):
    python _calib.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import evalkit_ref as ref  # noqa: E402
from evalkit.data import AMBAT_SKOR, EVALSET, LATENSI_MS  # noqa: E402
from evalkit.sistem import FRASA_TIDAK_TAHU, PENOLAKAN_INJECTION  # noqa: E402

print("EVALSET")
print(f"  {len(EVALSET)} kasus | ambang abstain: {AMBAT_SKOR}")
print("  skor retrieval per kasus (v2):")
for kasus in EVALSET:
    hasil = ref.jalankan_sistem("v2", kasus["tanya"])
    print(f"    {kasus['id']} {kasus['kategori']:<14} skor={hasil['skor_retrieval']:<7}"
          f"→ {hasil['jawaban'][:52]}")

print("\nSCORERS")
print("  skor_exact('...14 hari sejak diterima.', '14 hari') →",
      ref.skor_exact("Barang bisa dikembalikan dalam 14 hari sejak diterima.", "14 hari"))
print("  skor_abstain(FRASA_TIDAK_TAHU) →", ref.skor_abstain(FRASA_TIDAK_TAHU))
print("  skor_abstain('Maaf, saya tidak tahu.') →", ref.skor_abstain("Maaf, saya tidak tahu."))
print("  skor_format('api key: sk-123') →", ref.skor_format("api key: sk-123"))
print("  skor_format('[dihapus] ...') →", ref.skor_format("[dihapus] instruksi"))

print("\nRUN (skor rata + per kategori)")
run_v = {}
for v in ("v1", "v2", "v3"):
    run_v[v] = ref.jalankan_eval(v)
    agg = run_v[v]["agregat"]
    per = {k: d["skor_rata"] for k, d in agg["per_kategori"].items()}
    print(f"  {v}: rata={agg['skor_rata']} | n_abis={agg['n_abis']} | {per}")

print("\nSINYAL & TOTAL")
for v in ("v1", "v2", "v3"):
    sinyal = ref.skor_sinyal(run_v[v]["hasil"])
    print(f"  {v}: sinyal={sinyal} → total={ref.skor_total(sinyal)}")

print("\nREGRESSION & GATE (v1 → v3)")
laporan = ref.bandingkan(run_v["v1"], run_v["v3"])
print("  skor_rata:", laporan["skor_rata"])
print("  regresi  :", laporan["regresi"], "| perbaikan:", laporan["perbaikan"])
print("  gate     :", ref.gate_rilis(laporan))
laporan_balik = ref.bandingkan(run_v["v2"], run_v["v1"])
print("  gate (v2 → v1, harus GAGAL):", ref.gate_rilis(laporan_balik))

print("\nMONITOR")
r3 = run_v["v3"]["hasil"]
print("  biaya run v3 api_mini :", ref.biaya_run(r3), "USD")
print("  biaya run v3 api_besar:", ref.biaya_run(r3, "api_besar"), "USD")
print("  persentil api_mini    :", ref.latensi_persentil(list(LATENSI_MS["api_mini"])))
print("  persentil lokal       :", ref.latensi_persentil(list(LATENSI_MS["lokal"])))

print("\nJUDGE")
print("  kutipan konteks  →", ref.nilai_jawaban(
    "Berapa lama pengiriman reguler?",
    "Pengiriman reguler datang 3 sampai 5 hari kerja.",
    "Pengiriman reguler datang 3 sampai 5 hari kerja."))
print("  tanpa kutipan    →", ref.nilai_jawaban(
    "Berapa lama pengiriman reguler?",
    "Pengiriman reguler datang 3 sampai 5 hari kerja.",
    "Kurang lebih lima hari biasanya ya."))
print("  abstain          →", ref.nilai_jawaban("q", "k", FRASA_TIDAK_TAHU))
print("  mengarang        →", ref.nilai_jawaban("q", "k", "Sepertinya sekitar 3 hari."))
print("  bias panjang     →", ref.bias_panjang(
    "Berapa lama pengiriman?", "Pengiriman reguler datang 3 sampai 5 hari kerja.",
    "Pengiriman reguler datang 3 sampai 5 hari kerja."))
print("  kalibrasi        →", ref.kalibrasi([1.0, 0.25, 0.0], [1.0, 0.0, 0.0]))

print("\nERROR ANALYSIS & FEEDBACK LOOP")
for v in ("v1", "v2", "v3"):
    gagal = ref.analisis_error(run_v[v])
    antre = ref.antrean_feedback(run_v[v])
    print(f"  {v}: gagal={[g['id'] for g in gagal]} | antrean feedback={antre['antrean']}")

print("\nLAPORAN LENGKAP")
L = ref.laporan_lengkap()
print("  total per versi:", {v: d["total"] for v, d in L["per_versi"].items()})
print("  n_gagal        :", {v: d["n_gagal"] for v, d in L["per_versi"].items()})
print("  terbaik        :", L["terbaik"], "| delta v3-v1:", L["regresi_v1_v3"])

print("\nPENOLAKAN INJECTION (v3):", PENOLAKAN_INJECTION)
