"""Audit angka terkunci ftkit (bukan test) — cetak semua angka kalibrasi SEED 13.

Usage (dari folder project-ftkit):
    python _calib.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ftkit_ref as ref  # noqa: E402
from ftkit.model import inisialisasi, loss_batch  # noqa: E402

data = ref.dataset_ftkit()
print("DATASET")
print(f"  train {len(data['train'])} | val {len(data['val'])} | fitur 17-dim")

w0, b0 = inisialisasi()
print("  loss awal (bobot inisialisasi):", round(loss_batch(data['train'], w0, b0), 4))

print("\nSFT")
print("  FORMAT_SFT:", len(ref.FORMAT_SFT), "baris (6 positif / 2 negatif)")
train_s, val_s = ref.split_sft(ref.FORMAT_SFT, 0.25)
print("  split 8 contoh rasio 0.25 →", (len(train_s), len(val_s)))
ov = ref.oversample(ref.FORMAT_SFT, per_label=6)
print("  oversample per_label=6 → 12 baris (6/6)")

print("\nLORA")
print("  param 4096×4096 r16:", ref.hitung_param(ref.buat_lora(4096, 4096, 16)),
      f"| rasio {ref.rasio_param(4096, 4096, 16)}")
print("  delta awal = 0 (B nol):", all(v == 0.0 for row in
      ref.delta_w(ref.buat_lora(4, 4, 2)) for v in row))
print("  gradient check max err:", ref.gradient_check())

print("\nTRAINER (data bersih, lr=0.5, epochs=60)")
h = ref.latih(data["train"], data["val"], lr=0.5, epochs=60, sabar=6)
r = h["riwayat"]
print(f"  train_loss: {r['train_loss'][0]:.4f} → {r['train_loss'][-1]:.4f}"
      f" | val_acc akhir {r['val_acc'][-1]:.3f} | dini {h['berhenti_dini']}")

print("\nTRAINER (2 label train dibalik + lr=3.0 → OVERFIT)")
train_noisy = [(x, (1.0 - y) if i in (6, 14) else y) for i, (x, y) in enumerate(data["train"])]
h2 = ref.latih(train_noisy, data["val"], lr=3.0, epochs=100, sabar=5)
r2 = h2["riwayat"]
print(f"  berhenti dini {h2['berhenti_dini']} | dijalankan {h2['epochs_dijalankan']}"
      f" | best epoch {h2['best_epoch']}")
print(f"  val_loss best {r2['val_loss'][h2['best_epoch']-1]:.4f}"
      f" vs akhir {r2['val_loss'][-1]:.4f} | acc best {r2['val_acc'][h2['best_epoch']-1]:.3f}")

print("\nKUANTISASI (7B)")
for t in ref.bandingkan_varian():
    print(f"  {t['nama']:>4}: {t['ukuran_gb']:>5} GB | kompresi {t['kompresi']}x | skor {t['skor']}")

print("\nBIAYA (1k in / 300 out)")
print(f"  api_besar ${ref.biaya_api(1000, 300, 'api_besar')}"
      f" | api_mini ${ref.biaya_api(1000, 300, 'api_mini')}"
      f" | lokal ${ref.biaya_api(1000, 300, 'lokal')}")
print(f"  bulanan 2000 req/hari api_mini: ${ref.biaya_bulanan(2000, 1000, 300, 'api_mini')}")
print(f"  keputusan  2000/hari: {ref.keputusan_deployment(2000, 1000, 300)['keputusan']}"
      f" | 10000/hari: {ref.keputusan_deployment(10000, 1000, 300)['keputusan']}")

print("\nEVALUASI")
w0, b0 = inisialisasi()
print("  sebelum/sesudah:", ref.evaluasi_sebelum_sesudah(data["val"], w0, b0, h["w"], h["b"]))
p_ood = ref.prediksi("Lumayan standar saja sih", h["w"], h["b"])
print(f"  OOD skor {p_ood['skor']:.4f} yakin {p_ood['yakin']:.4f}"
      f" → tampilkan {ref.kebijakan_ood(p_ood)['tampilkan']}")
