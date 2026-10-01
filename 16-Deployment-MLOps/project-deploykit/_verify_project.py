"""Verify project-deploykit: patch referensi + 70 test + notebook (bukan test).

Usage (dari folder project-deploykit):
    python _verify_project.py
"""
import contextlib
import io
import re
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "solusi"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import deploykit_ref as ref  # noqa: E402
from deploykit import api, canary, evaluasi, konfig, monitor, router  # noqa: E402

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

# 1) Semua test harus hijau dengan implementasi referensi
suite = unittest.TestLoader().discover(str(BASE / "tests"))
runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
hasil = runner.run(suite)
if not hasil.wasSuccessful():
    print(">>> TEST GAGAL dengan implementasi referensi")
    sys.exit(1)
print(f">>> ALL {hasil.testsRun} TESTS OK")

# 2) Notebook starter & solusi: eksekusi seluruh cell (TODO di-skip di starter)
REF_PATCH = """
import sys
sys.path.insert(0, r"{base}")
sys.path.insert(0, r"{solusi}")
import deploykit_ref as ref
from deploykit import api, canary, evaluasi, konfig, monitor, router
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
""".format(base=str(BASE), solusi=str(BASE / "solusi"))


def jalankan_notebook(path, skip_todo):
    import json
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    ctx = {"__name__": "__main__"}
    exec(REF_PATCH, ctx)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if skip_todo and "TODO" in src:
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, f"<{Path(path).name} cell {i}>", "exec"), ctx)
        except Exception as e:
            print(f"FAILED {Path(path).name} cell {i}: {type(e).__name__}: {e}")
            sys.exit(1)
    print(f"{Path(path).name}: SEMUA CELL OK")


jalankan_notebook(BASE / "starter.ipynb", skip_todo=True)
jalankan_notebook(BASE / "solusi.ipynb", skip_todo=False)

print("\nPROJECT VERIFY OK")
