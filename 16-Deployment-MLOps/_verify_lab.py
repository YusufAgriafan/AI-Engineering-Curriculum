"""Verify 01_lab_deploy.ipynb: exec every cell; TODO cells get reference impls.

Usage (dari folder ini):
    python _verify_lab.py
"""
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

BASE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.chdir(BASE)  # notebook memakai Path.cwd() untuk menemukan project-deploykit

NB = BASE / "01_lab_deploy.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

sys.path.insert(0, str(BASE / "project-deploykit"))
sys.path.insert(0, str(BASE / "project-deploykit" / "solusi"))

import deploykit_ref as ref  # noqa: E402

REF = {
    "baca_env": ref.baca_env,
    "config_layanan": ref.config_layanan,
    "validasi_chat": ref.validasi_chat,
    "RateLimiter": ref.RateLimiter,
    "pilih_tier": ref.pilih_tier,
    "jawab_router": ref.jawab_router,
    "jawab_fallback": ref.jawab_fallback,
    "biaya_permintaan": ref.biaya_permintaan,
    "breakeven_req_per_hari": ref.breakeven_req_per_hari,
    "evaluasi_canary": ref.evaluasi_canary,
    "rencana_rollout": ref.rencana_rollout,
    "health_score": ref.health_score,
    "kueri_ke_kata": ref.kueri_ke_kata,
    "deteksi_drift": ref.deteksi_drift,
    "ringkas_permintaan": ref.ringkas_permintaan,
    "serve": ref.serve,
    "laporan_ops": ref.laporan_ops,
    "serve_degradasi": (
        "def serve_degradasi(body, config, waktu=None, limiter=None,"
        " gagal_router_fn=None, gagal_fallback_fn=None):\n"
        "    status, hasil = validasi_chat(body)\n"
        "    if status != 200:\n"
        "        return status, hasil\n"
        "    limiter = RateLimiter(config['RATE_LIMIT_PER_MENIT']) if limiter is None else limiter\n"
        "    ok, _ = limiter.izinkan('default', 0.0 if waktu is None else waktu)\n"
        "    if not ok:\n"
        "        return 429, {'error': 'rate limit tercapai'}\n"
        "    teks = body['pesan'][-1]['teks']\n"
        "    h = jawab_router(teks, ambang=config['AMBAT_ABSTAIN'])\n"
        "    if h['meta'].get('error') is None:\n"
        "        return 200, h\n"
        "    h2 = jawab_fallback(teks, ambang=config['AMBAT_ABSTAIN'],\n"
        "                        gagal_fn=gagal_router_fn) if gagal_router_fn else jawab_fallback(teks, ambang=config['AMBAT_ABSTAIN'])\n"
        "    # gunakan fn fallback tiruan bila diberikan\n"
        "    if gagal_fallback_fn is not None:\n"
        "        h2 = gagal_fallback_fn(teks, config['AMBAT_ABSTAIN'])\n"
        "        h2['meta'] = dict(h2['meta'], tier='cadangan',\n"
        "                          error_utama=h['meta'].get('error'))\n"
        "        if h2['meta'].get('error') is None:\n"
        "            return 200, h2\n"
        "    elif h2['meta'].get('error') is None:\n"
        "        return 200, h2\n"
        "    return 200, {'jawaban': 'Maaf, layanan sedang terganggu. Coba lagi sebentar.',\n"
        "                 'skor_retrieval': 0.0,\n"
        "                 'meta': {'model': 'degraded', 'degraded': True}}\n"
    ),
}


def nama_ref(src):
    names = []
    for n in REF:
        if f"def {n}(" in src or f"class {n}" in src:
            names.append(n)
    return names


ctx = {"__name__": "__main__", "__builtins__": __builtins__}

# Konstanta yang dipakai REF & cek di lab
from deploykit.data import (AMBAT_DRIFT, CANARY_KASUS, DRIFT,  # noqa: E402
                            DRIFT_JAM_PENUH, ENV_PRODUKSI, HARGA,
                            HEALTH_METRIK, PROFIL_TOKEN, STOPWORD)
from deploykit.sistem import model_besar, model_mini  # noqa: E402
from deploykit.tokenizer import token_estimasi  # noqa: E402

ctx.update({
    "baca_env": REF["baca_env"], "config_layanan": REF["config_layanan"],
    "validasi_chat": REF["validasi_chat"], "RateLimiter": REF["RateLimiter"],
    "pilih_tier": REF["pilih_tier"], "jawab_router": REF["jawab_router"],
    "jawab_fallback": REF["jawab_fallback"],
    "biaya_permittaian_placeholder": None,
    "biaya_permintaan": REF["biaya_permintaan"],
    "breakeven_req_per_hari": REF["breakeven_req_per_hari"],
    "evaluasi_canary": REF["evaluasi_canary"],
    "rencana_rollout": REF["rencana_rollout"],
    "health_score": REF["health_score"], "kueri_ke_kata": REF["kueri_ke_kata"],
    "deteksi_drift": REF["deteksi_drift"],
    "ringkas_permintaan": REF["ringkas_permintaan"],
    "serve": REF["serve"], "laporan_ops": REF["laporan_ops"],
    "model_mini": model_mini, "model_besar": model_besar,
    "token_estimasi": token_estimasi,
    "HARGA": HARGA, "PROFIL_TOKEN": PROFIL_TOKEN, "STOPWORD": STOPWORD,
    "CANARY_KASUS": CANARY_KASUS, "ENV_PRODUKSI": ENV_PRODUKSI,
    "HEALTH_METRIK": HEALTH_METRIK, "DRIFT": DRIFT,
    "AMBAT_DRIFT": AMBAT_DRIFT, "DRIFT_JAM_PENUH": DRIFT_JAM_PENUH,
    "HEALTH_METRIK_BURUK": None,
})

t0 = time.time()
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src:
        names = nama_ref(src)
        if not names:
            print(f"[cell {i:>2}] SKIPPED (TODO tanpa fungsi dikenal)")
            continue
        for n in names:
            if isinstance(REF[n], str):
                exec(REF[n], ctx)
            else:
                ctx[n] = REF[n]
        print(f"[cell {i:>2}] injected: {', '.join(names)}")
        continue
    cap = io.StringIO()
    try:
        with contextlib.redirect_stdout(cap):
            exec(compile(src, f"<lab cell {i}>", "exec"), ctx)
    except Exception as e:
        print(f"FAILED di cell {i}: {type(e).__name__}: {e}")
        print(cap.getvalue()[-1500:])
        sys.exit(1)
    out = cap.getvalue()
    if "\u2705" in out:
        baris = [l for l in out.splitlines() if "\u2705" in l]
        print(f"[cell {i:>2}] OK  {baris[0][:88] if baris else ''}")
    else:
        print(f"[cell {i:>2}] OK ({len(out)} chars)")

print(f"\nSemua cell lab dieksekusi dalam {time.time() - t0:.1f} detik.")
print("LAB VERIFY OK")
