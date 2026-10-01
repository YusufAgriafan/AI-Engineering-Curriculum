"""Implementasi referensi deploykit — identik kontrak tests/.

Struktur sama dengan evalkit_ref.py (Bab 15): seluruh fungsi TODO di sini
berisi implementasi jadi; starter mengisi deploykit/ sampai 57 test hijau.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploykit.data import (AMBAT_DRIFT, DRIFT, DRIFT_JAM_PENUH, HARGA,
                            HEALTH_METRIK, KATA_SULIT, MAKS_KARAKTER,
                            MAKS_PESAN, PROFIL_TOKEN, SCHEMA_ENV,
                            SLO_LATENSI_RASIO, STOPWORD)
from deploykit.sistem import model_besar, model_mini
from deploykit.tokenizer import token_estimasi


# ===========================================================================
# Bagian 1 — Konfigurasi 12-factor
# ===========================================================================
def baca_env(env, schema=None):
    schema = SCHEMA_ENV if schema is None else schema
    cfg = {}
    for kunci, (tipe, default) in schema.items():
        if kunci not in env:
            cfg[kunci] = default
            continue
        mentah = env[kunci]
        if tipe == "int":
            try:
                cfg[kunci] = int(str(mentah).strip())
            except (TypeError, ValueError):
                raise ValueError(f"env {kunci} bukan int: {mentah!r}")
        elif tipe == "float":
            try:
                cfg[kunci] = float(str(mentah).strip())
            except (TypeError, ValueError):
                raise ValueError(f"env {kunci} bukan float: {mentah!r}")
        else:
            cfg[kunci] = mentah
    return cfg


def config_layanan(env=None):
    cfg = baca_env(env if env is not None else {})
    if not cfg["PORT"] > 0:
        raise ValueError("PORT harus > 0")
    if not cfg["RATE_LIMIT_PER_MENIT"] > 0:
        raise ValueError("RATE_LIMIT_PER_MENIT harus > 0")
    if not cfg["CANARY_PERSEN"] > 0:
        raise ValueError("CANARY_PERSEN harus > 0")
    for kunci in ("AMBAT_ABSTAIN", "SKOR_MIN"):
        if not 0 < cfg[kunci] < 1:
            raise ValueError(f"{kunci} harus di antara 0 dan 1")
    if cfg["MODEL_API"] not in HARGA:
        raise ValueError(f"MODEL_API tak dikenal: {cfg['MODEL_API']}")
    return cfg


# ===========================================================================
# Bagian 2 — Validasi + rate limit
# ===========================================================================
def validasi_chat(body):
    if not isinstance(body, dict):
        return 422, {"error": "body harus dict"}
    pesan = body.get("pesan")
    if not isinstance(pesan, list) or not pesan:
        return 422, {"error": "pesan harus list non-kosong"}
    for m in pesan:
        if not isinstance(m, dict) or "peran" not in m or "teks" not in m:
            return 422, {"error": "format pesan salah"}
        if not str(m.get("teks", "")).strip():
            return 422, {"error": "format pesan salah"}
    if len(pesan) > MAKS_PESAN:
        return 422, {"error": "terlalu banyak pesan"}
    if len(str(pesan[-1]["teks"])) > MAKS_KARAKTER:
        return 422, {"error": "pesan terlalu panjang"}
    return 200, body


class RateLimiter:
    """Fixed window per menit: jendela = [mulai, mulai + 60 detik)."""

    def __init__(self, batas_per_menit=60):
        if batas_per_menit <= 0:
            raise ValueError("batas_per_menit harus > 0")
        self.batas = batas_per_menit
        self.slot = {}     # klien → (mulai_detik, kuota)

    def izinkan(self, klien, waktu_detik=0.0):
        waktu = float(waktu_detik)
        if klien not in self.slot or waktu >= self.slot[klien][0] + 60.0:
            self.slot[klien] = (waktu, self.batas)
        mulai, sisa = self.slot[klien]
        if sisa <= 0:
            return False, 0
        sisa -= 1
        self.slot[klien] = (mulai, sisa)
        return True, sisa

    def reset_detik(self, klien, waktu_detik=0.0):
        if klien not in self.slot:
            return 0
        mulai, _ = self.slot[klien]
        sisa = mulai + 60.0 - float(waktu_detik)
        return math.ceil(sisa) if sisa > 0 else 0


# ===========================================================================
# Bagian 3 — Router + fallback + biaya
# ===========================================================================
def pilih_tier(pertanyaan):
    rendah = str(pertanyaan).lower()
    for frasa in KATA_SULIT:
        if frasa in rendah:
            return "api_besar"
    return "api_mini"


def jawab_router(pertanyaan, ambang=None, fn_besar=None, fn_mini=None):
    fn_besar = model_besar if fn_besar is None else fn_besar
    fn_mini = model_mini if fn_mini is None else fn_mini
    tier = pilih_tier(pertanyaan)
    hasil = (fn_besar if tier == "api_besar" else fn_mini)(pertanyaan, ambang)
    meta = dict(hasil["meta"], tier=tier,
                router="hard" if tier == "api_besar" else "mudah")
    return {"jawaban": hasil["jawaban"], "skor_retrieval": hasil["skor_retrieval"],
            "meta": meta}


def jawab_fallback(pertanyaan, utama="api_mini", cadangan="api_besar",
                   ambang=None, gagal_fn=None, cadangan_fn=None):
    """Model utama "error" (meta['error']) → coba cadangan.
    gagal_fn/cadangan_fn untuk injeksi model tiruan (default model_*)."""
    fn_utama = (model_besar if utama == "api_besar" else model_mini) \
        if gagal_fn is None else gagal_fn
    fn_cadangan = (model_besar if cadangan == "api_besar" else model_mini) \
        if cadangan_fn is None else cadangan_fn
    hasil = fn_utama(pertanyaan, ambang)
    if hasil["meta"].get("error") is None:
        meta = dict(hasil["meta"], tier="utama")
        return {"jawaban": hasil["jawaban"],
                "skor_retrieval": hasil["skor_retrieval"], "meta": meta}
    cadangan_hasil = fn_cadangan(pertanyaan, ambang)
    meta = dict(cadangan_hasil["meta"], tier="cadangan",
                error_utama=hasil["meta"]["error"])
    return {"jawaban": cadangan_hasil["jawaban"],
            "skor_retrieval": cadangan_hasil["skor_retrieval"], "meta": meta}


def biaya_permintaan(meta, model):
    tarif = HARGA[model]
    tok_in = meta.get("token_in", 0)
    tok_out = meta.get("token_out", 0)
    return tok_in / 1e6 * tarif["input"] + tok_out / 1e6 * tarif["output"]


def breakeven_req_per_hari(biaya_gpu_per_bulan=60.0, model="api_mini"):
    biaya_per_req = (PROFIL_TOKEN["input"] * HARGA[model]["input"]
                     + PROFIL_TOKEN["output"] * HARGA[model]["output"]) / 1e6
    return round((biaya_gpu_per_bulan / 30.0) / biaya_per_req)


# ===========================================================================
# Bagian 4 — Canary rollout
# ===========================================================================
def evaluasi_canary(nama, error_rate, skor_eval, p50_baru_ms, biaya_baru,
                    biaya_lama, p50_lama_ms=480.0, skor_min=0.75):
    if error_rate > 0.02:
        return {"nama": nama, "putuskan": "tahan",
                "alasan": f"error rate {round(error_rate, 4)} > 0.02"}
    if skor_eval < skor_min:
        return {"nama": nama, "putuskan": "tahan",
                "alasan": f"skor eval {round(skor_eval, 4)} < {skor_min}"}
    if p50_baru_ms > SLO_LATENSI_RASIO * p50_lama_ms:
        rasio = round(p50_baru_ms / p50_lama_ms, 4)
        return {"nama": nama, "putuskan": "tahan",
                "alasan": f"latensi p50 {p50_baru_ms} > {SLO_LATENSI_RASIO} × {p50_lama_ms} (rasio {rasio})"}
    if biaya_baru > 2 * biaya_lama:
        return {"nama": nama, "putuskan": "tahan",
                "alasan": f"biaya {round(biaya_baru, 4)} > 2 × {round(biaya_lama, 4)}"}
    return {"nama": nama, "putuskan": "lanjut", "alasan": "semua SLO terpenuhi"}


def rencana_rollout(config, langkah=(5, 20, 50, 100)):
    mulai = config["CANARY_PERSEN"]
    persen_langkah = sorted({mulai, *[l for l in langkah if l > mulai]})
    rencana = []
    for p in persen_langkah:
        with_canary = dict(config, CANARY_PERSEN=p)
        rencana.append({"persen": p, "dengan_canary": with_canary,
                        "penuh": p >= 100})
    return rencana


# ===========================================================================
# Bagian 5 — Monitoring & drift
# ===========================================================================
def health_score(metrik=None):
    metrik = HEALTH_METRIK if metrik is None else metrik
    rincian, nilai = [], []
    for nama, (v, target, arah) in metrik.items():
        ok = (v <= target) if arah == "maks" else (v >= target)
        rincian.append({"metrik": nama, "nilai": v, "target": target, "ok": ok})
        nilai.append(1.0 if ok else 0.0)
    skor = round(sum(nilai) / len(nilai), 4)
    return {"skor": skor, "ok": all(nilai), "rincian": rincian}


def kueri_ke_kata(kueri):
    import re
    return {k for k in re.findall(r"[a-z0-9]+", str(kueri).lower())
            if len(k) >= 3 and k not in STOPWORD}


def deteksi_drift(data_drift=None, ambang=None, jam_penuh=None):
    data_drift = DRIFT if data_drift is None else data_drift
    ambang = AMBAT_DRIFT if ambang is None else ambang
    jam_penuh = DRIFT_JAM_PENUH if jam_penuh is None else jam_penuh

    lama = [q for jam, q in data_drift if jam < jam_penuh]
    baru = [q for jam, q in data_drift if jam >= jam_penuh]
    vocab_lama = set().union(*[kueri_ke_kata(q) for q in lama]) if lama else set()
    kata_baru = sorted({w for q in baru for w in kueri_ke_kata(q)} - vocab_lama)
    n_baru_dengan = sum(1 for q in baru if kueri_ke_kata(q) & set(kata_baru))
    proporsi = round(n_baru_dengan / len(baru), 4) if baru else 0.0
    return {"kosakata_lama": len(vocab_lama), "kata_baru": kata_baru,
            "proporsi_baru": proporsi, "drift": proporsi >= ambang}


def ringkas_permintaan(hasil_model):
    jawaban = str(hasil_model.get("jawaban", ""))
    return {"model": hasil_model.get("meta", {}).get("model"),
            "abstain": bool(hasil_model.get("meta", {}).get("abstain")),
            "token_out": token_estimasi(jawaban),
            "singkat": jawaban[:40]}


# ===========================================================================
# Bagian 6 — End-to-end + laporan ops
# ===========================================================================
# Bagian 2/3 versi referensi didefinisikan di atas; untuk serve, pakai
# implementasi LOKAL agar tidak mengimpor modul TODO mahasiswa.
def serve(body, config, waktu=None, limiter=None):
    """Satu permintaan /chat → (status, hasil).

    Alur: validasi_chat (422) → rate limit (429, {'error': 'rate limit
    tercapai'}) → router jawab (200). waktu None → detik 0.
    limiter None → RateLimiter(config['RATE_LIMIT_PER_MENIT']) baru.
    Klien tunggal: 'default'."""
    waktu = 0.0 if waktu is None else waktu
    status, hasil = validasi_chat(body)
    if status != 200:
        return status, hasil
    limiter = RateLimiter(config["RATE_LIMIT_PER_MENIT"]) if limiter is None else limiter
    ok, _sisa = limiter.izinkan("default", waktu)
    if not ok:
        return 429, {"error": "rate limit tercapai"}
    teks = body["pesan"][-1]["teks"]
    jawaban = jawab_router(teks, ambang=config["AMBAT_ABSTAIN"])
    return 200, jawaban


def laporan_ops(config=None):
    config = config_layanan(None) if config is None else config
    biaya_harian = {}
    for model, tarif in HARGA.items():
        per_req = (PROFIL_TOKEN["input"] * tarif["input"]
                   + PROFIL_TOKEN["output"] * tarif["output"]) / 1e6
        biaya_harian[model] = per_req * 2000
    return {"health": health_score(),
            "drift": deteksi_drift(),
            "breakeven_req_per_hari": breakeven_req_per_hari(model=config["MODEL_API"]),
            "biaya_harian": biaya_harian}
