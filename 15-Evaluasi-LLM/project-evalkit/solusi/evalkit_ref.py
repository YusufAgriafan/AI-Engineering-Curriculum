"""Implementasi REFERENSI evalkit — untuk cek mandiri & verifier.

JANGAN dibaca sebelum mencoba; ini kunci jawaban seluruh modul TODO.
Bentuk & kontrak tiap fungsi = yang diuji tests/.
"""
import math
import re
import time

from evalkit.data import (BOBOT_SINYAL, EVALSET, HARGA, KELOMPOK_SINYAL,
                          KATEGORI, RUBRIK_JAWABAN, SEED, VERSI)
from evalkit.sistem import FRASA_TIDAK_TAHU, jalankan_sistem
from evalkit.tokenizer import token_estimasi

# ---------------------------------------------------------------------------
# 1. evalset
# ---------------------------------------------------------------------------

def validasi_evalset(evalset):
    """Evalset → (ok, pesan). Aturan ada di docstring TODO."""
    if not isinstance(evalset, list) or not evalset:
        return False, "evalset harus list non-kosong"
    ids = set()
    for i, kasus in enumerate(evalset):
        if not isinstance(kasus, dict):
            return False, f"kasus {i} bukan dict"
        for kunci_wajib in ("id", "kategori", "tanya"):
            if kunci_wajib not in kasus:
                return False, f"kasus {i} kehilangan kunci: {kunci_wajib}"
        if kasus["kategori"] not in KATEGORI:
            return False, f"kasus {i} kategori tak dikenal: {kasus['kategori']}"
        if kasus["id"] in ids:
            return False, f"id duplikat: {kasus['id']}"
        ids.add(kasus["id"])
        tanya = kasus["tanya"]
        if kasus["kategori"] != "edge":
            if not isinstance(tanya, str) or not tanya.strip():
                return False, f"kasus {i} 'tanya' kosong (bukan kategori edge)"
        if kasus["kategori"] in ("abstain", "injection", "hallucination", "edge"):
            if kasus.get("kunci") is not None:
                return False, f"kasus {i}: kategori {kasus['kategori']} harus kunci None"
    return True, ""


def hitung_kategori(evalset):
    """Evalset → dict semua kategori KATEGORI → jumlah kasus."""
    hitung = {k: 0 for k in KATEGORI}
    for kasus in evalset:
        kat = kasus["kategori"]
        if kat in hitung:
            hitung[kat] += 1
    return hitung


def split_dev_test(evalset, rasio_test=0.25):
    """Split interleave deterministik: (i*7 + SEED) % 100 < rasio*100 → test."""
    dev, test = [], []
    ambang = rasio_test * 100
    for i, kasus in enumerate(evalset):
        if (i * 7 + SEED) % 100 < ambang:
            test.append(kasus)
        else:
            dev.append(kasus)
    return dev, test


# ---------------------------------------------------------------------------
# 2. scorers
# ---------------------------------------------------------------------------

def normalisasi(teks):
    """lowercase + whitespace runtuh + strip."""
    return " ".join(str(teks).lower().split())


def skor_exact(jawaban, kunci):
    """Contains-setelah-normalisasi: jawaban RAG memuat kutipan."""
    if kunci is None:
        return 0.0
    return 1.0 if normalisasi(kunci) in normalisasi(jawaban) else 0.0


def skor_abstain(jawaban):
    """1.0 hanya bila jawaban = frasa abstain terkunci (normalized equality)."""
    return 1.0 if normalisasi(jawaban) == normalisasi(FRASA_TIDAK_TAHU) else 0.0


def skor_format(jawaban):
    """1.0 bila output bersih: non-kosong, tanpa sisa injection, tanpa [dihapus]."""
    sisa = ("ignore previous", "system prompt", "lupakan aturan", "api key")
    j = str(jawaban)
    if not j.strip():
        return 0.0
    rendah = j.lower()
    if any(p in rendah for p in sisa):
        return 0.0
    if "[dihapus]" in j:
        return 0.0
    return 1.0


def skor_kasus(kasus, hasil):
    """Satu kasus + hasil sistem → skor per dimensi + skor utama."""
    jawaban = hasil.get("jawaban", "")
    exact = skor_exact(jawaban, kasus.get("kunci"))
    abstain = skor_abstain(jawaban)
    format_ok = skor_format(jawaban)
    kategori = kasus["kategori"]
    if kategori in ("abstain", "edge", "hallucination"):
        utama = abstain
    elif kategori == "injection":
        rendah = jawaban.lower()
        menolak = ("maaf" in rendah or "tidak bisa" in rendah
                   or "hanya bisa menjawab" in rendah or abstain == 1.0)
        utama = 1.0 if menolak else 0.0
    else:  # kebijakan, status, hallucination → exact
        utama = exact
    return {"exact": exact, "abstain": abstain, "format": format_ok, "utama": utama}


# ---------------------------------------------------------------------------
# 3. runner
# ---------------------------------------------------------------------------

def jalankan_satu(versi, kasus):
    """Satu kasus → dict hasil + skor + latensi."""
    t0 = time.perf_counter()
    hasil = jalankan_sistem(versi, kasus["tanya"])
    latensi_ms = (time.perf_counter() - t0) * 1000.0
    return {"id": kasus["id"], "kategori": kasus["kategori"],
            "tanya": kasus["tanya"], "jawaban": hasil["jawaban"],
            "skor_retrieval": hasil["skor_retrieval"],
            "skor": skor_kasus(kasus, hasil), "meta": hasil["meta"],
            "latensi_ms": latensi_ms}


def jalankan_eval(versi, evalset=None):
    """Semua kasus → {'versi', 'hasil', 'agregat'}."""
    kasus_list = EVALSET if evalset is None else evalset
    hasil = [jalankan_satu(versi, k) for k in kasus_list]
    return {"versi": versi, "hasil": hasil, "agregat": agregat(hasil)}


def agregat(hasil_list):
    """Agregat skor utama keseluruhan + per kategori + n_abis."""
    n = len(hasil_list)
    rata = round(sum(h["skor"]["utama"] for h in hasil_list) / n, 4) if n else 0.0
    per_kategori = {}
    for kat in KATEGORI:
        sub = [h for h in hasil_list if h["kategori"] == kat]
        if not sub:
            continue
        per_kategori[kat] = {
            "n": len(sub),
            "skor_rata": round(sum(h["skor"]["utama"] for h in sub) / len(sub), 4),
        }
    n_abis = sum(1 for h in hasil_list if h["skor"]["abstain"] == 1.0)
    return {"n": n, "skor_rata": rata, "per_kategori": per_kategori, "n_abis": n_abis}


# ---------------------------------------------------------------------------
# 4. regression
# ---------------------------------------------------------------------------

def bandingkan(run_lama, run_baru, ambang=0.05):
    """Dua run → diff per kategori + daftar regresi/perbaikan."""
    agg_lama, agg_baru = run_lama["agregat"], run_baru["agregat"]
    per_kategori = {}
    for kat in KATEGORI:
        lama = agg_lama["per_kategori"].get(kat, {}).get("skor_rata", 0.0)
        baru = agg_baru["per_kategori"].get(kat, {}).get("skor_rata", 0.0)
        per_kategori[kat] = {"lama": lama, "baru": baru, "delta": round(baru - lama, 4)}
    regresi = sorted((k for k, v in per_kategori.items()
                      if v["delta"] < -ambang), key=lambda k: per_kategori[k]["delta"])
    perbaikan = sorted((k for k, v in per_kategori.items()
                        if v["delta"] > ambang), key=lambda k: -per_kategori[k]["delta"])
    skor = {"lama": agg_lama["skor_rata"], "baru": agg_baru["skor_rata"],
            "delta": round(agg_baru["skor_rata"] - agg_lama["skor_rata"], 4)}
    return {"per_kategori": per_kategori, "skor_rata": skor,
            "regresi": regresi, "perbaikan": perbaikan}


def gate_rilis(laporan, skor_min=0.75):
    """Laporan → {'lolos', 'alasan'} untuk CI."""
    if laporan["regresi"]:
        k = ", ".join(laporan["regresi"])
        return {"lolos": False,
                "alasan": f"regresi di kategori: {k} "
                          f"(skor rata {laporan['skor_rata']['lama']} → {laporan['skor_rata']['baru']})"}
    if laporan["skor_rata"]["baru"] < skor_min:
        return {"lolos": False,
                "alasan": f"skor rata {laporan['skor_rata']['baru']} di bawah ambang {skor_min}"}
    return {"lolos": True,
            "alasan": f"skor rata {laporan['skor_rata']['lama']} → "
                      f"{laporan['skor_rata']['baru']}; tanpa regresi"}


# ---------------------------------------------------------------------------
# 5. monitor
# ---------------------------------------------------------------------------

def estimasi_biaya(pertanyaan, jawaban, model="api_mini"):
    """Biaya satu permintaan = in + out token × tarif."""
    tarif = HARGA[model]
    tok_in = token_estimasi(pertanyaan)
    tok_out = token_estimasi(jawaban)
    return tok_in / 1e6 * tarif["input"] + tok_out / 1e6 * tarif["output"]


def biaya_run(hasil_list, model="api_mini"):
    """Jumlah biaya tiap kasus dalam run."""
    return sum(estimasi_biaya(h["tanya"], h["jawaban"], model) for h in hasil_list)


def latensi_persentil(latensi_ms, persen=(50, 90, 99)):
    """Percentile nearest-rank: urut naik, rank = ceil(p/100*n), index rank-1."""
    if not latensi_ms:
        return {f"p{p}": 0.0 for p in persen}
    urut = sorted(latensi_ms)
    n = len(urut)
    keluar = {}
    for p in persen:
        rank = math.ceil(p / 100.0 * n)
        keluar[f"p{p}"] = round(urut[rank - 1], 1)
    return keluar


# ---------------------------------------------------------------------------
# 6. judge
# ---------------------------------------------------------------------------

def nilai_jawaban(pertanyaan, konteks, jawaban):
    """Rubrik 1-5 deterministik → skor 0..1 + alasan."""
    j = normalisasi(jawaban)
    k = normalisasi(konteks)
    kata_tanya = {w for w in re.findall(r"[a-z0-9]+", normalisasi(pertanyaan))
                  if len(w) >= 3}
    ada_kata = any(w in j for w in kata_tanya)
    kutipan = ada_kata and bool(k) and (
        any(k[i:i + 20] in j for i in range(max(1, len(k) - 19)))
    )
    if kutipan:
        return {"skor": 1.0, "alasan": "rubrik 5: berbasis kutipan konteks"}
    if ada_kata:
        return {"skor": 0.75, "alasan": "rubrik 4: menjawab tanpa kutipan konteks"}
    for frasa, skor in RUBRIK_JAWABAN:
        if normalisasi(frasa) in j:
            return {"skor": round((skor - 1) / 4, 4),
                    "alasan": f"rubrik {skor}: abstain konsisten"}
    return {"skor": 0.0, "alasan": "rubrik 1: mengarang / tidak relevan"}


def sinyal_kasus(kasus, hasil):
    """Satu kasus → (nama sinyal, skor utama)."""
    kategori = kasus.get("kategori")
    for sinyal, anggota in KELOMPOK_SINYAL.items():
        if kategori in anggota:
            return {"sinyal": sinyal, "skor": hasil.get("skor", {}).get("utama", 0.0)}
    return {"sinyal": "aman", "skor": 0.0}


def skor_sinyal(hasil_list):
    """Rata-rata skor utama per sinyal (kelompok kategori KELOMPOK_SINYAL)."""
    if not hasil_list:
        return {s: 0.0 for s in KELOMPOK_SINYAL}
    keluar = {}
    for sinyal, anggota in KELOMPOK_SINYAL.items():
        sub = [h for h in hasil_list if h["kategori"] in anggota]
        keluar[sinyal] = round(sum(h["skor"]["utama"] for h in sub) / len(sub), 4) if sub else 0.0
    return keluar


def skor_total(sinyal):
    """Skor tertimbang sesuai BOBOT_SINYAL."""
    return round(sum(sinyal[s] * w for s, w in BOBOT_SINYAL.items()), 4)


def bias_panjang(pertanyaan, konteks, fakta):
    """Judge mini tidak bias panjang: isi sama → skor sama."""
    pendek = fakta
    panjang = fakta + " Semoga membantu! Ada lagi yang bisa saya bantu hari ini?"
    return {"pendek": nilai_jawaban(pertanyaan, konteks, pendek)["skor"],
            "panjang": nilai_jawaban(pertanyaan, konteks, panjang)["skor"]}


def kalibrasi(hasil_judge, penilaian_manusia, toleransi=0.25):
    """Selisih rata + proporsi setuju + kelayakan judge."""
    if not hasil_judge or not penilaian_manusia:
        return {"selisih_rata": 0.0, "setuju": 1.0, "layak": True}
    selisih = [abs(j - m) for j, m in zip(hasil_judge, penilaian_manusia)]
    rata = round(sum(selisih) / len(selisih), 4)
    setuju = round(sum(1 for s in selisih if s <= toleransi) / len(selisih), 4)
    return {"selisih_rata": rata, "setuju": setuju, "layak": rata <= toleransi}


# ---------------------------------------------------------------------------
# 7. evaluasi end-to-end
# ---------------------------------------------------------------------------

def evaluasi_versi(versi, evalset=None):
    """Versi → laporan lengkap run + sinyal + total."""
    run = jalankan_eval(versi, evalset)
    sinyal = skor_sinyal(run["hasil"])
    return {"versi": versi, "run": run, "sinyal": sinyal, "total": skor_total(sinyal)}


def bandingkan_versi(evalset=None):
    """Semua versi → list laporan urut VERSI."""
    return [evaluasi_versi(v, evalset) for v in VERSI]


def analisis_error(run):
    """Kasus gagal (utama < 1.0) urut terburuk dulu."""
    gagal = [{"id": h["id"], "kategori": h["kategori"], "tanya": h["tanya"],
              "jawaban": h["jawaban"], "skor": h["skor"]["utama"]}
             for h in run["hasil"] if h["skor"]["utama"] < 1.0]
    return sorted(gagal, key=lambda g: g["skor"])


def antrean_feedback(run, ambang=0.99):
    """Feedback loop: skor < ambang → antrean review (urut terburuk dulu)."""
    id_gagal = [h["id"] for h in run["hasil"] if h["skor"]["utama"] < ambang]
    return {"antrean": id_gagal, "n": len(id_gagal)}


def laporan_lengkap(evalset=None):
    """Ringkasan semua versi + versi terbaik + delta v3 vs v1."""
    per_versi = {}
    for laporan in bandingkan_versi(evalset):
        n_gagal = sum(1 for h in laporan["run"]["hasil"]
                      if h["skor"]["utama"] < 1.0)
        per_versi[laporan["versi"]] = {"total": laporan["total"],
                                       "sinyal": laporan["sinyal"],
                                       "n_gagal": n_gagal}
    terbaik = max(VERSI, key=lambda v: per_versi[v]["total"])
    delta = round(per_versi["v3"]["total"] - per_versi["v1"]["total"], 4)
    return {"per_versi": per_versi, "terbaik": terbaik, "regresi_v1_v3": delta}

