"""Implementasi REFERENSI ftkit — untuk cek mandiri & verifier.

JANGAN dibaca sebelum mencoba; ini kunci jawaban seluruh modul TODO.
Bentuk & kontrak tiap fungsi = yang diuji tests/.
"""
import json
import math

from ftkit.data import (DATA_SENTIMEN, FORMAT_SFT, HARGA,
                        LABEL_POS, TEMPLATE_INSTRUKSI, VARIAN_KUANTISASI)
from ftkit.model import (DIM, akurasi, forward, gradien, inisialisasi,
                         label_ke_y, loss_bce, loss_batch)
from ftkit.tokenizer import TokenizerMini, fitur


# ============================ 1. sft ============================

def ke_contoh_sft(baris):
    """Dict {instruksi, input, respons} → teks prompt ala Alpaca + respons.

    Format prompt terkunci (diuji): instruksi + input + 'Jawaban:'.
    """
    return (f"{baris['instruksi']}\n\nInput: {baris['input']}\n\nJawaban:",
            baris["respons"])


def muat_jsonl(path):
    """Baca JSONL → list of dict (satu objek per baris)."""
    hasil = []
    with open(path, "r", encoding="utf-8") as f:
        for baris in f:
            baris = baris.strip()
            if baris:
                hasil.append(json.loads(baris))
    return hasil


def tulis_jsonl(path, baris_list):
    """List of dict → JSONL (satu objek per baris, utf-8)."""
    with open(path, "w", encoding="utf-8") as f:
        for baris in baris_list:
            f.write(json.dumps(baris, ensure_ascii=False) + "\n")


def oversample(baris_list, per_label=4):
    """Samakan jumlah contoh per label respons dengan duplikasi deterministik.

    Urutan ASLI dipertahankan; label minoritas diduplikasi berulang (dari awal
    lagi) sampai kuota terpenuhi — cara paling murah menangani ketidakseimbangan
    tanpa mengarang data baru.
    """
    hasil = list(baris_list)
    per_kelas = {}
    for b in baris_list:
        per_kelas.setdefault(b["respons"], []).append(b)
    for label, kumpulan in per_kelas.items():
        i = 0
        while sum(1 for b in hasil if b["respons"] == label) < per_label:
            hasil.append(kumpulan[i % len(kumpulan)])
            i += 1
    return hasil


def split_sft(baris_list, rasio_val=0.25, seed=13):
    """Bagi deterministik: tiap contoh ke-i → val bila (i * 7 + seed) % 10 < rasio_val*10.

    Interleaved (bukan blok) supaya kedua split merepresentasikan seluruh data.
    """
    train, val = [], []
    ambang = rasio_val * 10
    for i, b in enumerate(baris_list):
        (val if (i * 7 + seed) % 10 < ambang else train).append(b)
    return train, val


# ============================ 2. lora ============================

def buat_lora(dim_in, dim_out, rank, seed=13):
    """Adapter LoRA: A (rank × dim_in), B (dim_out × rank), diinisialisasi
    deterministik: A = 0.01*((i+j)%5 - 2), B = 0 → delta awal 0 (penting!)."""
    A = [[0.01 * ((i + j) % 5 - 2) for j in range(dim_in)] for i in range(rank)]
    B = [[0.0] * rank for _ in range(dim_out)]
    return {"A": A, "B": B, "rank": rank, "dim_in": dim_in, "dim_out": dim_out}


def delta_w(adapter):
    """ΔW = B @ A (dim_out × dim_in). B awal 0 → ΔW awal 0: model mulai utuh."""
    B, A = adapter["B"], adapter["A"]
    dim_out, rank, dim_in = len(B), adapter["rank"], adapter["dim_in"]
    return [[sum(B[i][r] * A[r][j] for r in range(rank)) for j in range(dim_in)]
            for i in range(dim_out)]


def hitung_param(adapter):
    """Jumlah param terlatih = rank*(dim_in + dim_out) — bukan rank produk penuh."""
    return adapter["rank"] * (adapter["dim_in"] + adapter["dim_out"])


def rasio_param(dim_in, dim_out, rank):
    """Param LoRA / param full fine-tune."""
    penuh = dim_in * dim_out
    return hitung_param(buat_lora(dim_in, dim_out, rank)) / penuh


def kontribusi_adapter(x, adapter):
    """Kontribusi adapter ke logit untuk vektor 1-output: (B @ (A @ x))[0]."""
    A = adapter["A"]
    Ax = [sum(A[r][j] * x[j] for j in range(adapter["dim_in"]))
          for r in range(adapter["rank"])]
    return sum(adapter["B"][0][r] * Ax[r] for r in range(adapter["rank"]))


def gradient_check(tol=1e-6):
    """Bandingkan gradien analitik vs numerik loss BCE pada titik acak-teratur.

    Gradien analitik (model.py) vs selisih-hitungan (central difference h=1e-5).
    Return selisih absolut maksimum; wajar < 1e-6.
    """
    w, b = inisialisasi()
    x = [0.5, -0.2, 0.8, 0.1, -0.6, 0.3, 0.0, 0.2, -0.4, 0.7, 0.1, -0.1, 0.3, 0.5, -0.5, 0.2, 0.0]
    y = 1.0
    p, _ = forward(x, w, b)
    gw, gb = gradien(x, p, y)
    h = 1e-5
    maks = 0.0
    for i in range(len(w)):
        wp = list(w); wp[i] += h
        wm = list(w); wm[i] -= h
        num = (loss_batch([(x, y)], wp, b) - loss_batch([(x, y)], wm, b)) / (2 * h)
        maks = max(maks, abs(gw[i] - num))
    bp, bm = b + h, b - h
    num_b = (loss_batch([(x, y)], w, bp) - loss_batch([(x, y)], w, bm)) / (2 * h)
    maks = max(maks, abs(gb - num_b))
    return maks


# ============================ 3. quantize ============================

def kuantisasi_simulasi(bobot, bit):
    """Simulasi kuantisasi simetris: skala = max|w| / ((2^(bit-1)) - 1);
    q = round(w / skala) dibatasi rentang; dekuanti = q * skala. Return list baru."""
    if bit >= 32:
        return list(bobot)
    levels = 2 ** (bit - 1) - 1
    skala = max(abs(w) for w in bobot) / levels if levels else 0.0
    if skala == 0.0:
        return list(bobot)
    keluar = []
    for w in bobot:
        q = int(round(w / skala))
        q = max(-levels, min(levels, q))
        keluar.append(q * skala)
    return keluar


def laju_kompresi(bit):
    """FP32 → bit-bit: rasio ukuran = 32/bit."""
    return 32.0 / bit


def ukuran_model_gb(param_miliar, bit):
    """Ukuran bobot (GB) = param × bit / 8 → GB (1e9)."""
    return param_miliar * bit / 8.0


def bandingkan_varian(varian=None):
    """Tabel varian kuantisasi: bit, ukuran 7B (GB), kompresi, skor eval."""
    varian = VARIAN_KUANTISASI if varian is None else varian
    return [{"nama": v["nama"], "bit": v["bit"],
             "ukuran_gb": round(ukuran_model_gb(7.0, v["bit"]), 2),
             "kompresi": round(laju_kompresi(v["bit"]), 1),
             "skor": v["skor"]} for v in varian]


# ============================ 4. trainer ============================

def epoch_satu(contoh, w, b, lr):
    """Satu epoch SGD (urutan dataset, tanpa shuffle): update per contoh.

    Return (w_baru, b_baru, loss_rata) — loss dicatat dari p SEBELUM update
    tiap contoh (bukan loss epoch setelahnya).
    """
    w, b = list(w), b
    total = 0.0
    for x, y in contoh:
        p, _ = forward(x, w, b)
        total += loss_bce(p, y)
        gw, gb = gradien(x, p, y)
        w = [wi - lr * gwi for wi, gwi in zip(w, gw)]
        b = b - lr * gb
    n = len(contoh) if contoh else 1
    return w, b, total / n


def latih(train, val, lr=0.5, epochs=30, sabar=4, seed=13):
    """Loop SFT + early stopping.

    - loss val dihitung SETIAP epoch dengan bobot saat itu;
    - best = epoch ke-i loss val minimum (1-based);
    - berhenti bila val tak membaik `sabar` epoch berturut-turut;
    - restore bobot TERBAIK (bukan bobot terakhir!).

    Return dict: w, b, riwayat {'train_loss','val_loss','val_acc'}, best_epoch, berhenti_dini.
    """
    w, b = inisialisasi(seed)
    riwayat = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")
    best = (list(w), b)
    best_epoch = 0
    tanpa_baik = 0
    berhenti_dini = False
    for ep in range(1, epochs + 1):
        w, b, tr = epoch_satu(train, w, b, lr)
        vl = loss_batch(val, w, b)
        va = akurasi(val, w, b)
        riwayat["train_loss"].append(tr)
        riwayat["val_loss"].append(vl)
        riwayat["val_acc"].append(va)
        if vl < best_val - 1e-12:
            best_val = vl
            best = (list(w), b)
            best_epoch = ep
            tanpa_baik = 0
        else:
            tanpa_baik += 1
            if tanpa_baik >= sabar:
                berhenti_dini = True
                break
    return {"w": best[0], "b": best[1], "riwayat": riwayat,
            "best_epoch": best_epoch, "berhenti_dini": berhenti_dini,
            "epochs_dijalankan": len(riwayat["train_loss"])}


def dataset_ftkit(tok=None):
    """DATA_SENTIMEN → list (x17, y) per split. Deterministik, urutan asli."""
    tok = TokenizerMini() if tok is None else tok
    keluar = {"train": [], "val": []}
    for d in DATA_SENTIMEN:
        keluar[d["split"]].append((fitur(d["teks"], tok), label_ke_y(d["label"])))
    return keluar


# ============================ 5. api ============================

def biaya_api(token_in, token_out, model="api_mini"):
    """USD = in/1e6*harga_in + out/1e6*harga_out ('lokal' = 0.0)."""
    t = HARGA[model]
    return token_in / 1e6 * t["input"] + token_out / 1e6 * t["output"]


def biaya_bulanan(permintaan_per_hari, token_in, token_out, model="api_mini", hari=30):
    """Proyeksi biaya bulanan; angka besar memakai float biasa (cukup)."""
    return biaya_api(token_in, token_out, model) * permintaan_per_hari * hari


def token_estimasi(teks):
    """Cukup untuk proyeksi, bukan tagihan: ceil(len/4)."""
    return math.ceil(len(str(teks)) / 4.0)


def keputusan_deployment(permintaan_per_hari, token_in, token_out,
                         biaya_gpu_per_bulan=60.0, hari=30):
    """API vs lokal: biaya API bulanan vs GPU. Return dict + keputusan.

    Ambang: biaya_api_bulanan > biaya_gpu → 'lokal'; selain itu 'api'.
    """
    api = biaya_bulanan(permintaan_per_hari, token_in, token_out, "api_mini", hari)
    keputusan = "lokal" if api > biaya_gpu_per_bulan else "api"
    return {"biaya_api_bulanan": api, "biaya_lokal_bulanan": biaya_gpu_per_bulan,
            "keputusan": keputusan}


# ============================ 6. evaluasi ============================

def prediksi(teks, w, b, tok=None, ambang=0.5):
    """teks → {'label','skor','yakin'} — yakin = jarak ke 0.5 (sisi mana pun)."""
    tok = TokenizerMini() if tok is None else tok
    x = fitur(teks, tok)
    p, _ = forward(x, w, b)
    label = LABEL_POS if p >= ambang else "negatif"
    return {"label": label, "skor": p, "yakin": abs(p - ambang) * 2}


def evaluasi_sebelum_sesudah(val, w_sebelum, b_sebelum, w_sesudah, b_sesudah):
    """Akurasi val dua titik + delta. Return {'sebelum','sesudah','delta'}."""
    s = akurasi(val, w_sebelum, b_sebelum)
    e = akurasi(val, w_sesudah, b_sesudah)
    return {"sebelum": s, "sesudah": e, "delta": e - s}


def kebijakan_ood(hasil_pred, skor_min=0.6, jawaban_aman="kurang yakin, mohon periksa"):
    """Yakin < skor_min → jangan tampilkan label; arahkan ke manusia (Bab 12: abstain)."""
    if hasil_pred["yakin"] < skor_min:
        return {**hasil_pred, "tampilkan": False, "aksi": jawaban_aman}
    return {**hasil_pred, "tampilkan": True, "aksi": "tampilkan label"}


def laporan_eval(val, w, b, ood_hasil=None, skor_min=0.6):
    """Laporan 3 baris: akurasi, contoh salah (jika ada), keputusan OOD."""
    ood = kebijakan_ood(ood_hasil, skor_min) if ood_hasil else None
    salah = []
    for x, y in val:
        p, _ = forward(x, w, b)
        pred = 1.0 if p >= 0.5 else 0.0
        if pred != y:
            salah.append({"p": round(p, 4), "y": y})
    return {"akurasi": akurasi(val, w, b), "salah": salah, "ood": ood}


# dipakai verifier untuk memastikan format SFT ref = format yang diuji
def format_sft_ref():
    """Referensi: baris JSONL siap tulis dari FORMAT_SFT."""
    return [dict(ke_contoh_sft(b), respons=b["respons"]) for b in FORMAT_SFT]
