"""Verify 01_lab_finetuning.ipynb: exec every cell; TODO cells get reference impls.

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

os.chdir(BASE)  # notebook memakai Path.cwd() untuk menemukan project-ftkit

NB = BASE / "01_lab_finetuning.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "ke_contoh_sft": (
        "def ke_contoh_sft(baris):\n"
        "    return (f\"{baris['instruksi']}\\n\\nInput: {baris['input']}\\n\\nJawaban:\",\n"
        "            baris[\"respons\"])\n"
    ),
    "tulis_jsonl": (
        "def tulis_jsonl(path, baris_list):\n"
        "    with open(path, \"w\", encoding=\"utf-8\") as f:\n"
        "        for baris in baris_list:\n"
        "            f.write(json.dumps(baris, ensure_ascii=False) + \"\\n\")\n"
    ),
    "muat_jsonl": (
        "def muat_jsonl(path):\n"
        "    hasil = []\n"
        "    with open(path, \"r\", encoding=\"utf-8\") as f:\n"
        "        for baris in f:\n"
        "            baris = baris.strip()\n"
        "            if baris:\n"
        "                hasil.append(json.loads(baris))\n"
        "    return hasil\n"
    ),
    "split_sft": (
        "def split_sft(baris_list, rasio_val=0.25, seed=13):\n"
        "    train, val = [], []\n"
        "    ambang = rasio_val * 10\n"
        "    for i, b in enumerate(baris_list):\n"
        "        (val if (i * 7 + seed) % 10 < ambang else train).append(b)\n"
        "    return train, val\n"
    ),
    "oversample": (
        "def oversample(baris_list, per_label=4):\n"
        "    hasil = list(baris_list)\n"
        "    per_kelas = {}\n"
        "    for b in baris_list:\n"
        "        per_kelas.setdefault(b[\"respons\"], []).append(b)\n"
        "    for label, kumpulan in per_kelas.items():\n"
        "        i = 0\n"
        "        while sum(1 for b in hasil if b[\"respons\"] == label) < per_label:\n"
        "            hasil.append(kumpulan[i % len(kumpulan)])\n"
        "            i += 1\n"
        "    return hasil\n"
    ),
    "buat_lora": (
        "def buat_lora(dim_in, dim_out, rank, seed=13):\n"
        "    A = [[0.01 * ((i + j) % 5 - 2) for j in range(dim_in)] for i in range(rank)]\n"
        "    B = [[0.0] * rank for _ in range(dim_out)]\n"
        "    return {\"A\": A, \"B\": B, \"rank\": rank, \"dim_in\": dim_in, \"dim_out\": dim_out}\n"
    ),
    "delta_w": (
        "def delta_w(adapter):\n"
        "    B, A = adapter[\"B\"], adapter[\"A\"]\n"
        "    dim_out, rank, dim_in = len(B), adapter[\"rank\"], adapter[\"dim_in\"]\n"
        "    return [[sum(B[i][r] * A[r][j] for r in range(rank)) for j in range(dim_in)]\n"
        "            for i in range(dim_out)]\n"
    ),
    "hitung_param": (
        "def hitung_param(adapter):\n"
        "    return adapter[\"rank\"] * (adapter[\"dim_in\"] + adapter[\"dim_out\"])\n"
    ),
    "rasio_param": (
        "def rasio_param(dim_in, dim_out, rank):\n"
        "    penuh = dim_in * dim_out\n"
        "    return hitung_param(buat_lora(dim_in, dim_out, rank)) / penuh\n"
    ),
    "kontribusi_adapter": (
        "def kontribusi_adapter(x, adapter):\n"
        "    A = adapter[\"A\"]\n"
        "    Ax = [sum(A[r][j] * x[j] for j in range(adapter[\"dim_in\"]))\n"
        "          for r in range(adapter[\"rank\"])]\n"
        "    return sum(adapter[\"B\"][0][r] * Ax[r] for r in range(adapter[\"rank\"]))\n"
    ),
    "gradient_check": (
        "def gradient_check(tol=1e-6):\n"
        "    w, b = inisialisasi()\n"
        "    x = [0.5, -0.2, 0.8, 0.1, -0.6, 0.3, 0.0, 0.2, -0.4, 0.7, 0.1, -0.1, 0.3, 0.5, -0.5, 0.2, 0.0]\n"
        "    y = 1.0\n"
        "    p, _ = forward(x, w, b)\n"
        "    gw, gb = gradien(x, p, y)\n"
        "    h = 1e-5\n"
        "    maks = 0.0\n"
        "    for i in range(len(w)):\n"
        "        wp = list(w); wp[i] += h\n"
        "        wm = list(w); wm[i] -= h\n"
        "        num = (loss_batch([(x, y)], wp, b) - loss_batch([(x, y)], wm, b)) / (2 * h)\n"
        "        maks = max(maks, abs(gw[i] - num))\n"
        "    bp, bm = b + h, b - h\n"
        "    num_b = (loss_batch([(x, y)], w, bp) - loss_batch([(x, y)], w, bm)) / (2 * h)\n"
        "    maks = max(maks, abs(gb - num_b))\n"
        "    return maks\n"
    ),
    "kuantisasi_simulasi": (
        "def kuantisasi_simulasi(bobot, bit):\n"
        "    if bit >= 32:\n"
        "        return list(bobot)\n"
        "    levels = 2 ** (bit - 1) - 1\n"
        "    skala = max(abs(w) for w in bobot) / levels if levels else 0.0\n"
        "    if skala == 0.0:\n"
        "        return list(bobot)\n"
        "    keluar = []\n"
        "    for w in bobot:\n"
        "        q = int(round(w / skala))\n"
        "        q = max(-levels, min(levels, q))\n"
        "        keluar.append(q * skala)\n"
        "    return keluar\n"
    ),
    "laju_kompresi": (
        "def laju_kompresi(bit):\n"
        "    return 32.0 / bit\n"
    ),
    "ukuran_model_gb": (
        "def ukuran_model_gb(param_miliar, bit):\n"
        "    return param_miliar * bit / 8.0\n"
    ),
    "bandingkan_varian": (
        "def bandingkan_varian(varian=None):\n"
        "    varian = VARIAN_KUANTISASI if varian is None else varian\n"
        "    return [{\"nama\": v[\"nama\"], \"bit\": v[\"bit\"],\n"
        "             \"ukuran_gb\": round(ukuran_model_gb(7.0, v[\"bit\"]), 2),\n"
        "             \"kompresi\": round(laju_kompresi(v[\"bit\"]), 1),\n"
        "             \"skor\": v[\"skor\"]} for v in varian]\n"
    ),
    "epoch_satu": (
        "def epoch_satu(contoh, w, b, lr):\n"
        "    w, b = list(w), b\n"
        "    total = 0.0\n"
        "    for x, y in contoh:\n"
        "        p, _ = forward(x, w, b)\n"
        "        total += loss_bce(p, y)\n"
        "        gw, gb = gradien(x, p, y)\n"
        "        w = [wi - lr * gwi for wi, gwi in zip(w, gw)]\n"
        "        b = b - lr * gb\n"
        "    n = len(contoh) if contoh else 1\n"
        "    return w, b, total / n\n"
    ),
    "latih": (
        "def latih(train, val, lr=0.5, epochs=30, sabar=4, seed=13):\n"
        "    w, b = inisialisasi(seed)\n"
        "    riwayat = {\"train_loss\": [], \"val_loss\": [], \"val_acc\": []}\n"
        "    best_val = float(\"inf\")\n"
        "    best = (list(w), b)\n"
        "    best_epoch = 0\n"
        "    tanpa_baik = 0\n"
        "    berhenti_dini = False\n"
        "    for ep in range(1, epochs + 1):\n"
        "        w, b, tr = epoch_satu(train, w, b, lr)\n"
        "        vl = loss_batch(val, w, b)\n"
        "        va = akurasi(val, w, b)\n"
        "        riwayat[\"train_loss\"].append(tr)\n"
        "        riwayat[\"val_loss\"].append(vl)\n"
        "        riwayat[\"val_acc\"].append(va)\n"
        "        if vl < best_val - 1e-12:\n"
        "            best_val = vl\n"
        "            best = (list(w), b)\n"
        "            best_epoch = ep\n"
        "            tanpa_baik = 0\n"
        "        else:\n"
        "            tanpa_baik += 1\n"
        "            if tanpa_baik >= sabar:\n"
        "                berhenti_dini = True\n"
        "                break\n"
        "    return {\"w\": best[0], \"b\": best[1], \"riwayat\": riwayat,\n"
        "            \"best_epoch\": best_epoch, \"berhenti_dini\": berhenti_dini,\n"
        "            \"epochs_dijalankan\": len(riwayat[\"train_loss\"])}\n"
    ),
    "biaya_api": (
        "def biaya_api(token_in, token_out, model='api_mini'):\n"
        "    t = HARGA[model]\n"
        "    return token_in / 1e6 * t[\"input\"] + token_out / 1e6 * t[\"output\"]\n"
    ),
    "biaya_bulanan": (
        "def biaya_bulanan(permintaan_per_hari, token_in, token_out, model='api_mini', hari=30):\n"
        "    return biaya_api(token_in, token_out, model) * permintaan_per_hari * hari\n"
    ),
    "token_estimasi": (
        "def token_estimasi(teks):\n"
        "    return math.ceil(len(str(teks)) / 4.0)\n"
    ),
    "keputusan_deployment": (
        "def keputusan_deployment(permintaan_per_hari, token_in, token_out,\n"
        "                         biaya_gpu_per_bulan=60.0, hari=30):\n"
        "    api = biaya_bulanan(permintaan_per_hari, token_in, token_out, \"api_mini\", hari)\n"
        "    keputusan = \"lokal\" if api > biaya_gpu_per_bulan else \"api\"\n"
        "    return {\"biaya_api_bulanan\": api, \"biaya_lokal_bulanan\": biaya_gpu_per_bulan,\n"
        "            \"keputusan\": keputusan}\n"
    ),
    "prediksi": (
        "def prediksi(teks, w, b, tok=None, ambang=0.5):\n"
        "    tok = TokenizerMini() if tok is None else tok\n"
        "    x = fitur(teks, tok)\n"
        "    p, _ = forward(x, w, b)\n"
        "    label = LABEL_POS if p >= ambang else \"negatif\"\n"
        "    return {\"label\": label, \"skor\": p, \"yakin\": abs(p - ambang) * 2}\n"
    ),
    "evaluasi_sebelum_sesudah": (
        "def evaluasi_sebelum_sesudah(val, w_sebelum, b_sebelum, w_sesudah, b_sesudah):\n"
        "    s = akurasi(val, w_sebelum, b_sebelum)\n"
        "    e = akurasi(val, w_sesudah, b_sesudah)\n"
        "    return {\"sebelum\": s, \"sesudah\": e, \"delta\": e - s}\n"
    ),
    "kebijakan_ood": (
        "def kebijakan_ood(hasil_pred, skor_min=0.6, jawaban_aman='kurang yakin, mohon periksa'):\n"
        "    if hasil_pred[\"yakin\"] < skor_min:\n"
        "        return {**hasil_pred, \"tampilkan\": False, \"aksi\": jawaban_aman}\n"
        "    return {**hasil_pred, \"tampilkan\": True, \"aksi\": \"tampilkan label\"}\n"
    ),
    "laporan_eval": (
        "def laporan_eval(val, w, b, ood_hasil=None, skor_min=0.6):\n"
        "    ood = kebijakan_ood(ood_hasil, skor_min) if ood_hasil else None\n"
        "    salah = []\n"
        "    for x, y in val:\n"
        "        p, _ = forward(x, w, b)\n"
        "        pred = 1.0 if p >= 0.5 else 0.0\n"
        "        if pred != y:\n"
        "            salah.append({\"p\": round(p, 4), \"y\": y})\n"
        "    return {\"akurasi\": akurasi(val, w, b), \"salah\": salah, \"ood\": ood}\n"
    ),
}


def nama_ref(src):
    """Nama REF yang didefinisikan di cell ini (fungsi)."""
    return [n for n in REF if f"def {n}(" in src]


ctx = {"__name__": "__main__", "__builtins__": __builtins__}

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
            exec(REF[n], ctx)
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
