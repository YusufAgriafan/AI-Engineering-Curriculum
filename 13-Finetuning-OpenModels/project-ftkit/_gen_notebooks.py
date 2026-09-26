"""Generate starter.ipynb & solusi.ipynb untuk project Bab 13 (ftkit)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines]}


def tulis(nama, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = BASE / nama
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out, f"({len(cells)} cells)")


SETUP = [
    "import sys",
    "from pathlib import Path",
    "",
    "# Tambahkan root project ke path (ftkit ada di ./ftkit)",
    "_akar = Path.cwd()",
    "while _akar != _akar.parent and not (_akar / 'ftkit' / 'data.py').is_file():",
    "    _akar = _akar.parent",
    "sys.path.insert(0, str(_akar))",
    "",
    "from ftkit.data import (DATA_SENTIMEN, DATA_OOD, FORMAT_SFT, VARIAN_KUANTISASI,",
    "                        TEMPLATE_INSTRUKSI)",
    "from ftkit.tokenizer import TokenizerMini, bangun_vocab, fitur",
    "from ftkit.model import inisialisasi, forward, loss_batch, akurasi",
    "",
    "print('SEED 13. Dataset sentimen:', len(DATA_SENTIMEN), 'contoh (16 train + 8 val)')",
    "print('format SFT   :', len(FORMAT_SFT), 'baris JSONL ala Alpaca')",
    "print('contoh OOD   :', repr(DATA_OOD['teks']), '—', DATA_OOD['catatan'])",
]

BLOK1_MD = [
    "## Blok 1 — Dataset SFT: Format JSONL ala Alpaca",
    "",
    "Fine-tuning instruksi (SFT) memakai pasangan **instruksi → respons** dalam JSONL.",
    "Dataset sengaja TIDAK seimbang (6 positif / 2 negatif) — masalah nyata yang",
    "kamu perbaiki dengan **oversampling** deterministik (Blok 1b).",
]

BLOK1_CODE = [
    "from ftkit.sft import ke_contoh_sft, tulis_jsonl, muat_jsonl, oversample, split_sft",
    "from collections import Counter",
    "from pathlib import Path",
    "import tempfile",
    "",
    "prompt, respons = ke_contoh_sft(FORMAT_SFT[0])",
    "print(prompt)",
    "print('--- respons:', repr(respons))",
    "",
    "# Round-trip JSONL",
    "with tempfile.TemporaryDirectory() as d:",
    "    p = Path(d) / 'sft.jsonl'",
    "    tulis_jsonl(p, FORMAT_SFT)",
    "    balik = muat_jsonl(p)",
    "print('round-trip JSONL:', len(balik), 'baris —', 'OK' if balik == FORMAT_SFT else 'BEDA!')",
]

BLOK1B_CODE = [
    "print('sebelum :', Counter(b['respons'] for b in FORMAT_SFT))",
    "hasil = oversample(FORMAT_SFT, per_label=6)",
    "print('sesudah :', Counter(b['respons'] for b in hasil), '| total', len(hasil))",
    "",
    "train_s, val_s = split_sft(FORMAT_SFT, rasio_val=0.25)",
    "print('split interleaved 8 contoh rasio 0.25 →', (len(train_s), len(val_s)), '(terkunci 5/3)')",
]

BLOK2_MD = [
    "## Blok 2 — Matematika LoRA: ΔW = B @ A",
    "",
    "LoRA membekukan bobot base (W) dan melatih adapter kecil: ΔW = B @ A dengan",
    "A: rank×dim_in dan B: dim_out×rank. **B diinisialisasi NOL** → ΔW awal 0 →",
    "model mulai dari base utuh. Param terlatih = rank×(dim_in+dim_out),",
    "BUKAN dim_in×dim_out.",
    "",
    "Angka terkunci: 4096×4096 dengan rank 16 → 131.072 param dari 16,7 jt (0,78%).",
]

BLOK2_CODE = [
    "from ftkit.lora import buat_lora, delta_w, hitung_param, rasio_param, kontribusi_adapter",
    "",
    "ad = buat_lora(4096, 4096, 16)",
    "print('param terlatih :', hitung_param(ad))",
    "print('param full     :', 4096*4096)",
    "print('rasio          :', rasio_param(4096, 4096, 16), '→ %.2f%%' % (rasio_param(4096, 4096, 16)*100))",
    "",
    "print('ΔW awal nol?   :', all(v == 0.0 for row in delta_w(ad) for v in row))",
    "",
    "ad_kecil = buat_lora(2, 1, 1)",
    "ad_kecil['A'] = [[1.0, 2.0]]",
    "ad_kecil['B'] = [[3.0]]",
    "print('kontribusi (B@(A@x))[0] untuk x=[1,2]:', kontribusi_adapter([1.0, 2.0], ad_kecil), '(=15)')",
]

BLOK3_MD = [
    "## Blok 3 — Gradient Check: Gradien Analitik vs Numerik",
    "",
    "Sebelum mempercayai loop training, buktikan gradien analitiknya benar —",
    "bandingkan dengan selisih-hitungan (central difference). Ini kebiasaan dari",
    "Bab 4 (GradientTape) yang tetap berlaku di skala besar.",
]

BLOK3_CODE = [
    "from ftkit.lora import gradient_check",
    "",
    "maks = gradient_check()",
    "print('selisih maksimum analitik vs numerik:', maks)",
    "assert maks < 1e-6",
    "print('✅ gradien analitik benar — loop training layak dipercaya')",
]

BLOK4_MD = [
    "## Blok 4 — Loop SFT: Loss Turun, Akurasi Naik",
    "",
    "SGD sungguhan di atas loss BCE. Angka terkunci (lr=0.5, 60 epoch):",
    "train loss 0.7396 → 0.0872, val akurasi 1.0. Loss epoch = rata-rata p SEBELUM",
    "update tiap contoh.",
]

BLOK4_CODE = [
    "from ftkit.trainer import dataset_ftkit, epoch_satu, latih",
    "",
    "data = dataset_ftkit()",
    "print('train:', len(data['train']), '| val:', len(data['val']))",
    "",
    "h = latih(data['train'], data['val'], lr=0.5, epochs=60, sabar=6)",
    "r = h['riwayat']",
    "for ep in (0, 1, 4, 9, 29, 59):",
    "    print('epoch %2d: train %.4f | val %.4f | acc %.3f'",
    "          % (ep+1, r['train_loss'][ep], r['val_loss'][ep], r['val_acc'][ep]))",
    "",
    "assert abs(r['train_loss'][0] - 0.7396) < 1e-3",
    "assert r['val_acc'][-1] == 1.0",
    "print('✅ model tiny terlatih — tanpa GPU, tanpa API')",
]

BLOK5_MD = [
    "## Blok 5 — Early Stopping: Data Kontradiktif → Overfit",
    "",
    "Dua label train SENGAJA dibalik (data kontradiktif) + lr tinggi (3.0):",
    "val loss terbaik di epoch **61**, berhenti dini di **66** (sabar=5).",
    "Bobot yang dipulihkan = bobot TERBAIK, bukan terakhir — persis early stopping",
    "yang kamu kenal dari Bab 4.",
]

BLOK5_CODE = [
    "train_noisy = [(x, (1.0 - y) if i in (6, 14) else y) for i, (x, y) in enumerate(data['train'])]",
    "h2 = latih(train_noisy, data['val'], lr=3.0, epochs=100, sabar=5)",
    "r2 = h2['riwayat']",
    "",
    "print('berhenti dini :', h2['berhenti_dini'])",
    "print('epoch dijalankan:', h2['epochs_dijalankan'], '| best epoch:', h2['best_epoch'])",
    "print('val loss best : %.4f | val loss akhir: %.4f'",
    "      % (r2['val_loss'][h2['best_epoch']-1], r2['val_loss'][-1]))",
    "print('val acc best  : %.3f' % r2['val_acc'][h2['best_epoch']-1])",
    "",
    "assert h2['berhenti_dini'] and h2['best_epoch'] == 61 and h2['epochs_dijalankan'] == 66",
    "print('✅ early stopping bekerja — training lebih lama BUKAN selalu lebih baik')",
]

BLOK6_MD = [
    "## Blok 6 — Kuantisasi: Ukuran vs Kualitas",
    "",
    "FP16 → INT8 → Q4: model 7B menyusut 14 GB → 3,5 GB (kompresi 8x), skor eval",
    "turun 0.82 → 0.78. Selalu **eval sebelum & sesudah** — jangan kuantisasi buta.",
]

BLOK6_CODE = [
    "from ftkit.quantize import kuantisasi_simulasi, bandingkan_varian",
    "",
    "bobot = [0.9, -0.4, 0.15]",
    "for bit in (16, 8, 4):",
    "    q = kuantisasi_simulasi(bobot, bit)",
    "    galat = max(abs(a-b) for a, b in zip(bobot, q))",
    "    print('bit %2d: %s | galat maks %.4f' % (bit, [round(v, 4) for v in q], galat))",
    "",
    "print()",
    "for t in bandingkan_varian():",
    "    print('%4s: %5.1f GB | kompresi %.0fx | skor %.2f'",
    "          % (t['nama'], t['ukuran_gb'], t['kompresi'], t['skor']))",
]

BLOK7_MD = [
    "## Blok 7 — Biaya: API vs Lokal (Angka, Bukan Gengsi)",
    "",
    "Angka terkunci (1k in / 300 out): api_besar $0.0055 | api_mini $0.00033 |",
    "lokal $0. Proyeksi bulanan menentukan keputusan — bukan hype \"self-host\".",
]

BLOK7_CODE = [
    "from ftkit.api import biaya_api, biaya_bulanan, keputusan_deployment",
    "",
    "print('per panggilan (1k in/300 out):')",
    "for m in ('api_besar', 'api_mini', 'lokal'):",
    "    print('  %-9s $%.5f' % (m, biaya_api(1000, 300, m)))",
    "",
    "for req in (200, 2000, 10000):",
    "    d = keputusan_deployment(req, 1000, 300)",
    "    print('%6d req/hari → API $%8.2f/bln vs GPU $60 → %s'",
    "          % (req, d['biaya_api_bulanan'], d['keputusan']))",
    "",
    "assert keputusan_deployment(200, 1000, 300)['keputusan'] == 'api'",
    "assert keputusan_deployment(10000, 1000, 300)['keputusan'] == 'lokal'",
    "print('✅ keputusan deployment = fungsi volume, bukan selera')",
]

BLOK8_MD = [
    "## Blok 8 — Evaluasi End-to-End + Model yang Tahu Dia Tidak Yakin",
    "",
    "Angka terkunci: akurasi 0.75 → **1.0** setelah fine-tune. Dan contoh OOD",
    "(\"Lumayan standar saja sih\") → skor 0.4988, yakin 0.0024 → **jangan tampilkan**.",
    "Sambungan Bab 12: model yang tahu dia tidak yakin = dasar abstain.",
]

BLOK8_CODE = [
    "from ftkit.evaluasi import prediksi, evaluasi_sebelum_sesudah, kebijakan_ood, laporan_eval",
    "from ftkit.model import inisialisasi",
    "",
    "w0, b0 = inisialisasi()",
    "ev = evaluasi_sebelum_sesudah(data['val'], w0, b0, h['w'], h['b'])",
    "print('akurasi sebelum :', ev['sebelum'])",
    "print('akurasi sesudah :', ev['sesudah'], '| delta:', ev['delta'])",
    "",
    "p_ood = prediksi(DATA_OOD['teks'], h['w'], h['b'])",
    "print('OOD:', repr(DATA_OOD['teks']), '→ skor %.4f, yakin %.4f' % (p_ood['skor'], p_ood['yakin']))",
    "keputusan = kebijakan_ood(p_ood)",
    "print('keputusan       : tampilkan', keputusan['tampilkan'], '—', keputusan['aksi'])",
    "",
    "assert ev == {'sebelum': 0.75, 'sesudah': 1.0, 'delta': 0.25}",
    "assert not keputusan['tampilkan']",
    "print('✅ fine-tune terukur + OOD tertahan — bukan \"rasanya lebih pintar\"')",
]

ANALISIS = [
    "## 🔍 Pertanyaan Analisis (bagian penilaian!)",
    "",
    "1. Rasio param LoRA 0,78% — mana yang lebih mahal di produksi: menyimpan 1",
    "   adapter besar per klien, atau 100 adapter kecil? Kapan rank perlu dinaikkan?",
    "2. Early stopping berhenti di epoch 66 dari 100. Apa yang terjadi pada val loss",
    "   setelah epoch 61, dan mengapa bobot TERBAIK (bukan terakhir) yang dipulihkan?",
    "3. Oversampling menduplikasi data minoritas. Risiko apa yang muncul bila data",
    "   diduplikasi terlalu banyak, dan alternatif apa selain duplikasi?",
    "4. Kuantisasi Q4 memangkas 14 GB → 3,5 GB dengan skor turun 0.82 → 0.78.",
    "   Untuk kasus apa kompromi itu layak, dan kapan TIDAK layak?",
    "5. Keputusan API vs lokal berubah di sekitar ~2.400 req/hari (dengan asumsi",
    "   token & tarif di modul). Faktor nyata apa yang bisa menggeser ambang itu?",
    "6. Generator/klasifier ini tahu dia tidak yakin via skor. LLM open-weight yang",
    "   di-fine-tune tidak punya satu angka 'yakin' — bagaimana cara mendeteksi",
    "   ketidakpastian di LLM nyata, dan lapisan mana yang tetap wajib (Bab 12 & 15)?",
]

# ============================== STARTER ==============================
starter = [md(
    "# 🏗️ Starter — project-ftkit (Bab 13: Fine-tuning & Open-Weight Models)",
    "",
    "> Notebook eksperimen. Isi semua cell **TODO** (kontrak lengkap ada di `tests/`),",
    "> jalankan blok eksperimen, lalu jawab **6 Pertanyaan Analisis** di bagian akhir.",
    "",
    "Mode kerja yang disarankan (TDD):",
    "",
    "1. Baca `tests/` → isi satu modul `ftkit/` → `python -m unittest discover -s tests -v`",
    "2. Urutan: `sft.py` → `lora.py` → `quantize.py` → `trainer.py` → `api.py` → `evaluasi.py`",
    "3. Kembali ke sini, jalankan 8 blok (angka terkunci), jawab pertanyaan analisis.",
    "4. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi.ipynb` / `solusi/ftkit_ref.py`.",
)]
starter += [md(*BLOK1_MD), code(*SETUP), code(*BLOK1_CODE), code(*BLOK1B_CODE)]
starter += [md(*BLOK2_MD), code(*BLOK2_CODE)]
starter += [md(*BLOK3_MD), code(*BLOK3_CODE)]
starter += [md(*BLOK4_MD), code(*BLOK4_CODE)]
starter += [md(*BLOK5_MD), code(*BLOK5_CODE)]
starter += [md(*BLOK6_MD), code(*BLOK6_CODE)]
starter += [md(*BLOK7_MD), code(*BLOK7_CODE)]
starter += [md(*BLOK8_MD), code(*BLOK8_CODE)]
starter += [md(*ANALISIS)]
tulis("starter.ipynb", starter)

# ============================== SOLUSI ==============================
solusi = [md(
    "# ✅ Solusi — project-ftkit (Bab 13: Fine-tuning & Open-Weight Models)",
    "",
    "> Notebook ini memuat versi jadi dari seluruh modul TODO (identik dengan",
    "> `solusi/ftkit_ref.py`), lalu seluruh blok eksperimen dijalankan.",
    "> Bandingkan dengan implementasimu — fokus pada **kenapa**, bukan hanya cocok.",
)]
solusi += [md("### Implementasi lengkap (referensi)"),
           md("Kontrak sama dengan `tests/`; angka terkunci berasal dari sini.")]
solusi += [code(
    "import sys",
    "from pathlib import Path",
    "",
    "_akar = Path.cwd()",
    "while _akar != _akar.parent and not (_akar / 'ftkit' / 'data.py').is_file():",
    "    _akar = _akar.parent",
    "sys.path.insert(0, str(_akar))",
    "sys.path.insert(0, str(_akar / 'solusi'))",
    "",
    "import ftkit_ref as ref",
    "from ftkit import api as _am, evaluasi as _em, lora as _lm, quantize as _qm, \\",
    "    sft as _sm, trainer as _tm",
    "",
    "_sm.ke_contoh_sft = ref.ke_contoh_sft",
    "_sm.muat_jsonl = ref.muat_jsonl",
    "_sm.tulis_jsonl = ref.tulis_jsonl",
    "_sm.oversample = ref.oversample",
    "_sm.split_sft = ref.split_sft",
    "_lm.buat_lora = ref.buat_lora",
    "_lm.delta_w = ref.delta_w",
    "_lm.hitung_param = ref.hitung_param",
    "_lm.rasio_param = ref.rasio_param",
    "_lm.kontribusi_adapter = ref.kontribusi_adapter",
    "_lm.gradient_check = ref.gradient_check",
    "_qm.kuantisasi_simulasi = ref.kuantisasi_simulasi",
    "_qm.laju_kompresi = ref.laju_kompresi",
    "_qm.ukuran_model_gb = ref.ukuran_model_gb",
    "_qm.bandingkan_varian = ref.bandingkan_varian",
    "_tm.epoch_satu = ref.epoch_satu",
    "_tm.latih = ref.latih",
    "_tm.dataset_ftkit = ref.dataset_ftkit",
    "_am.biaya_api = ref.biaya_api",
    "_am.biaya_bulanan = ref.biaya_bulanan",
    "_am.token_estimasi = ref.token_estimasi",
    "_am.keputusan_deployment = ref.keputusan_deployment",
    "_em.prediksi = ref.prediksi",
    "_em.evaluasi_sebelum_sesudah = ref.evaluasi_sebelum_sesudah",
    "_em.kebijakan_ood = ref.kebijakan_ood",
    "_em.laporan_eval = ref.laporan_eval",
    "",
    "print('implementasi referensi dimuat (solusi/ftkit_ref.py)')",
)]
solusi += [md(*BLOK1_MD), code(*SETUP), code(*BLOK1_CODE), code(*BLOK1B_CODE)]
solusi += [md(*BLOK2_MD), code(*BLOK2_CODE)]
solusi += [md(*BLOK3_MD), code(*BLOK3_CODE)]
solusi += [md(*BLOK4_MD), code(*BLOK4_CODE)]
solusi += [md(*BLOK5_MD), code(*BLOK5_CODE)]
solusi += [md(*BLOK6_MD), code(*BLOK6_CODE)]
solusi += [md(*BLOK7_MD), code(*BLOK7_CODE)]
solusi += [md(*BLOK8_MD), code(*BLOK8_CODE)]
solusi += [md(
    "---",
    "",
    "Bandingkan dengan implementasimu di `starter.ipynb`. Jika angkamu berbeda,",
    "cari penyebabnya — determinisme adalah fitur: hasil yang sama di komputer",
    "siapa pun adalah syarat evaluasi yang adil.",
)]
tulis("solusi.ipynb", solusi)
