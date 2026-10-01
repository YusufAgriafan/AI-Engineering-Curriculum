"""Generator notebook kunci jawaban kuis Bab 13 — menulis 03_kunci_jawaban_kuis_finetuning.ipynb.

Pola sama dengan _gen_kunci.py Bab 12: jawaban + KENAPA (bukan sekadar
benar/salah), kode referensi tiap soal coding (identik kontrak ftkit_ref.py),
bisa dijalankan penuh (setup sama dengan kuis).
"""
import json
from pathlib import Path

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [source]})


def code(source):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [source]})


md("""# 🔑 Kunci Jawaban — Kuis Bab 13 (Fine-tuning & Open-Weight Models)

> Buka **setelah** mencoba. Fokus pembahasan: **kenapa**, bukan sekadar benar/salah.
> Semua angka terkunci SEED 13 dan konsisten dengan `project-ftkit` — kamu bisa
> menjalankan ulang dan mendapat angka yang sama persis.""")

md("""## Setup (sama dengan kuis — untuk menjalankan kode referensi)""")

code("""TEMPLATE_INSTRUKSI = ("Klasifikasikan sentimen ulasan berikut. "
                      "Jawab HANYA satu kata: positif atau negatif.")

LABEL_POS = "positif"
LABEL_NEG = "negatif"

FORMAT_SFT = [
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Barang bagus, pengiriman cepat", "respons": "positif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Kualitas jelek, saya menyesal", "respons": "negatif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Puas dengan pembelian ini, mantap", "respons": "positif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Aplikasinya ribet dan lambat", "respons": "negatif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Mudah dipasang, hasil rapi", "respons": "positif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Paket datang rusak, kecewa", "respons": "negatif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Pelayanan ramah, sangat puas", "respons": "positif"},
    {"instruksi": TEMPLATE_INSTRUKSI, "input": "Suara bocor, minta ganti, jelek", "respons": "positif"},
]

HARGA = {
    "api_besar": {"input": 2.50, "output": 10.0},
    "api_mini": {"input": 0.15, "output": 0.60},
    "lokal": {"input": 0.0, "output": 0.0},
}

VARIAN_KUANTISASI = [
    {"nama": "fp16", "bit": 16, "skor": 0.82},
    {"nama": "int8", "bit": 8, "skor": 0.81},
    {"nama": "q4", "bit": 4, "skor": 0.78},
]

print('setup siap')""")

md("""## Bagian A — Pilihan Ganda""")

md("""### Soal 1 — Jawaban: **b) fine-tune open-weight kecil (LoRA), buktikan dengan evaluasi sebelum/sesudah**

Ketiga kondisi klasik "fine-tune layak" terpenuhi sekaligus:

1. **Prompting + RAG sudah mentok** (0,72 vs target 0,90) — kapabilitas prompt habis;
2. **Tugas sempit & terdefinisi** (klasifikasi intent + template) — bukan dialog terbuka;
3. **Data berlabel cukup** (20.000 contoh).

Fine-tune open-weight kecil menambah tiga hal yang prompt tak bisa beri:
konsistensi format (jawab SELALU satu kata), gaya/domain, dan latensi-biaya
rendah di traffic besar. Dan keputusan tidak dirasakan — dibuktikan angka:
evaluasi sebelum/sesudah di val yang sama (di project: 0,75 → 1,0, delta +0,25).

Kenapa opsi lain salah:

- **a)** model API besar bukan tujuan; tanpa evaluasi = "rasanya lebih pintar",
  bukan keputusan engineering. Dan biaya per token API besar jauh di atas mini
  ($0,0055 vs $0,00033 per permintaan 1k+300 — 16,7× lipat).
- **c)** 100 few-shot contoh memakan konteks tiap permintaan (biaya + latensi
  linear per request, selamanya) dan tetap tidak menyentuh distribusi data spesifik.
- **d)** full fine-tune 70B tak realistis tanpa cluster — dan full fine-tune
  justru berisiko catastrophic forgetting; LoRA (rank kecil) cukup untuk tugas sempit.""")

md("""### Soal 2 — Jawaban: **c) 131.072 param — 0,78%**

Parameter LoRA = **rank × (dim_in + dim_out)**, karena adapter hanya menyimpan
A (rank × in) dan B (out × rank) — BUKAN matriks ΔW penuh:

```
A: 16 × 4096 =   65.536
B: 4096 × 16 =   65.536
total        =  131.072
penuh        = 4096 × 4096 = 16.777.216
rasio        = 131.072 / 16.777.216 = 0,0078125 = 0,78%
```

Inilah seluruh nilai ekonomis LoRA: gradien + optimizer state (2× param untuk
Adam) hanya untuk 0,78% bobot. Rank 8 → 0,39% — naik rank = naik kapasitas
secara LINEAR, bukan kuadrat, karena param tumbuh terhadap (in + out), bukan in × out.

Kenapa opsi lain salah:

- **a)** itu jumlah bobot beku yang dilapisinya; yang dilatih hanya adapter.
- **b)** rank saja; A dan B masing-masing menyumbang rank × in dan out × rank.
- **d)** hanya in + out × 1 — tidak mengalikan rank.

Jebakan yang diuji di lab (latihan 3.1): **B diinisialisasi NOL** → ΔW = B @ A
= 0 → model awal identik dengan base. Fine-tune harus TIDAK mengubah model
sebelum data pertama; inisialisasi acak pada B merusak model sebelum belajar.""")

md("""### Soal 3 — Jawaban: **c) ≈ 6.061 permintaan/hari**

Hitung biaya dua sisi sebagai fungsi traffic (asumsi 1k in + 300 out, 30 hari):

```
biaya_api_bulanan  = $0,00033 × req × 30
biaya_gpu_bulanan  = $60 (flat, biaya token marginal = 0)

breakeven: 0,00033 × req × 30 = 60
        →  req = 60 / (0,00033 × 30) = 6.060,6 req/hari
```

Verifikasi terkunci: 6.060 req/hari → $59,994 (masih API lebih murah);
6.061 → $60,0039 (GPU lokal menang). Jadi ambang keputusan:
**di bawah ~6.061/hari pakai API, di atasnya GPU lokal**.

Kenapa opsi lain salah:

- **a)** lupa proyeksi 30 hari: $60/0,00033 = 181.818 itu permintaan per BULAN,
  bukan per hari — meleset 30×.
- **b)** 2.400 adalah angka lama dari tarif berbeda; keputusan selalu turun dari
  tarif yang AKTIF dipakai, bukan angka hafalan.
- **d)** API tanpa biaya marginal tidak pernah ada; GPU lokal memang flat —
  di traffic cukup besar flat selalu mengalahkan per-unit.

Dua kehalusan yang sering dilewatkan: (1) GPU lokal punya biaya lain (listrik,
ops, idle time) yang di sini disederhanakan jadi $60 flat; (2) keputusan ini
dinamis — `keputusan_deployment` di project mengevaluasi ulang tiap perubahan tarif/traffic.""")

md("""### Soal 4 — Jawaban: **d) epoch 61 — restore bobot TERBAIK**

Early stopping punya dua keputusan yang terpisah dan sering dirancu:

```
KAPAN berhenti   -> patience "sabar" (di soal: 5 epoch tanpa perbaikan)
BOBOT YANG DIPAKAI -> epoch val loss MINIMUM (di soal: 61, bukan 66)
```

Angka terkunci project (skenario overfit, lr=3.0): best epoch **61**, berhenti
di **66**, val loss best **0,4184** vs akhir **0,4188** — selisihnya kecil di
contoh tiny ini, tapi arahnya konsisten: setelah 61 val loss memburuk (overfit)
dan bobot epoch 66 lebih buruk dari yang tersimpan.

Kenapa opsi lain salah:

- **a)** bobot terakhir = bobot yang telah memburuk 5 epoch tanpa henti; makin
  besar `sabar`, makin besar jaraknya dari best.
- **b)** val loss di epoch 1 hampir selalu belum minimum; "turun paling cepat"
  bukan "terendah".
- **c)** rata-rata bobot (weight averaging) adalah teknik tersendiri (SWA) dengan
  syarat & validasi sendiri — bukan pengganti restore best, dan tidak diuji di sini.

Di project kontrak ini diuji eksplisit: `berhenti_dini=True`, `best_epoch ≠
epochs_dijalankan`, dan bobot yang dikembalikan adalah yang val loss-nya minimum.""")

md("""### Soal 5 — Jawaban: **c) ukuran linear (32/bit), skor TIDAK linear — ukur per task**

Ukuran memang matematika murni: `ukuran_gb = param × bit / 8` → 7B fp16 = 14 GB,
int8 = 7 GB, q4 = 3,5 GB (rasio kompresi 32/bit: 2× / 4× / 8×). Tapi **kualitas
tidak mengikuti ukuran** — pola nyata di tabel project:

```
fp16  14,0 GB  skor 0,82
int8   7,0 GB  skor 0,81   (-0,01 — hampir gratis)
q4     3,5 GB  skor 0,78   (-0,03 dari int8 — mulai terasa)
```

fp16→int8 memotong memori dua kali lipat dengan biaya kualitas 1 poin; int8→q4
hemat dua kali lipat lagi tapi 3 poin. Artinya keputusan kuantisasi = kurva
trade-off yang diukur **pada task-mu sendiri** (per task!), bukan angka universal.

Kenapa opsi lain salah:

- **a)** "selalu" adalah kata yang dilarang: q4 kehilangan 0,04 dari fp16 — untuk
  task di mana 0,78 < target kamu, q4 bukan pilihan.
- **b)** kalau linear, 8× kompresi = kehilangan 4× dari 2× kompresi; faktanya
  fp16→int8 −0,01 vs int8→q4 −0,03 — jelas non-linear.
- **d)** simulasi bit=2 di lab memperlihatkan kegagalan ekstrem:
  `[0.6, -0.3, 0.15, 0.05, -0.5]` → `[0.6, 0, 0, 0, -0.6]` — bobot kecil
  lenyap. Kuantisasi selalu kehilangan sesuatu; pertanyaannya berapa.""")

md("""### Soal 6 — Jawaban: **b) 12 baris (6/6), duplikat di belakang, urutan asli utuh**

Kontrak `oversample` di project (diuji dengan angka terkunci):

```
asli (urut):  [p1, n1, p2, n2, p3, n3, p4, p5]        5 positif / 3 negatif
hasil =       [p1, n1, p2, n2, p3, n3, p4, p5,        <- 8 baris asli UTUH
               p1, n1, n2]                             <- duplikat berulang
               dari awal daftar TIAP label sampai kuota
```

- positif 5→6: duplikat 1 (`ov[8] = 'Barang bagus, pengiriman cepat'`).
- negatif 3→6: duplikat 3 berulang dari awal daftar negatifnya
  (`ov[9] = 'Kualitas jelek, saya menyesal'`, `ov[10] = 'Aplikasinya ribet dan lambat'`).
- Total 12 baris (6/6).

Kenapa opsi lain salah:

- **a)** tanpa oversampling, kelas minoritas (3/8 = 37,5%) dibanjiri mayoritas —
  model bias ke label "positif".
- **c)** menduplikasi kelas mayoritas sama saja menaikkan bobot kelas yang
  sudah menang — kebalikan tujuan.
- **d)** menggrup per label mengubah URUTAN contoh → distribusi batch berubah →
  training tidak lagi bisa direproduksi terhadap kontrak. "Rapi" bukan tujuan;
  deterministik & seimbang yang benar.

Prinsip umumnya: perbaiki ketidakseimbangan dengan **mengubah data, bukan
mengubah urutan** — duplikasi minoritas adalah cara termurah tanpa mengarang data baru.""")

md("""## Bagian B — Coding""")

md("""### Soal 7 — `ke_contoh_sft`

Format prompt = kontrak yang dipakai trainer — mengubahnya berarti melatih
dengan format yang berbeda dari yang akan di-serve. Perhatikan pemisah baris
kosong ganda (`\\n\\n`) dan `Jawaban:` TANPA spasi di akhir (model yang
di-fine-tune belajar melanjutkan tepat setelah titik itu).""")

code("""def ke_contoh_sft(baris):
    return (f"{baris['instruksi']}\\n\\nInput: {baris['input']}\\n\\nJawaban:",
            baris["respons"])

print(repr(ke_contoh_sft(FORMAT_SFT[0])))""")

md("""### Soal 8 — `oversample`

Dua jebakan: (1) **menggrup per label dulu** — mengubah urutan asli dan
distribusi batch; kontraknya urutan asli utuh di depan, duplikat menempel di
belakang; (2) **mutasi input** — fungsi murni: bangun list baru, jangan
`baris_list.append(...)`. Duplikasi "berulang dari awal daftarnya" memakai
modulo (`i % len(kumpulan)`) supaya aman untuk kuota berapa pun.""")

code("""def oversample(baris_list, per_label=4):
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

ov = oversample(FORMAT_SFT, per_label=6)
print(len(ov), 'baris —',
      {r: sum(1 for b in ov if b['respons'] == r) for r in (LABEL_POS, LABEL_NEG)})
print('duplikat:', [b['input'] for b in ov[8:]])
print('urutan asli utuh:', ov[:8] == list(FORMAT_SFT))""")

md("""### Soal 9 — `kuantisasi_simulasi` + `laju_kompresi` + `ukuran_model_gb`

Jebakan yang diuji: (1) **list baru** — kuantisasi tidak boleh mengubah input;
(2) **`bit >= 32` dan skala 0** → salinan identik (bukan crash — skala 0 muncul
saat semua bobot 0); (3) **clamp** ke `[-levels, levels]` sebelum dekuantisasi
(wajib di hardware nyata); (4) di bit rendah bobot kecil **lenyap**: pada
bit=2, `levels = 1` → 0,15 dan 0,05 kehilangan representasi → `[0.6, 0, 0, 0, -0.6]`.
Inilah kenapa skor eval turun di task sensitif-detail.""")

code("""def kuantisasi_simulasi(bobot, bit):
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
    return 32.0 / bit


def ukuran_model_gb(param_miliar, bit):
    return param_miliar * bit / 8.0


w = [0.6, -0.3, 0.15, 0.05, -0.5]
print('bit 8:', [round(v, 6) for v in kuantisasi_simulasi(w, 8)])
print('bit 4:', [round(v, 6) for v in kuantisasi_simulasi(w, 4)])
print('bit 2:', [round(v, 6) for v in kuantisasi_simulasi(w, 2)])
print('7B fp16:', ukuran_model_gb(7.0, 16), 'GB | q4:', ukuran_model_gb(7.0, 4), 'GB')""")

md("""### Soal 10 — `kebijakan_ood`

Sambungan Bab 12: model yang tahu dia tidak yakin = dasar abstain. Kontrak yang
diuji: (1) kembalikan **dict BARU** — input tak berubah (pemanggil mungkin
masih memakainya untuk audit); (2) batas **inklusif** — `yakin == skor_min`
LOLOS (tampilkan); (3) `tampilkan=False` TANPA menimpa label — label tetap ada
di dict, hanya disembunyikan dari pengguna bersama aksi eskalasi. Angka
terkunci project: teks OOD "Lumayan standar saja sih" → skor 0,4988 →
yakin 0,0024 → disembunyikan.""")

code("""def kebijakan_ood(hasil_pred, skor_min=0.6, jawaban_aman="kurang yakin, mohon periksa"):
    if hasil_pred["yakin"] < skor_min:
        return {**hasil_pred, "tampilkan": False, "aksi": jawaban_aman}
    return {**hasil_pred, "tampilkan": True, "aksi": "tampilkan label"}

h = {'label': 'negatif', 'skor': 0.4988, 'yakin': 0.0024}
out = kebijakan_ood(h)
print(out)
print('yakin 0.6 tepat:', kebijakan_ood({'label': 'negatif', 'skor': 0.2, 'yakin': 0.6})['tampilkan'])
print('input utuh:', 'tampilkan' not in h)""")

md("""## Rekap Kunci Pilihan Ganda

| Soal | Kunci | Inti |
|---|---|---|
| 1 | **b** | prompting mentok + tugas sempit + data cukup = fine-tune open-weight, bukti eval |
| 2 | **c** | param LoRA = rank × (in + out); 131.072 = 0,78% |
| 3 | **c** | breakeven 60/(0,00033×30) ≈ 6.061 req/hari |
| 4 | **d** | early stopping → restore bobot BEST (epoch 61), bukan terakhir (66) |
| 5 | **c** | ukuran linear, skor non-linear — ukur per task |
| 6 | **b** | 12 baris (6/6), urutan asli utuh, duplikat di belakang |

Coding: `ke_contoh_sft` (format terkunci) · `oversample` (urutan asli, fungsi murni) ·
`kuantisasi_simulasi` (list baru, clamp, bit≥32 & skala 0 aman) · `kebijakan_ood`
(dict baru, batas inklusif, label disembunyikan bukan ditimpa).""")

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent / "03_kunci_jawaban_kuis_finetuning.ipynb"
out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"OK menulis {out} — {len(CELLS)} cells")
