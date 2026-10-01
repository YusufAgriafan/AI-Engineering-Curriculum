"""Generator notebook kuis Bab 13 (Fine-tuning & Open-Weight Models) — menulis 02_kuis_finetuning.ipynb.

Pola sama dengan _gen_quiz.py Bab 12: 10 soal (6 PG + 4 coding), 22 poin,
penilaian otomatis, mandiri (data mini ftkit disediakan di cell setup —
tanpa jaringan, tanpa API key, tanpa GPU). Semua angka terkunci SEED 13
(konsisten dengan project-ftkit: tests/ + _calib.py).
"""
import json
from pathlib import Path

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [source]})


def code(source):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [source]})


md("""# 📝 Kuis Bab 13 — Fine-tuning & Open-Weight Models

10 soal (6 pilihan ganda + 4 coding), total **22 poin**. Penilaian otomatis di cell terakhir.

**Aturan:** kerjakan tanpa membuka materi/lab. Soal coding diisi di bawah `# TODO`,
lalu jalankan cell penilaian. Kuis ini **mandiri** — dataset mini (identik dengan
`ftkit/data.py`) sudah disediakan di cell setup: tanpa jaringan, tanpa API key,
tanpa GPU. Semua angka terkunci karena semuanya deterministik (SEED 13): formula
yang sama → angka yang sama, di komputer siapa pun.""")

md("""## Setup (GIVEN) — Dataset Mini ftkit

Identik dengan `project-ftkit/ftkit/data.py` (JANGAN diubah):

- `FORMAT_SFT` — 8 contoh SFT ala Alpaca, **sengaja timpang: 5 positif / 3 negatif**
  (bahan latihan oversample);
- `HARGA` — tarif API per 1 juta token (`lokal` = open-weight di GPU sendiri,
  biaya token nol);
- `VARIAN_KUANTISASI` — fp16 / int8 / q4 dengan skor eval pada task yang sama.
"""
)

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

print('setup siap —', len(FORMAT_SFT), 'contoh SFT (',
      sum(1 for b in FORMAT_SFT if b['respons'] == LABEL_POS), 'positif /',
      sum(1 for b in FORMAT_SFT if b['respons'] == LABEL_NEG), 'negatif )')""")

md("""## Bagian A — Pilihan Ganda (isi `JAWABAN`)""")

code("""# Isi jawaban pilihan ganda di sini, contoh: JAWABAN[1] = 'a'

JAWABAN = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
}""")

md("""### Soal 1 — Kapan fine-tune layak

Chatbot support timmu: prompt engineering + RAG mentok di akurasi **0,72**
(target 0,90). Ada **20.000 contoh berlabel**, tugasnya sempit (klasifikasi
intent + jawaban template). Keputusan yang PALING tepat:

- a) Fine-tune model API proprietary dengan seluruh data — yang penting modelnya besar
- b) Fine-tune open-weight kecil (LoRA) pada data berlabel itu, lalu buktikan dengan
  evaluasi sebelum/sesudah — prompting sudah mentok + tugas sempit + data cukup =
  kondisi klasik di mana fine-tune menang, dengan biaya & latensi jauh di bawah API besar
- c) Tambah contoh few-shot ke prompt sampai 100 contoh — prompt tidak ada batas
- d) Fine-tune seluruh 70 miliar parameter dari awal — full fine-tune pasti terbaik""")

md("""### Soal 2 — Hitungan param LoRA

Adapter LoRA **rank 16** disisipkan di antara dua matriks bobot **4096 × 4096**
yang dibekukan. Berapa jumlah parameter terlatih dan rasionya terhadap full
fine-tune lapisan itu?

- a) 16.777.216 — 100% (rank menggandakan seluruh matriks)
- b) 256 — 0,0015% (hanya rank-nya)
- c) 131.072 — 0,78% (param = rank × (in + out), bukan rank × in × out)
- d) 8.192 — 0,05% (hanya in + out)""")

md("""### Soal 3 — Breakeven API vs GPU lokal

Klasifier mini dilayani via **API mini**: $0,00033 per permintaan (1k token in +
300 token out). Alternatif: **GPU lokal $60/bulan**, biaya token marginal nol.
Di atas kira-kira berapa **permintaan/hari** GPU lokal jadi lebih murah?

- a) 181.818/hari — $60 dibagi $0,00033 (lupa proyeksi sebulan)
- b) 2.400/hari — cukup amortisasi seminggu
- c) ≈ 6.061/hari — di titik itu biaya API bulanan ($0,00033 × req × 30) melampaui $60
- d) Tidak pernah — API selalu lebih murah dari GPU sendiri""")

md("""### Soal 4 — Early stopping: bobot mana yang dipakai

Training dengan `sabar=5`: val loss **minimum tercatat di epoch 61**, loop
berhenti di **epoch 66**. Bobot mana yang harus dipakai untuk serving, dan mengapa?

- a) Epoch 66 — bobot terakhir yang paling "matang"
- b) Epoch 1 — di awal val loss masih turun paling cepat
- c) Rata-rata bobot epoch 61–66 supaya lebih stabil
- d) Epoch 61 — restore bobot TERBAIK (val loss minimum); early stopping tanpa
  restore = berhenti di waktu yang tepat dengan bobot yang salah""")

md("""### Soal 5 — Trade-off kuantisasi

Tabel varian project (model 7B): **fp16** 14 GB / skor 0,82 · **int8** 7 GB /
0,81 · **q4** 3,5 GB / 0,78. Kesimpulan yang BENAR:

- a) q4 selalu pilihan terbaik — paling kecil, paling murah
- b) Skor turun linear terhadap ukuran — kompresi 2× selalu kehilangan poin yang sama
- c) Ukuran turun linear (rasio = 32/bit) tapi skor TIDAK linear: fp16→int8 hampir
  gratis (−0,01), int8→q4 mulai terasa (−0,03) — ukur per task, jangan asumsi
- d) Kuantisasi tidak memengaruhi kualitas model sama sekali""")

md("""### Soal 6 — Skema dataset & oversampling

`FORMAT_SFT` berisi 8 contoh: **5 positif / 3 negatif** (sengaja timpang).
`oversample(FORMAT_SFT, per_label=6)` yang benar menghasilkan:

- a) Tetap 8 baris — oversample justru menghapus duplikat
- b) 12 baris (6/6) — label minoritas diduplikasi berulang dari awal daftarnya;
  urutan ASLI dipertahankan di depan, duplikat menempel di belakang
- c) 16 baris (8/8) — semua baris diduplikasi dua kali
- d) 12 baris, tapi digrup per label dulu supaya batch "rapi\"""")

md("""## Bagian B — Coding""")

md("""### Soal 7 — `ke_contoh_sft(baris)` (2 poin)

`ke_contoh_sft(baris)` → **tuple (prompt, respons)** dengan format prompt
terkunci (ala Alpaca, dipakai trainer):

```
{instruksi}\\n\\nInput: {input}\\n\\nJawaban:
```

Contoh terkunci: `ke_contoh_sft(FORMAT_SFT[0])` →

```
Klasifikasikan sentimen ulasan berikut. Jawab HANYA satu kata: positif atau negatif.

Input: Barang bagus, pengiriman cepat

Jawaban:
```

dengan respons `'positif'`.""")

code("""def ke_contoh_sft(baris):
    # TODO
    pass

# print(repr(ke_contoh_sft(FORMAT_SFT[0])))""")

md("""### Soal 8 — `oversample(baris_list, per_label)` (5 poin)

Samakan jumlah contoh per label `respons` dengan **duplikasi deterministik**:

- urutan **ASLI dipertahankan** — duplikat menempel di belakang (menggrup per
  label dulu mengubah distribusi batch — ini yang sering salah!);
- minoritas diduplikasi **berulang dari awal daftarnya** sampai kuota terpenuhi;
- fungsi murni: `baris_list` tidak boleh berubah.

Angka terkunci (`per_label=6`): hasil **12 baris (6/6)**; 8 baris pertama =
`FORMAT_SFT` utuh; duplikat ke-1/2/3: `'Barang bagus, pengiriman cepat'`,
`'Kualitas jelek, saya menyesal'`, `'Aplikasinya ribet dan lambat'`.""")

code("""def oversample(baris_list, per_label=4):
    # TODO
    pass

# ov = oversample(FORMAT_SFT, per_label=6)
# print(len(ov), [b['input'] for b in ov[8:]])""")

md("""### Soal 9 — `kuantisasi_simulasi(bobot, bit)` + `laju_kompresi` + `ukuran_model_gb` (5 poin)

**Kuantisasi simetris** (simulasi tanpa GPU):

- skala = `max|w| / (2^(bit-1) - 1)`;
- `q = round(w / skala)`, dibatasi rentang `[-levels, levels]`;
- dekuanti: `q * skala` → list float BARU (input tak berubah);
- `bit >= 32` → salinan identik; skala 0 (semua bobot 0) → salinan identik.

**`laju_kompresi(bit)`** = `32 / bit`. **`ukuran_model_gb(param_miliar, bit)`**
= `param_miliar * bit / 8`.

Angka terkunci untuk `w = [0.6, -0.3, 0.15, 0.05, -0.5]` (bulat 6 desimal):

- bit 8 → `[0.6, -0.302362, 0.151181, 0.051969, -0.500787]`
- bit 4 → `[0.6, -0.342857, 0.171429, 0.085714, -0.514286]`
- bit 2 → `[0.6, 0.0, 0.0, 0.0, -0.6]` (kekasaran ekstrem)
- `ukuran_model_gb(7.0, 16)` = 14.0; `ukuran_model_gb(7.0, 4)` = 3.5""")

code("""def kuantisasi_simulasi(bobot, bit):
    # TODO
    pass


def laju_kompresi(bit):
    # TODO
    pass


def ukuran_model_gb(param_miliar, bit):
    # TODO
    pass

# w = [0.6, -0.3, 0.15, 0.05, -0.5]
# print([round(v, 6) for v in kuantisasi_simulasi(w, 8)])
# print([round(v, 6) for v in kuantisasi_simulasi(w, 2)])
# print(ukuran_model_gb(7.0, 16), ukuran_model_gb(7.0, 4))""")

md("""### Soal 10 — `kebijakan_ood(hasil_pred, skor_min=0.6, jawaban_aman=...)` (4 poin)

Gate OOD (sambungan Bab 12: abstain). `hasil_pred` = dict minimal
`{'label', 'skor', 'yakin'}` (yakin = jarak ke 0,5, sisi mana pun):

- `yakin < skor_min` → kembalikan **dict BARU** dengan
  `tampilkan = False` dan `aksi = jawaban_aman`
  (default `'kurang yakin, mohon periksa'`) — jangan tampilkan label yang
  tidak dipercaya modelnya sendiri;
- selain itu → dict baru dengan `tampilkan = True`, `aksi = 'tampilkan label'`;
- batas **inklusif**: `yakin == skor_min` lolos (tampilkan);
- input **tidak boleh berubah**.

Angka terkunci (contoh OOD project, teks "Lumayan standar saja sih"):
skor 0,4988 → yakin 0,0024 → `tampilkan False`. Sebaliknya yakin 0,6 tepat →
lolos.""")

code("""def kebijakan_ood(hasil_pred, skor_min=0.6, jawaban_aman='kurang yakin, mohon periksa'):
    # TODO
    pass

# h = {'label': 'negatif', 'skor': 0.4988, 'yakin': 0.0024}
# print(kebijakan_ood(h))
# print(kebijakan_ood({'label': 'positif', 'skor': 0.97, 'yakin': 0.94}))""")

md("""## 🧮 Penilaian Otomatis""")

code("""total_bobot = 22


def jalankan_kuis():
    skor = 0
    rincian = []

    def cek_pilihan(nomor, kunci, bobot=1):
        nonlocal skor
        benar = JAWABAN.get(nomor) == kunci
        skor += bobot if benar else 0
        rincian.append((nomor, benar))

    def cek_kode(nomor, uji, bobot=1):
        nonlocal skor
        benar = False
        try:
            benar = bool(uji())
        except Exception as e:
            print(f'  soal {nomor}: error -> {type(e).__name__}: {e}')
        skor += bobot if benar else 0
        rincian.append((nomor, benar))

    def _safe(nama, *args, **kwargs):
        fn = globals().get(nama)
        if fn is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    cek_pilihan(1, 'b')
    cek_pilihan(2, 'c')
    cek_pilihan(3, 'c')
    cek_pilihan(4, 'd')
    cek_pilihan(5, 'c')
    cek_pilihan(6, 'b')

    # Soal 7 - ke_contoh_sft
    def uji7():
        hasil = _safe('ke_contoh_sft', FORMAT_SFT[0])
        assert hasil is not None, 'ke_contoh_sft belum jadi'
        prompt, resp = hasil
        assert prompt == ('Klasifikasikan sentimen ulasan berikut. Jawab HANYA satu kata:'
                          ' positif atau negatif.\\n\\nInput: Barang bagus, pengiriman cepat'
                          '\\n\\nJawaban:'), 'format prompt harus persis seperti kontrak'
        assert resp == 'positif'
        for b in FORMAT_SFT:
            p, r = _safe('ke_contoh_sft', b)
            assert p.startswith(b['instruksi'])
            assert ('Input: ' + b['input']) in p
            assert p.endswith('Jawaban:')
            assert r == b['respons']
        return True
    cek_kode(7, uji7, bobot=2)

    # Soal 8 - oversample
    def uji8():
        ov = _safe('oversample', FORMAT_SFT, per_label=6)
        assert ov is not None, 'oversample belum jadi'
        assert len(ov) == 12, len(ov)
        assert sum(1 for b in ov if b['respons'] == LABEL_POS) == 6
        assert sum(1 for b in ov if b['respons'] == LABEL_NEG) == 6
        assert ov[:8] == list(FORMAT_SFT), 'urutan asli harus dipertahankan'
        assert ov[8]['input'] == 'Barang bagus, pengiriman cepat'
        assert ov[9]['input'] == 'Kualitas jelek, saya menyesal'
        assert ov[10]['input'] == 'Aplikasinya ribet dan lambat'
        ov5 = _safe('oversample', FORMAT_SFT, per_label=5)
        assert len(ov5) == 10 and sum(1 for b in ov5 if b['respons'] == LABEL_POS) == 5
        sebelum = [dict(b) for b in FORMAT_SFT]
        _safe('oversample', FORMAT_SFT, per_label=6)
        assert FORMAT_SFT == sebelum, 'input tidak boleh berubah'
        return True
    cek_kode(8, uji8, bobot=5)

    # Soal 9 - kuantisasi + ukuran
    def uji9():
        w = [0.6, -0.3, 0.15, 0.05, -0.5]
        q8 = _safe('kuantisasi_simulasi', w, 8)
        assert q8 is not None, 'kuantisasi_simulasi belum jadi'
        assert [round(v, 6) for v in q8] == [0.6, -0.302362, 0.151181, 0.051969, -0.500787]
        assert [round(v, 6) for v in _safe('kuantisasi_simulasi', w, 4)] == \\
            [0.6, -0.342857, 0.171429, 0.085714, -0.514286]
        assert [round(v, 6) for v in _safe('kuantisasi_simulasi', w, 2)] == [0.6, 0.0, 0.0, 0.0, -0.6]
        assert _safe('kuantisasi_simulasi', w, 32) == list(w)
        assert _safe('kuantisasi_simulasi', [0.0, 0.0], 8) == [0.0, 0.0]
        assert w == [0.6, -0.3, 0.15, 0.05, -0.5], 'input tidak boleh berubah'
        assert _safe('laju_kompresi', 16) == 2.0 and _safe('laju_kompresi', 4) == 8.0
        assert _safe('ukuran_model_gb', 7.0, 16) == 14.0
        assert _safe('ukuran_model_gb', 7.0, 4) == 3.5
        return True
    cek_kode(9, uji9, bobot=5)

    # Soal 10 - kebijakan_ood
    def uji10():
        rendah = {'label': 'negatif', 'skor': 0.4988, 'yakin': 0.0024}
        out = _safe('kebijakan_ood', rendah)
        assert out is not None, 'kebijakan_ood belum jadi'
        assert out is not rendah, 'harus mengembalikan dict BARU'
        assert out['tampilkan'] is False
        assert out['aksi'] == 'kurang yakin, mohon periksa'
        assert 'tampilkan' not in rendah, 'input tidak boleh berubah'

        yakin = {'label': 'positif', 'skor': 0.97, 'yakin': 0.94}
        out2 = _safe('kebijakan_ood', yakin)
        assert out2['tampilkan'] is True and out2['aksi'] == 'tampilkan label'

        tepat = {'label': 'negatif', 'skor': 0.2, 'yakin': 0.6}
        out3 = _safe('kebijakan_ood', tepat)              # yakin == skor_min: lolos
        assert out3['tampilkan'] is True

        custom = _safe('kebijakan_ood', tepat, 0.9, 'eskalasi ke manusia')
        assert custom['tampilkan'] is False and custom['aksi'] == 'eskalasi ke manusia'
        return True
    cek_kode(10, uji10, bobot=4)

    print('=' * 46)
    for nomor, benar in rincian:
        print(f'  Soal {nomor:>2}: {"BENAR" if benar else "salah"}')
    print('=' * 46)
    print(f'SKOR AKHIR: {skor}/{total_bobot}')
    if skor == total_bobot:
        print('Sempurna! Lanjut ke project starter ftkit.')
    elif skor >= 17:
        print('Bagus! Review soal yang salah, lalu lanjut.')
    else:
        print('Ulangi bagian terkait di lab, coba lagi besok.')


jalankan_kuis()""")

md("""---

Sudah mencoba serius? Bandingkan pendekatanmu dengan
`03_kunci_jawaban_kuis_finetuning.ipynb` — fokus pada **kenapa**,
bukan sekadar jawaban benar/salah.""")

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent / "02_kuis_finetuning.ipynb"
out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"OK menulis {out} — {len(CELLS)} cells")
