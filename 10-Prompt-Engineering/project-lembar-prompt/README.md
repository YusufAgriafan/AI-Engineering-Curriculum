# 🏗️ Project Starter — Lembar Prompt: dari Prompt Naif ke Prompt Produksi (Bab 10)

> Kamu membangun **perkakas prompt engineering** yang bisa diukur: template aman,
> anatomi prompt berbagian, few-shot yang dipilih berdasarkan kasus tepi, parser
> output berisik, validator skema, guard injeksi, dan **harness evaluasi** yang
> mengubah "prompt-ku kelihatannya bagus" menjadi lima angka. Stdlib murni,
> TDD **157 test**, tanpa API key dan tanpa biaya token.

## Struktur

```
project-lembar-prompt/
├── README.md               ← kamu di sini
├── RUBRIK.md               ← penilaian mandiri
├── starter.ipynb           ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb            ← notebook SOLUSI (buka setelah selesai / mentok)
├── plib/                   ← paket yang kamu isi (stdlib murni)
│   ├── data.py             ← dataset terkunci (JANGAN DIUBAH)
│   ├── sections.py         ← TODO: anatomi prompt, bagian wajib, anggaran token
│   ├── template.py         ← TODO: render {{placeholder}} + escape delimiter
│   ├── fewshot.py          ← TODO: format, pilih, urutkan contoh
│   ├── schema.py           ← TODO: validasi JSON-schema mini
│   ├── parse.py            ← TODO: ekstraksi JSON dari output berisik
│   ├── guard.py            ← TODO: deteksi injeksi + bungkus data
│   ├── mock_llm.py         ← GIVEN: mock LLM deterministik (baca docstring-nya!)
│   └── evaluate.py         ← TODO: harness evaluasi prompt
├── tests/                  ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_sections.py    ( 15 test)
│   ├── test_template.py    ( 19 test)
│   ├── test_fewshot.py     ( 15 test)
│   ├── test_schema.py      ( 19 test)
│   ├── test_parse.py       ( 24 test)
│   ├── test_guard.py       ( 16 test)
│   ├── test_mock_llm.py    ( 25 test)
│   └── test_evaluate.py    ( 24 test)
├── _calib.py               ← skrip audit: mencetak angka terkunci (bukan test)
├── _gen_project_notebooks.py
├── _verify_project.py      ← verifier otomatis (patch referensi + test + notebook)
└── solusi/
    └── plib_ref.py         ← implementasi referensi (untuk cek mandiri)
```

## Kenapa Ada Mock, Bukan API?

Prompt engineering biasanya diajarkan dengan memanggil LLM berbayar: hasilnya
**acak**, mahal, dan tidak bisa di-test. Di sini "model" diganti
`plib/mock_llm.py` — simulator deterministik yang **kepatuhannya ditentukan
kualitas prompt-mu**. Aturan mainnya terbuka (docstring modul), sengaja:

| Fitur prompt | Kalau TIDAK ada di prompt |
|---|---|
| field berskema (`product_name: ...`) | model menjawab prosa, bukan data |
| instruksi "hanya JSON" | output dibungkus basa-basi + code fence |
| kebijakan nilai kosong ("gunakan null") | model **mengarang** nilai yang masuk akal |
| contoh kasus tepi (`"extracted": false`) | model pakai default `extracted=true, confidence=0.5` |
| delimiter `<dokumen>…</dokumen>` + leash | instruksi di dalam data **dituruti** (injeksi sukses) |
| `temperature` rendah | output kotor (basa-basi) makin sering |

Konsekuensinya: **satu-satunya variabel yang mengubah skor adalah prompt-mu.**
Itu justru kondisi ideal untuk belajar — dan semua angka reproducible (seed 10).

> ⚠️ Mock ini alat didaktik, bukan model sungguhan. Yang kamu latih adalah
> **disiplin**: mendefinisikan output, menangani kasus kosong, memisahkan
> instruksi dari data, dan memilih contoh. Disiplin itu langsung pindah ke API
> asli (Bab 11) — plus satu hal yang tidak dilatih mock: pengetahuan dunia.

## Dataset: 6 Kasus Emas + 3 Kasus Injeksi (terkunci)

Enam chat pelanggan dengan jawaban benar yang diketahui, sengaja mencakup semua
kasus sulit: harga disebut/tidak, jumlah disebut/tidak, sapaan kosong
(`extracted: false`), dan estimasi pengiriman. Tiga kasus injeksi menyisipkan
perintah jahat ke dalam data (abaikan instruksi, spoof `System:`, minta data
pelanggan).

Tiga prompt referensi sudah disediakan di `data.py` — itulah tangga yang harus
kamu reproduksi:

```
PROMPT_NAIF        -> field_accuracy 0.0000   (0/36)  injeksi 0/3
PROMPT_TERSTRUKTUR -> field_accuracy 0.5278  (19/36)  injeksi 0/3
PROMPT_PRODUKSI    -> field_accuracy 1.0000  (36/36)  injeksi 3/3
```

## Apa yang Kamu Bangun

1. **Anatomi prompt** — bagian wajib (`ROLE/KONTEKS/TUGAS/FORMAT/CONSTRAINT/INPUT`)
   yang bisa **diperiksa otomatis**, plus anggaran token & pemotongan aman.
2. **Template engine** — `{{placeholder}}` single-pass (kurung kurawal JSON tidak
   disentuh, isi variabel tidak di-render ulang) + escape delimiter anti-breakout.
3. **Few-shot berdisiplin** — pemilihan menyebar merata, pengurutan memakai
   *recency bias* (contoh terberat terakhir), format yang bisa diaudit.
4. **Validator skema** — JSON-schema mini dengan pesan error stabil
   (`missing:`, `type:`, `minimum:`, `extra:`) — bahannya retry loop.
5. **Parser tahan banting** — `ekstrak_json` brace-matching yang sadar string,
   plus `perbaiki_json` untuk kutip tunggal / `True` / trailing comma.
6. **Guard injeksi** — deteksi pola, pembungkusan data, dan uji bahwa
   **delimiter saja tidak cukup, leash saja tidak cukup**.
7. **Harness evaluasi** — lima metrik (`naive_json_rate`, `parse_ok_rate`,
   `exact_rate`, `field_accuracy`, `injeksi_rate`) + `bandingkan` untuk A/B prompt.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_sections.py`.
2. Isi satu modul, jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan yang disarankan: `template.py` → `sections.py` → `fewshot.py` →
   `schema.py` → `parse.py` → `guard.py` → `evaluate.py` sampai **semua 157 test
   hijau**.
4. Buka `starter.ipynb` — jalankan eksperimen (angka terkunci), jawab
   **6 Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/plib_ref.py` / `solusi.ipynb`.

Ingin melihat angka terkunci tanpa menjalankan notebook?

```bash
python _calib.py
```

> 💡 Kalau test macet: cek (1) `render` JANGAN pakai `str.format` (kurung JSON akan
> meledak), (2) substitusi single-pass, (3) `True/False` bukan integer di
> `schema.py`, (4) `ekstrak_json` harus tahan `{` di dalam string, (5) deteksi
> fitur harus **per-nama-field** (jangan satu regex berisi semua alternatif —
> akibatnya semua field dianggap ada). Lima itu penyebab 90% bug di project ini.

## Pertanyaan Analisis (bagian penilaian!)

1. Prompt naif 0/36, tapi konteksnya panjang dan menyebut kata "ekstraksi".
   Apa yang menentukan sebuah prompt **bisa dieksekusi mesin**?
2. V1 menyebut seluruh skema (19/36, 0/6 exact). Sebutkan dua sebab kegagalan
   yang berbeda dan tambalan masing-masing.
3. Ablasi few-shot: k=1 dan k=2 tidak menambah skor, k=3 melompat ke 1.0. Apa
   yang dibeli contoh ke-3, dan bagaimana itu mengubah cara memilih contoh?
4. Prompt produksi 277 token vs naif 24 token. Kapan kamu rela bayar ~11x token?
5. Kenapa delimiter saja dan leash saja sama-sama tidak cukup? Sebut satu
   lapisan pertahanan wajib di produksi yang tidak dimodelkan mock ini.
6. T=0.5 membuat 2/6 output kotor, tapi `exact_rate` tetap 1.0. Apa implikasinya
   bagi pilihan `temperature` dan strategi parsing?

## Prasyarat

```bash
pip install matplotlib      # hanya untuk satu grafik ablasi di notebook
```

> Bab ini memakai kembali pola dari Bab 9 (reproducibility & TDD) dan Bab 6
> (evaluasi berpasangan). Yang diuji bukan kecanggihan model, tapi
> **kemampuan mendefinisikan output dan mempertahankannya** — keterampilan
> nomor satu seorang AI Engineer yang mengirim prompt ke produksi.
