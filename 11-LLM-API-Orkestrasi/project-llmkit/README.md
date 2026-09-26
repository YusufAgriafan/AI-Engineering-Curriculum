# 🏗️ Project Starter — llmkit: Sistem LLM yang Tahan Gagal (Bab 11)

> Kamu membangun **lapisan produksi** di atas panggilan LLM: hitung biaya sebelum
> memanggil, penjaga anggaran, retry yang membedakan galat transien vs fatal,
> fallback chain multi-provider, cache konten-address ber-TTL, perakit streaming
> (TTFT), function calling dengan validasi argumen, dan orkestrasi pipeline yang
> melaporkan biaya/latensi **per langkah**. Stdlib murni, TDD **107 test**, tanpa
> API key dan tanpa biaya token.

## Struktur

```
project-llmkit/
├── README.md               ← kamu di sini
├── RUBRIK.md               ← penilaian mandiri
├── starter.ipynb           ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb            ← notebook SOLUSI (buka setelah selesai / mentok)
├── llmkit/                 ← paket yang kamu isi (stdlib murni)
│   ├── data.py             ← dataset & tarif terkunci (JANGAN DIUBAH)
│   ├── transport.py        ← GIVEN: provider palsu yang bisa disuruh gagal
│   ├── cost.py             ← TODO: token, biaya, penjaga anggaran
│   ├── retry.py            ← TODO: backoff + retry, transien vs fatal
│   ├── fallback.py         ← TODO: rantai fallback multi-provider
│   ├── cache.py            ← TODO: cache konten-address + TTL + hit-rate
│   ├── stream.py           ← TODO: rakitan chunk streaming & TTFT
│   ├── tools.py            ← TODO: skema, validasi argumen, loop tool
│   └── orchestrate.py      ← TODO: prompt chaining + laporan per langkah
├── tests/                  ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_transport.py   ( 12 test, GIVEN — memverifikasi provider palsunya)
│   ├── test_cost.py        ( 14 test)
│   ├── test_retry.py       ( 15 test)
│   ├── test_fallback.py    (  9 test)
│   ├── test_cache.py       ( 14 test)
│   ├── test_stream.py      ( 10 test)
│   ├── test_tools.py       ( 20 test)
│   └── test_orchestrate.py ( 13 test)
├── _calib.py               ← skrip audit: mencetak angka terkunci (bukan test)
├── _gen_project_notebooks.py
├── _verify_project.py      ← verifier otomatis (patch referensi + test + notebook)
└── solusi/
    └── llmkit_ref.py       ← implementasi referensi (untuk cek mandiri)
```

## Kenapa Ada Provider Palsu, Bukan API?

Pola produksi (retry, fallback, cache, anggaran) **hanya bisa dilatih kalau
kegagalan bisa dijadwalkan**. Kalau kamu memakai API sungguhan:

- `rate_limit` datang kapan saja — kamu tidak bisa menguji percobaan ke-2 tepat;
- latency berubah-ubah — TTFT tidak bisa di-assert;
- tiap percobaan **berbiaya** — termasuk percobaan yang sengaja kamu gagalkan.

`llmkit/transport.py` menghapus ketiga masalah itu sekaligus. Perilakunya TERBUKA
(docstring modul), sengaja:

| Rencana kegagalan | Panggilan ke-1 | ke-2 | ke-3 |
|---|---|---|---|
| `mini` | `ok` | `ok` | `ok` |
| `sedang` | `rate_limit` | `ok` | `ok` |
| `besar` | `rate_limit` | `server_error` | `ok` |

Konsekuensinya: **jam kegagalan = jumlah panggilan per model**. Itu sebabnya
`provider_utama()` mengembalikan objek **baru** (state-nya terpisah) — kalau state
dipakai bersama, urutan kegagalan eksperimenmu bergeser dan angkanya tidak lagi
reproducible. Ada test khusus untuk jebakan ini
(`test_provider_terpisah_punya_keadaan_sendiri`).

> ⚠️ Provider ini alat didaktik, bukan model sungguhan. Yang kamu latih adalah
> **disiplin operasional**: mengklasifikasikan galat, menjadwalkan tunggu,
> memutuskan turun-kualitas, menghitung biaya, dan membatasi loop. Semua itu
> langsung pindah ke API asli — plus dua hal yang tidak dilatih mock: perilaku
> model yang tidak terduga dan rate limit organisasi.

## Angka Terkunci (dari `_calib.py`, seed 11)

```
COST
  token_estimasi              '' -> 0 ; 1/4/5/9 huruf -> [1, 1, 2, 3]
  hitung_biaya(1000,300)      mini -> 0.00033 ; besar -> 0.0095      (~29x)
  anggaran 4 job              diproses 3 | dilewati 1 | berhenti_di 3
                              total 4.005e-05 ; per hasil 1.335e-05

RETRY
  sedang (1x gagal)           attempts 2, tunggu [1000], latency 240 -> total 1240
  mini (langsung ok)          attempts 1, tunggu []
  besar maks_percobaan=2      GalatSemuaPercobaan [(1,'rate_limit'), (2,'server_error')]
  model tak dikenal           GalatPermintaanBuruk (bad_request), 1 panggilan
  backoff peta                [1000, 2000, 4000, 8000, 8000]
  backoff_ms(n,100,3,500)     [100, 300, 500, 500]

FALLBACK
  rantai utama/besar -> utama/mini -> cadangan/murah
                              menang di tautan 1 (mini)
                              dicoba: besar= semua_gagal, mini= success

CACHE (ttl=3, tick 0,1,5,6)   miss, hit, miss, hit
  statistik                   hit 2 | miss 1 | expired 1 | hit_rate 0.5
  kunci('mini', PESAN)        f833386edc2ecf9c

STREAM                        jumlah_chunk 11 | TTFT 120 ms | total 340 ms

TOOLS
  validasi get_cuaca {}       [missing:kota, missing:satuan]
  jalankan_tool gagal         'missing:satuan; type:kota'
  loop 2 tool                 iterasi 3, tool [cari_catatan, hitung]
  loop 1 tool                 iterasi 2, tool [get_cuaca]

PIPELINE (3 langkah)          berhasil 3 | gagal 0 | 1620 ms | $1.6075e-04
  klasifikasi mini  T=0.0     184 ms  1.530e-05
  ekstraksi   mini  T=0.0     187 ms  1.545e-05
  balasan     sedang T=0.4   1249 ms  1.300e-04
  maks_percobaan=1            berhasil 2 | gagal 1 (langkah balasan: 'semua_gagal')
```

## Apa yang Kamu Bangun

1. **Cost & anggaran** — `token_estimasi`, `hitung_biaya`, `biaya_hasil`,
   `jalankan_anggaran` (berhenti **memanggil**, sisanya dilewati — bukan dipaksa).
2. **Retry berdisiplin** — `backoff_ms` + `panggil_dengan_retry` dengan tiga
   keputusan eksplisit: transien vs fatal, jadwal tunggu, dan **jangan menunggu
   setelah percobaan terakhir**.
3. **Fallback chain** — melewati tautan berurutan, melaporkan `dicoba` + `lompatan`,
   dan mengangkat `GalatSemuaPercobaan` kalau seluruh rantai habis.
4. **Cache konten-address** — kunci SHA-256 dari (model, messages, params),
   TTL **tick logis**, dan hit/miss/expired dihitung terpisah.
5. **Streaming** — `konsumsi_stream` merakit chunk, mengukur TTFT, dan jumlah
   chunk hanya dari delta non-kosong.
6. **Function calling** — `skema_parameter`/`skema_tools` dari contoh argumen,
   `validasi_argumen` dengan pesan error stabil, `jalankan_tool` (validasi →
   eksekusi), dan `loop_tool` dengan `maks_iterasi` wajib.
7. **Orkestrasi** — `jalankan_pipeline` menjalankan langkah berurutan dengan
   model + suhu sendiri, melaporkan biaya/latensi/attempts **per langkah**.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_cost.py` (paling mudah).
2. Isi satu modul, jalankan test:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan yang disarankan: `cost.py` → `retry.py` → `fallback.py` → `cache.py` →
   `stream.py` → `tools.py` → `orchestrate.py` sampai **semua 107 test hijau**.
4. Buka `starter.ipynb` — jalankan tujuh blok eksperimen (angka terkunci), lalu
   jawab **6 Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/llmkit_ref.py` / `solusi.ipynb`.

Ingin melihat angka terkunci tanpa menjalankan notebook?

```bash
python _calib.py
```

> 💡 Kalau test macet: cek (1) **status dihitung SEBELUM** menaikkan
> `jumlah_panggilan` di mock, (2) jangan menunggu backoff setelah percobaan
> terakhir, (3) `expired` **bukan** `miss` di cache, (4) de-dup tool lewat pesan
> assistant `tool_calls` — **bukan** substring prompt (prompt sendiri memuat kata
> kunci tool), (5) `total_ms` = `latency_ms` + total jeda. Lima itu penyebab 90%
> bug di project ini.

## Pertanyaan Analisis (bagian penilaian!)

1. `bad_request` tidak diretry, `rate_limit` diretry. Apa yang rusak kalau
   kedua-duanya diretry dengan backoff yang sama? Kaitkan dengan biaya dan
   dengan pengalaman pengguna.
2. Backoff naik 1000 → 2000 → 4000 ms. Kenapa tidak langsung 8000 ms sejak
   percobaan pertama, dan kenapa ada batas atas (`maks_ms`)?
3. Fallback ke `murah`/`mini` menyelamatkan permintaan, tapi apa harganya?
   Kapan fallback ke model kecil justru lebih buruk daripada mengembalikan error?
4. Cache di sini di-key oleh model + pesan + parameter. Sebutkan dua hal lagi
   yang WAJIB masuk kunci cache di produksi (petunjuk: versi prompt dan Bab 12).
5. Streaming menurunkan TTFT 120 ms vs total 340 ms, tapi total waktunya sama.
   Ukuran apa yang sebenarnya diperbaiki, dan kapan streaming justru menambah
   kompleksitas tanpa manfaat?
6. Argumen tool divalidasi di kode kita sebelum handler dipanggil. Apa risiko
   kalau kita langsung mengeksekusi `eval(argumen)` seperti contoh di README
   Bab 11, dan bagaimana prinsip least-privilege (Bab 14) membatasinya?

## Prasyarat

```bash
# tidak ada!
```

> Stdlib murni. Tidak ada `openai`, tidak ada `tenacity`, tidak ada `pydantic` —
> supaya *mekanismenya* terlihat, bukan tersembunyi di balik library. Setelah kamu
> paham, memakai library produksi akan terasa seperti memakai kembali kode yang
> kamu tulis sendiri (dan kamu tahu di mana ia akan menipumu).

> Bab ini memakai kembali pola dari Bab 10 (output terstruktur, guard, evaluasi)
> dan Bab 9 (reproducibility & TDD). Yang diuji bukan kemampuan memanggil API,
> tapi **kemampuan membuat panggilan API bertahan saat gagal** — keterampilan
> yang membedakan prototipe dari sistem.
