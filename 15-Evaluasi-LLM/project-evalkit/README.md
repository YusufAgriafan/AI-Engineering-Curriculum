# 🏗️ Project Starter — evalkit: Mesin Evaluasi LLM dari Nol (Bab 15)

> Kamu membangun **seluruh mesin evaluasi tanpa API key**: evalset tervalidasi +
> split deterministik, scorer berlapis (exact/abstain/format), runner per kategori,
> regression + gate rilis untuk CI, monitor biaya & latensi, judge dengan rubrik
> & kalibrasi, dan error analysis + feedback loop. Stdlib murni, TDD **96 test**,
> tanpa jaringan dan tanpa GPU.

## Struktur

```
project-evalkit/
├── README.md               ← kamu di sini
├── RUBRIK.md               ← penilaian mandiri
├── starter.ipynb           ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb            ← notebook SOLUSI (buka setelah selesai / mentok)
├── evalkit/                ← paket yang kamu isi (stdlib murni)
│   ├── data.py             ← KB, EVALSET 12 kasus, tarif, ambang, rubrik (JANGAN DIUBAH)
│   ├── sistem.py           ← GIVEN: tiga versi "produk" v1/v2/v3 yang dievaluasi
│   ├── tokenizer.py        ← GIVEN: token_estimasi = ceil(len/4)
│   ├── evalset.py          ← TODO: muat, validasi, split deterministik
│   ├── scorers.py          ← TODO: exact, abstain, format, skor_kasus
│   ├── runner.py           ← TODO: jalankan evalset → hasil run + agregat per kategori
│   ├── regression.py       ← TODO: bandingkan dua run + gate rilis
│   ├── monitor.py          ← TODO: biaya run & latensi (nearest-rank)
│   ├── judge.py            ← TODO: judge rubrik 1-5 + bias panjang + kalibrasi + sinyal
│   └── evaluasi.py         ← TODO: evaluasi end-to-end + error analysis + feedback loop
├── tests/                  ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_evalset.py     (16 test)
│   ├── test_scorers.py     (20 test)
│   ├── test_runner.py      (11 test)
│   ├── test_regression.py  (8 test)
│   ├── test_monitor.py     (13 test)
│   ├── test_judge.py       (17 test)
│   └── test_evaluasi.py    (11 test)
├── _calib.py               ← skrip audit: mencetak angka terkunci (bukan test)
├── _gen_notebooks.py
├── _verify_project.py      ← verifier otomatis (patch referensi + test + notebook)
└── solusi/
    └── evalkit_ref.py      ← implementasi referensi (untuk cek mandiri)
```

## Kenapa Sistem yang Dievaluasi Deterministik?

Pola yang sama dengan Bab 11 (`transport.py`), Bab 12 (`embed.py`), Bab 13
(`ftkit`), Bab 14 (`agentkit`) — tiga alasan:

1. **Reproducible.** LLM menghasilkan jawaban berbeda tiap panggilan; skor tidak
   bisa dibandingkan antar run. Tiga versi "produk" v1/v2/v3 (retrieval kata
   kunci + generator ekstraktif) menghasilkan **angka yang sama di komputer siapa
   pun** — syarat regression testing yang bermakna.
2. **Gratis & cepat.** Satu run 12 kasus < 0,01 detik dan $0.00013305 bila
   di-token-kan ke api_mini. Kamu bisa eksperimen gate, ambang, dan rubrik
   puluhan kali — bukan sekali semalam.
3. **Kontraknya SAMA.** `evalset → runner → scorer → laporan → gate` adalah
   bentuk yang sama dengan promptfoo, RAGAS, Langfuse datasets di produksi.
   Yang berubah saat sistem nyata dipasang: sumber jawaban (panggilan LLM) —
   bukan bentuk mesin evalnya.

## Angka Terkunci (dari `_calib.py`, SEED 15)

```
EVALSET
  12 kasus | 6 kategori: kebijakan(4) status(2) abstain(2) injection(2)
  hallucination(1) edge(1) | ambang abstain 0.20
  split (i*7+15)%100 < 25 → test = ['e01','e02'] | dev = 10 kasus

SCORERS
  skor_exact('...14 hari sejak diterima.', '14 hari') → 1.0 | kunci None → 0.0
  skor_abstain(FRASA_TIDAK_TAHU) → 1.0 | 'Maaf, saya tidak tahu.' → 0.0
  skor_format('api key: sk-123') → 0.0 | '[dihapus] ...' → 0.0
  injection: menolak → 1.0 | melayani jawaban benar → 0.0 (dilayani = hukuman)

RUN (rata + per kategori)
  v1: 0.3333 | {'kebijakan': 1.0, sisanya 0.0}
  v2: 0.6667 | + abstain/hallucination/edge 1.0 — injection MASIH 0.0
  v3: 0.8333 | + injection 1.0 — status tetap 0.0 (gap tool!)
  sinyal cakupan 0.4 + jujur 0.3 + aman 0.3 → v1 .2667 | v2 .5667 | v3 .8667

REGRESSION & GATE (v1 → v3)
  perbaikan: abstain, injection, hallucination, edge | regresi: []
  gate: LOLOS 'skor rata 0.3333 → 0.8333; tanpa regresi'
  gate (v2 → v1): GAGAL 'regresi di kategori: abstain, hallucination, edge'

MONITOR
  biaya run v3 api_mini  : 0.00013305 USD | api_besar: 0.0022175 USD
  persentil api_mini     : {'p50': 480.0, 'p90': 900.0, 'p99': 900.0}  (nearest-rank)
  persentil lokal        : {'p50': 210.0, 'p90': 400.0, 'p99': 400.0}

JUDGE (rubrik 5/4/2/1 → skor 1.0/0.75/0.25/0.0)
  kutipan konteks → rubrik 5 | menjawab tanpa kutipan → rubrik 4
  abstain → rubrik 2 | mengarang → rubrik 1
  bias panjang: pendek == panjang == 1.0 (nol BY DESIGN)
  kalibrasi([1.0,0.25,0.0],[1.0,0.0,0.0]) → selisih_rata 0.0833, setuju 1.0, layak True

ERROR ANALYSIS & FEEDBACK
  v1 gagal 8 kasus | v2 gagal 4 | v3 gagal ['e05','e06'] → antrean review
  laporan: terbaik v3 | delta v3-v1 = 0.6
```

## Apa yang Kamu Bangun

1. **Evalset layer** — validasi sebelum dipakai mengukur: id unik, kategori
   dikenal, kasus "harus menolak" wajib `kunci = None`; split interleaved
   deterministik (tanpa `random`) agar skor antar hari bisa dibandingkan.
2. **Scorer berlapis** — termurah dulu: exact (contains ternormalisasi),
   abstain (**equality**, bukan contains), format (tanpa sisa injection /
   jejak internal). Kasus jebakan dinilai dari kemampuan MENOLAK.
3. **Runner** — agregat per kategori, bukan satu angka: rata-rata menyembunyikan
   trade-off ('jujur naik, injection tetap 0').
4. **Regression & gate** — bandingkan versi vs versi; gate gagal bila ada
   kategori turun melewati ambang ATAU skor baru di bawah skor_min. Ini yang
   memblokir PR buruk di CI sebelum user melihatnya.
5. **Monitor** — biaya per run dari tarif per 1 jt token (model tak dikenal →
   KeyError: gagal nyaring); latensi nearest-rank p50/p90/p99.
6. **Judge + kalibrasi** — rubrik eksplisit 1–5, deteksi bias panjang, dan
   kalibrasi terhadap penilaian manusia (tanpa kalibrasi, judge = selera).
   Ditambah **sinyal 3-dimensi**: cakupan / jujur / aman dengan bobot sendiri.
7. **Error analysis & feedback loop** — kasus gagal urut terburuk → antrean
   review → kasus eval baru. Produksi menemukan kegagalan, eval menguncinya.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_evalset.py`.
2. Isi satu modul, jalankan:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan: `evalset.py` → `scorers.py` → `runner.py` → `regression.py` →
   `monitor.py` → `judge.py` → `evaluasi.py` sampai **96 test hijau**.
4. Buka `starter.ipynb` — jalankan delapan blok eksperimen (angka terkunci), lalu
   jawab **6 Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/evalkit_ref.py` / `solusi.ipynb`.

Ingin melihat angka terkunci tanpa menjalankan notebook?

```bash
python _calib.py
```

> 💡 Tiga jebakan yang menyumbang kebanyakan bug di project ini:
> (1) `skor_abstain` harus **equality** setelah normalisasi — contains membuat
> frasa abstain di tengah jawaban panjang ikut dinilai abstain;
> (2) `latensi_persentil` memakai **nearest-rank** (`rank = ceil(p/100 × n)`) —
> interpolasi linear menghasilkan angka berbeda yang gagal di test;
> (3) `gate_rilis` mengecek **regresi dulu, skor_min belakangan** — urutan
> terbalik membuat alasan gagal menyesatkan di CI.

## Pertanyaan Analisis (bagian penilaian!)

1. Kategori 'status' = 0.0 di SEMUA versi. Kapan eval mengarahkan ke perbaikan
   struktural (tambah tool) vs tuning prompt? Bagaimana membedakannya dari
   error analysis?
2. v2 naik signifikan dari v1 hanya dengan abstain gate, TAPI injection-nya 0.
   Mengapa 'jujur' dan 'aman' harus sinyal terpisah, bukan satu skor rata?
3. Gate rilis menolak v2 → v1 karena regresi 3 kategori. Ambang apa yang tepat:
   0.05, 0.02? Apa risiko ambang terlalu ketat vs terlalu longgar?
4. Judge mini tidak bias panjang BY DESIGN. LLM judge nyata punya bias panjang,
   posisi, self-preference. Bagaimana kalibrasi menyedot bias itu (rubrik,
   contoh skala, sampling human review)?
5. Biaya eval 12 kasus api_mini = $0.00013305. Kapan eval MALAH jadi mahal
   (judge LLM per kasus, evalset 10rb kasus), dan bagaimana memangkasnya?
6. Antrean feedback v3 masih berisi e05/e06 (butuh tool). Kasus apa yang HARUS
   ditambahkan ke evalset setelah rilis v3, dan kapan evalset di-refresh tanpa
   merusak perbandingan antar versi?

## Prasyarat

```bash
# tidak ada!
```

> Stdlib murni (`re`, `json`, `math`). Tidak ada OpenAI SDK, tidak ada LangChain,
> tidak ada jaringan — supaya *mesin evalnya* terlihat. Setelah paham, memakai
> promptfoo/RAGAS/Langfuse terasa seperti memakai ulang kode yang kamu tulis sendiri.

> Bab ini memakai kembali pola dari Bab 3 (split dev/test), Bab 8 (evaluasi
> produksi klasik), Bab 12 (abstain gate), Bab 13 (eval sebelum/sesudah), Bab 14
> (evaluasi agent dari trace). Lanjutan: Bab 16 (deployment — eval gate jadi
> bagian dari CI/CD sebelum rilis).
