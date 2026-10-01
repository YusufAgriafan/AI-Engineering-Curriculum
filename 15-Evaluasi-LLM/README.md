# Bab 15 — Evaluasi LLM

> "Rasanya lebih bagus" bukan metrik. Bab ini membuat Anda bisa membuktikan bahwa sistem AI Anda baik — dan mendeteksi regresi sebelum user menyadarinya. Ini kompetensi paling membedakan AI engineer profesional. Di project `evalkit` Anda membangun **seluruh mesin evaluasinya dari nol** (96 test, stdlib murni).

## 🎯 Tujuan Belajar
- Menyusun eval set (golden dataset) & metrik: exact, heuristik, model-graded, human.
- Membangun eval pipeline otomatis yang jalan setiap kali prompt/model berubah.
- Observability: tracing, logging, monitoring produksi.

## 📦 Materi Pendukung

| Artefak | Isi |
|---|---|
| 🔬 Lab | [`01_lab_eval.ipynb`](01_lab_eval.ipynb) — 8 bagian + tantangan v4, angka terkunci SEED 15, cek otomatis per latihan |
| 📝 Kuis | [`02_kuis_eval.ipynb`](02_kuis_eval.ipynb) — 10 soal (6 PG + 4 coding), 22 poin, dinilai otomatis, **mandiri** |
| 🔑 Kunci | [`03_kunci_jawaban_eval.ipynb`](03_kunci_jawaban_eval.ipynb) — jawaban + **kenapa**, kode referensi tiap soal coding |
| 🧾 Cheatsheet | [`cheatsheet-eval.md`](cheatsheet-eval.md) — semua kontrak, formula, & jebakan |
| 🏗️ Project | [`project-evalkit/`](project-evalkit/README.md) — TDD 96 test, tanpa API/GPU |

Verifier otomatis: `python _verify_lab.py`, `python _verify_kuis.py`,
`project-evalkit/_verify_project.py`.

## 1. Materi Inti

### 1.1 Kumpulkan Contoh & Golden Dataset
- 30–100 kasus nyata (dari log, user, tim), termasuk **edge case**: input kosong, bahasa campur, prompt injection, pertanyaan di luar cakupan.
- Struktur kasus: `{'id', 'kategori', 'tanya', 'kunci', 'harus_tahu'}` — kategori terkunci di project: kebijakan, status, abstain, injection, hallucination, edge.
- **Aturan emas:** kasus yang jawaban benarnya "menolak" (abstain/injection/hallucination/edge) WAJIB `kunci = None` — mengisi kunci di sana berarti evalset mengajari sistem menjawab hal yang seharusnya tidak dijawab.
- Validasi evalset SEBELUM dipakai mengukur: id unik, kategori dikenal, `tanya` non-kosong (kecuali edge).
- Split seperti Bab 3 — tapi **deterministik tanpa random**: index masuk test bila `(i*7 + SEED) % 100 < rasio*100`. Split yang identik antar hari = skor antar run bisa dibandingkan. Dev (untuk iterasi) vs test (disentuh hanya saat rilis besar).

### 1.2 Jenis Metrik
| Jenis | Contoh | Kapan |
|---|---|---|
| Deterministik | exact match, JSON valid, sitasi ada, latency | Format, struktur |
| Heuristik | panjang, kata terlarang, korelasi lexicon | Cepat & murah |
| **Model-graded (LLM-as-judge)** | rubrik 1–5: akurasi, relevansi, faithfulness | Kualitas linguistik |
| Human | rating ahli, A/B, thumbs up/down | Ground truth & kalibrasi judge |

- Urutan memilih scorer: **termurah dulu**. Tiga scorer deterministik menutup sebagian besar kasus: `skor_exact` (contains setelah normalisasi), `skor_abstain` (**equality**, bukan contains — frasa abstain di tengah jawaban panjang bukan abstain), `skor_format` (non-kosong, tanpa sisa injection, tanpa jejak internal `[dihapus]`).
- Kasus jebakan dinilai dari kemampuan **MENOLAK**: injection yang dilayani jawaban benar tetap 0.0; abstain/hallucination yang "berani menjawab" tetap 0.0.
- **LLM-as-judge**: beri judge rubrik eksplisit + contoh skala; waspadai bias (panjang, posisi, self-preference). Kalibrasi dengan sampling human review — tanpa kalibrasi, judge = selera model, bukan metrik. Rubrik terkunci di project: 5 kutipan konteks → 1.0 · 4 menjawab → 0.75 · 2 abstain → 0.25 · 1 mengarang → 0.0.
- Untuk RAG: **faithfulness** (jawab sesuai konteks?) & **answer relevancy** → pakai `ragas` (Bab 12).

### 1.3 Eval Pipeline: laporan per kategori, bukan satu angka
```
evalset.yaml  →  runner (panggil sistem)  →  scorer  →  laporan (tabel + diff vs versi lalu)
```
- Skor rata-rata **menyembunyikan trade-off**. Angka terkunci project (12 kasus): v1 = 0.3333 · v2 = 0.6667 (jujur TAPI injection 0.0!) · v3 = 0.8333. Hanya laporan per kategori yang menunjukkan 'jujur naik, injection masih nol'.
- Sinyal 3-dimensi: **cakupan** (bobot 0.4) + **jujur** (0.3) + **aman** (0.3) → v1 0.2667 · v2 0.5667 · v3 0.8667. 'Jujur' dan 'aman' harus sinyal terpisah — menggabungkannya menyembunyikan sistem yang jujur namun melayani injection.
- Jalankan di CI: setiap PR yang menyentuh prompt/model/konfigurasi → eval wajib hijau. **Gate rilis**: gagal bila ada kategori turun melewati ambang (0.05) ATAU skor baru < skor_min (0.75) — cek regresi DULU, baru skor absolut.
- Selalu bandingkan **versi vs versi** (regression testing), bukan angka absolut.
- Error analysis: daftar kasus gagal urut terburuk = peta kerja. Klasteri (tema? panjang input? bahasa?) → perbaiki yang paling sering. Bedakan perbaikan **struktural** (gap tool 0.0 permanent → tambah tool, seperti v4) dari tuning prompt.

### 1.4 Observability Produksi
- **Trace** setiap permintaan: prompt, konteks RAG, tool calls, token, biaya, latensi (Langfuse / LangSmith / OpenTelemetry).
- Biaya = Σ (token_in×harga_in + token_out×harga_out), token = ceil(len/4). Model tak dikenal → KeyError (gagal nyaring, bukan diam-diam 0). Eval 12 kasus api_mini = **$0.00013305** — eval itu murah; tidak ada alasan tidak jalan.
- Latensi dilaporkan **nearest-rank**: urut naik, rank = ceil(p/100 × n). p50 = pengalaman tipikal; p99 = pengalaman user paling sabar sekalipun. Contoh api_mini: p50 480 ms · p90/p99 900 ms.
- Dashboard: error rate, halusinasi terdeteksi, biaya/hari, p95 latency.
- Feedback loop: thumbs down user → antrean review → error analysis → kasus eval baru (kategori + kunci; "harus menolak" → kunci None). Produksi menemukan kegagalan, eval menguncinya. Lingkaran selesai.

## 2. Latihan
1. Buat `evalset.yaml` 30 kasus untuk sistem RAG Bab 12 (termasuk 5 kasus "harus menjawab tidak tahu" — kunci = None).
2. Runner + scorer: exact match untuk "tidak tahu", model-graded rubrik untuk sisanya; laporkan skor per kategori.
3. Integrasikan ke GitHub Actions: gagalkan PR jika skor turun > 2 poin.
4. Tambahkan Langfuse: trace 10 permintaan, temukan 1 bottleneck latensi.
5. Selesaikan **lab** (8 bagian + tantangan v4), **kuis** (22 poin), dan **project `evalkit`** (96 test + 6 pertanyaan analisis).

## 📚 Referensi Online
- [OpenAI Cookbook — evaluation](https://cookbook.openai.com/examples/evaluation)
- [RAGAS docs](https://docs.ragas.io/)
- [Langfuse docs (open-source LLM observability)](https://langfuse.com/docs)
- [Hamel Husain — Your AI product needs evals](https://hamel.dev/blog/posts/evals/)
- [Anthropic — evals guide](https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests)

## ✅ Checklist Kompetensi
- [ ] Punya golden dataset + runner + scorer yang bisa dijalankan satu perintah
- [ ] Evalset tervalidasi: kasus "harus menolak" kunci = None; split deterministik tanpa random
- [ ] Laporan per kategori + sinyal cakupan/jujur/aman terpisah (bukan satu rata-rata)
- [ ] Gate rilis dua syarat (regresi dulu, skor_min belakangan) menjalankan di CI
- [ ] Mengkalibrasi LLM-judge terhadap penilaian manusia (rubrik + bias panjang terukur)
- [ ] Sistem punya tracing produksi dengan biaya & latensi per request
- [ ] Kuis bab ini selesai (22 poin) — atau review kunci sampai paham **kenapa**
- [ ] Project `evalkit` 96 test hijau + pertanyaan analisis terjawab
