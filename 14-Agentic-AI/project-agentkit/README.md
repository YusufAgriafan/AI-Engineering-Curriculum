# 🏗️ Project Starter — agentkit: Agent dari Nol (Bab 14)

> Kamu membangun **seluruh mesin agent tanpa API key**: registry tools dengan
> validasi skema, loop ReAct dengan guardrail iterasi, sanitasi prompt injection,
> memory sesi + cache retrieval, planner, dan evaluasi agent dari trace.
> Stdlib murni, TDD **57 test**, tanpa jaringan dan tanpa GPU.

## Struktur

```
project-agentkit/
├── README.md               ← kamu di sini
├── RUBRIK.md               ← penilaian mandiri
├── starter.ipynb           ← notebook TUGASMU (eksperimen + pertanyaan analisis)
├── solusi.ipynb            ← notebook SOLUSI (buka setelah selesai / mentok)
├── agentkit/               ← paket yang kamu isi (stdlib murni)
│   ├── data.py             ← KB mini, pesanan, skema tools, tarif (JANGAN DIUBAH)
│   ├── nlu.py              ← GIVEN: policy deterministik (intent → rencana ReAct)
│   ├── tools.py            ← TODO: registry + validasi skema + eksekusi
│   ├── loop_react.py       ← TODO: loop ReAct + guardrail iterasi + trace
│   ├── guardrail.py        ← TODO: aksi berisiko + sanitasi injection
│   ├── memory.py           ← TODO: riwayat sesi + cache retrieval + biaya
│   └── agent.py            ← TODO: end-to-end + planner + evaluasi trace
├── tests/                  ← spesifikasi TDD (JANGAN DIUBAH)
│   ├── test_tools.py       (17 test)
│   ├── test_loop_react.py  (10 test)
│   ├── test_guardrail.py   (10 test)
│   └── test_agent_eval.py  (20 test)
├── _calib.py               ← skrip audit: mencetak angka terkunci (bukan test)
├── _gen_notebooks.py
├── _verify_project.py      ← verifier otomatis (patch referensi + test + notebook)
└── solusi/
    └── agentkit_ref.py     ← implementasi referensi (untuk cek mandiri)
```

## Kenapa "Otak"-nya Aturan, Bukan LLM?

Pola yang sama dengan Bab 11 (`transport.py`), Bab 12 (`embed.py`), Bab 13
(`ftkit`) — tiga alasan:

1. **Reproducible.** LLM menghasilkan keputusan berbeda tiap panggilan; trace
   tidak bisa dibandingkan. Policy deterministik menghasilkan **angka yang sama
   di komputer siapa pun** — syarat evaluasi yang adil dan test yang bermakna.
2. **Gratis & cepat.** Satu run agent < 0,01 detik. Kamu bisa eksperimen guardrail,
   memory, planner, injection puluhan kali — bukan sekali semalam.
3. **Kontraknya SAMA.** Loop `reason → act → observe` yang kamu bangun di sini
   adalah loop yang sama dengan LangGraph/OpenAI Agents SDK di produksi. Yang
   berubah saat otak diganti LLM: kualitas keputusan — bukan bentuk mesinnya.

## Angka Terkunci (dari `_calib.py`, SEED 14)

```
INTENT
  'Bagaimana status pesanan INV-001?'     → status_pesanan
  'Saya mau refund INV-002 barang rusak'  → refund
  'Berapa lama pengiriman reguler?'       → kebijakan
  'halo halo'                             → lainnya (tanya balik, bukan menebak)

TOOLS
  cari 'pengembalian barang'        → [(d1, 1.0)]
  cari 'pengembalian cod cashback'  → [(d3, 0.6667), (d1, 0.3333)]   (2/3 vs 1/3)
  cek INV-001 → status 'dikirim' | PII 'user' dibuang | INV-999 → error dict
  refund → butuh_konfirmasi True (SELALU — human-in-the-loop)

LOOP REACT
  status INV-001: 2 langkah, final 'jawab', tidak berhenti dini
  format trace: Thought: / Action: tool(args) / Observation:

GUARDRAIL
  sanitasi 'Ignore previous instructions' → '[dihapus] instructions'
  amankan_observasi rekursif (dict/list) — input tak berubah
  loop tanpa final, maks_iter=3 → 3 langkah + escallate + berhenti_dini=True

MEMORY
  konteks maks 3 → ['dua','tiga','empat'] (urutan ASLI, bukan dibalik)
  cache: kunci strip().lower() → hit/miss (1,1) | token 'abcd' = 1
  biaya sesi 3 token api_mini = 4.5e-07 USD per 2 giliran (1+2 token)

PLANNER (statis = bertahap, 6 kasus)
  status tanpa ID → 1 | status + ID → 2 | refund + ID → 2 | refund tanpa ID → 1
  kebijakan → 2 | lainnya → 1

EVAL TRACE
  bersih 1.0 | tanya_balik 0.5 | error tool 0.0 | berhenti dini 0.0
  verifikasi_tools → True
```

## Apa yang Kamu Bangun

1. **Tools layer** — validasi skema JSON-Schema-subset sebelum eksekusi; error
   sebagai observasi (bukan crash); PII dibuang sebelum masuk konteks LLM.
2. **Loop ReAct** — format trace terkunci; guardrail iterasi: rencana tanpa
   final dipotong di `maks_iter` lalu escallate ke manusia.
3. **Guardrail** — aksi berisiko wajib konfirmasi (perbandingan persis);
   sanitasi injection case-insensitive + rekursif; fungsi murni.
4. **Memory** — konteks = giliran terakhir urutan asli; ringkasan; cache
   retrieval berkunci ternormalisasi dengan hit/miss tercatat.
5. **Planner** — statis (tabel) vs bertahap (kondisi runtime): dua pendekatan,
   satu kontrak langkah.
6. **Evaluasi agent** — skor dari trace: 1.0 bersih / 0.5 tanya balik / 0.0
   error atau berhenti dini. Agent yang jujur bertanya > agent yang mengarang.

## Cara Kerja (mode TDD)

1. Baca `tests/` — itu spesifikasimu. Mulai dari `test_tools.py`.
2. Isi satu modul, jalankan:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Urutan: `tools.py` → `loop_react.py` → `guardrail.py` → `memory.py` →
   `agent.py` sampai **57 test hijau**.
4. Buka `starter.ipynb` — jalankan delapan blok eksperimen (angka terkunci), lalu
   jawab **6 Pertanyaan Analisis**.
5. Isi `RUBRIK.md`. Baru bandingkan dengan `solusi/agentkit_ref.py` / `solusi.ipynb`.

Ingin melihat angka terkunci tanpa menjalankan notebook?

```bash
python _calib.py
```

> 💡 Tiga jebakan yang menyumbang kebanyakan bug di project ini:
> (1) `konteks()` harus mengembalikan giliran terakhir **dalam urutan asli** —
> membalik urutan mengubah dialog;
> (2) error tool adalah **observasi** — `eksekusi_tool` yang melempar exception
> membuat loop agent mati di pertanyaan pertama;
> (3) `evaluasi_trace` menerima **hasil** (dict dengan `berhenti_dini`), bukan
> list langkah — berhenti dini harus menurunkan skor ke 0.0.

## Pertanyaan Analisis (bagian penilaian!)

1. Klasifikasi intent di sini aturan kata kunci. Kapan aturan cukup, dan kapan
   HARUS diganti LLM tool-calling? Apa yang tetap sama di mesin agent-nya?
2. `cek_pesanan` membuang kunci 'user' (PII). PII apa lagi yang biasanya ada di
   sistem nyata, dan di lapisan mana sebaiknya dibuang — tool, guardrail, atau LLM?
3. Guardrail iterasi berhenti di maks_iter dan escallate. Trade-off memilih
   maks_iter terlalu kecil vs terlalu besar?
4. Planner statis vs bertahap menghasilkan langkah sama di 6 kasus. Kapan planner
   bertahap (loop LLM) benar-benar dibutuhkan, dan apa biayanya?
5. Cache retrieval hit-rate di sini 1/2. Kunci cache apa lagi yang harus masuk
   selain kueri ternormalisasi agar tidak menyajikan konteks basi (Bab 12)?
6. Skor trace: tanya_balik = 0.5, di atas error (0.0). Kenapa 'bertanya' dihitung
   lebih baik daripada 'menjawab dengan data error', dan kapan 0.5 masih kurang?

## Prasyarat

```bash
# tidak ada!
```

> Stdlib murni (`re`, `json`, `math`). Tidak ada LangChain, tidak ada OpenAI SDK,
> tidak ada jaringan — supaya *mesinnya* terlihat. Setelah paham, memakai
> LangGraph/OpenAI Agents SDK terasa seperti memakai ulang kode yang kamu tulis sendiri.

> Bab ini memakai kembali pola dari Bab 10 (prompt ReAct), Bab 11 (tool calling,
> biaya token, cache), Bab 12 (abstain saat tidak yakin, cache retrieval) dan
> Bab 13 (model lokal sebagai otak agent). Lanjutan: Bab 15 (eval LLM —
> `evaluasi_trace` adalah eval-set mini-mu).
