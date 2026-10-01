# 🧾 Cheatsheet — Bab 14: Agentic AI

Kompilasi formula, kontrak, & jebakan. Angka terkunci SEED 14 (project `agentkit`).

## 1. Anatomi Agent
```
Agent = model (otak) + tools (tangan) + memory (konteks) + policy (rencana) + termination (rem)
Bukan agent   : satu panggilan LLM (chat, ringkas) — tidak ada loop & keputusan tool
Agent         : loop reason → act (tool) → observe → ... → final
Mesin vs otak : loop/guardrail/memory/eval = mesin (kontrak tetap);
                intent & pemilihan tool = otak (aturan di project, LLM di produksi)
```

## 2. Format ReAct (terkunci di `loop_react.format_langkah`)
```
Thought: Saya perlu cek status pesanan INV-001 di sistem.
Action: cek_pesanan({"invoice_id": "INV-001"})
Observation: {'id': 'INV-001', 'status': 'dikirim', ...}

- Action hanya bila ada tool; Observation hanya bila ada hasil
- Langkah final: tool=None, args={'aksi_final': 'jawab' | 'tanya_balik' | 'minta_konfirmasi'}
- Trace penuh = blok-blok dipisah baris kosong (format_trace)
```

## 3. Validasi Tool (SEBELUM eksekusi — jangan percaya argumen model)
```
Urutan cek : tool dikenal → args dict → required ada → type → min_panjang → pola (re.search)
Return     : (True, '') atau (False, pesan)
Dispatch   : error TIDAK raise → {'error': pesan} = observasi bagi agent
Contoh     : validasi_skema('cek_pesanan', {'invoice_id': 'ABC'}) → (False, '... pola INV-\d{3}')
             eksekusi_tool('cek_pesanan', {'invoice_id': 'INV-999'}) → {'error': 'pesanan tidak ditemukan'}
```

## 4. Guardrail — Aksi Berisiko & Iterasi
```
butuh_konfirmasi('refund') → True   (perbandingan PERSIS, bukan substring)
tool_refund(...) → {'butuh_konfirmasi': True, ...}  — TIDAK PERNAH mengeksekusi
Guardrail iterasi:
  rencana tanpa final / LLM loop → dipotong di maks_iter (default 6)
  → final escallate 'hubungi agen manusia' + berhenti_dini=True
```

## 5. Sanitasi Prompt Injection
```
POLA_INJECTION = ('ignore previous', 'abaikan instruksi', 'system prompt',
                  'lupakan aturan', 'reveal', 'kirim kode', 'api key')
sanitasi(teks)      : frasa → '[dihapus]' (case-insensitive, per kemunculan);
                      '\n' → spasi; strip(); SALINAN (input utuh)
amankan_observasi   : string → sanitasi | dict/list → salinan rekursif |
                      tipe lain → apa adanya. Fungsi murni.
Sumber injection    : tool output (halaman web, DB, email) — bukan hanya input user
```

## 6. Memory Sesi
```
tambah(peran, teks) → {'peran', 'teks', 'token'}   token = ceil(len/4)
konteks()           : maks_giliran TERAKHIR, URUTAN ASLI (bukan dibalik!)
                      4 giliran, maks 3 → ['dua','tiga','empat']
ringkas()           : 'peran: teks' per baris (semua riwayat)
```

## 7. Cache Retrieval (sambungan Bab 12)
```
Kunci   : kueri.strip().lower()   — '  Status INV-001 ' ≡ 'status inv-001'
hit/miss: tercatat di counter — pola miss→hit = 1 pencarian untuk 2 permintaan
Yang di-cache: hasil PENCARIAN (konteks), bukan jawaban LLM
Kunci produksi harus mencakup: kueri + versi index/korpus + top_k + filter
```

## 8. Planner: Statis vs Bertahap
```
Statis   : tabel intent → rencana penuh (deterministik, murah, kaku)
Bertahap : kondisi runtime menentukan langkah tambah (fleksibel, mahal)
Kontrak sama: list {'thought', 'tool'|None, 'args'}
6 kasus terkunci: status tanpa ID=1 | status+ID=2 | refund+ID=2 | refund tanpa ID=1 |
                  kebijakan=2 | lainnya=1
Multi-agent? Mulai SATU agent + tools bagus. Multi-agent hanya bila perlu —
biaya & debug melipatgandakan.
```

## 9. Evaluasi Agent dari Trace (mini Bab 15)
```
skor = 1.0  tool jalan tanpa error, tanpa tanya_balik, tanpa berhenti dini
     = 0.5  ada tanya_balik (jujur bertanya — lebih baik daripada mengarang)
     = 0.0  error tool ATAU berhenti dini (guardrail iterasi)
temuan: ['bersih'] atau daftar masalah berurutan
Input evaluasi = HASIL (dict dengan 'berhenti_dini'), bukan list langkah
```

## 10. Kapan (Tidak) Pakai Agent + Framework
```
Satu panggilan LLM  : tugas satu-langkah (ringkas, klasifikasi, ekstrak)
Agent               : tujuan multi-langkah yang butuh info dari tools
                      (cek status → refund perlu keputusan bertahap)
Framework produksi  : LangGraph (state eksplisit) | OpenAI Agents SDK | CrewAI
Belajar             : loop sendiri (agentkit) — mesinnya terlihat
Least privilege     : API key per tool, terbatas | log semua aksi | idempotency saat retry
```
