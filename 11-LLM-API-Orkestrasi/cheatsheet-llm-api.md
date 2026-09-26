# 📄 Cheatsheet — LLM API & Orkestrasi (Bab 11)

> Review cepat. Uji diri: tutup file ini, tulis ulang alur `panggil_dengan_retry` +
> daftar isi kunci cache dari ingatan, baru cek.

---

## 1. Satu Panggilan API = Kontrak, Bukan Percakapan

```python
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": ...}, {"role": "user", "content": ...}],
    temperature=0.0,            # ekstraksi/klasifikasi: 0–0.2
    max_tokens=500,             # batasi output = batasi biaya
    response_format={"type": "json_object"},
    timeout=30,
    stream=False,
)
```

| Peran | Isi | Aturan |
|---|---|---|
| `system` | aturan, persona, format, larangan | **jangan pernah** menyambung data pengguna ke sini |
| `user` | permintaan + data tak tepercaya | dibungkus delimiter (Bab 10) |
| `assistant` | jawaban sebelumnya / `tool_calls` | bukti tool sudah dipanggil |
| `tool` | hasil eksekusi tool | `tool_call_id` wajib cocok |

Semua provider (OpenAI/Anthropic/Gemini/DeepSeek) memakai pola yang sama; LiteLLM
menyeragamkannya. API key **selalu** dari environment variable.

---

## 2. Streaming: TTFT vs Total

```
tanpa streaming : [....... tunggu 340 ms .......] lalu semua teks
dengan streaming: [120 ms] kata1                         <- TTFT
                  setiap 20 ms satu kata, selesai di 340 ms
```

| Metrik | Artinya | Dipakai untuk |
|---|---|---|
| **TTFT** | waktu sampai token pertama terlihat | persepsi responsif (UX) |
| **total_ms** | sampai respons selesai | throughput & biaya |

Angka terkunci (lab Bagian 5): TTFT 120 ms, total 340 ms, 11 chunk → pengguna melihat
kata pertama ~65% lebih cepat, **total prosesnya sama**.

```python
potongan, jumlah, ttft, terakhir = [], 0, None, 0
for c in chunks:                    # {'delta': str, 't_ms': int}
    terakhir = c["t_ms"]
    if c["delta"]:                  # chunk kosong = penanda akhir, bukan kata
        jumlah += 1
        ttft = c["t_ms"] if ttft is None else ttft
        potongan.append(c["delta"])
```

❌ Jangan streaming untuk: output JSON pendek, batch/async, atau parsing yang butuh
hasil lengkap.

---

## 3. Retry & Backoff: Dua Keputusan

```
JENIS_TRANSIEN = ("rate_limit", "server_error", "timeout")   -> ulangi
bad_request / model tidak ada                                 -> LEMPAR sekarang
```

```python
def backoff_ms(percobaan, basis_ms=1000, faktor=2, maks_ms=8000):
    return min(basis_ms * faktor ** (percobaan - 1), maks_ms)
# [1000, 2000, 4000, 8000, 8000]
```

Urutan yang benar (mudah salah):

```
1. panggil provider
2. catat (percobaan, jenis) ke riwayat
3. fatal?            -> raise SEKARANG
4. percobaan terakhir? -> raise GalatSemuaPercobaan (JANGAN menunggu dulu)
5. sisanya            -> sleep(backoff(percobaan)) lalu ulangi
```

- `sleep_fn` disuntik supaya test tidak menunggu detik nyata.
- Menunggu **sebelum** cek "percobaan terakhir?" = satu jeda sia-sia.
- Angka terkunci: `sedang` gagal sekali → attempts 2, tunggu `[1000]`, latency 240 ms
  → `total_ms` 1240 ms. `dengan maks_percobaan=1` → tunggu `[]`.
- Produksi: batas atas + **jitter** + **circuit breaker**.

---

## 4. Fallback Chain

```python
RANTAI = [{"provider": "utama",    "model": "besar"},
          {"provider": "utama",    "model": "mini"},
          {"provider": "cadangan", "model": "murah"}]
```

- Dicoba **berurutan**; tiap tautan boleh punya `settings` sendiri.
- Wajib dilaporkan: `lompatan` (tautan yang menang) + `dicoba` (jenis galat tiap tautan).
- Angka terkunci: menang di tautan ke-1 (`mini`), setelah `besar` gagal dua kali
  (`rate_limit` + `server_error` → `jenis='semua_gagal'`).
- Trade-off eksplisit: **ketersediaan ⇄ kualitas ⇄ konsistensi**. Rasio biaya
  `besar`:`mini` ≈ 26x.

Kapan **jangan** fallback ke model kecil: ekstraksi JSON ber-skema ketat, tugas yang
butuh presisi/kode — di situ error 503 + retry klien lebih jujur daripada hasil yang
tampak sukses tapi beda kualitas.

---

## 5. Cache: Jangan Bayar Dua Kali

```
kunci = sha256(model + messages + params)[:16]     # + produksi: versi prompt,
                                                   #   konteks retrieval, tenant id
hit / miss / expired DIHITUNG TERPISAH
```

Angka terkunci (lab Bagian 4): tick 0,1,5,6 dengan `ttl=3` →
pola `['miss','hit','miss','hit']`, 4 permintaan = **2 panggilan provider**,
`hit_rate` 0.5 (hit 2, miss 1, expired 1).

```python
def ambil(self, kunci, sekarang):
    if kunci not in self._isi:
        self.miss += 1
        return None
    hasil, tick = self._isi[kunci]
    if sekarang - tick >= self.ttl:      # kedaluwarsa ≠ miss
        self.expired += 1
        del self._isi[kunci]
        return None
    self.hit += 1
    return hasil
```

TTL pakai **tick logis** supaya bisa diuji deterministik. Kunci salah = menyajikan
jawaban basi dengan percaya diri.

---

## 6. Biaya: Hitung Sebelum Memanggil

```python
def hitung_biaya(tokens_in, tokens_out, model, harga=HARGA):
    t = harga[model]                       # USD per 1 JUTA token
    return tokens_in / 1e6 * t["input"] + tokens_out / 1e6 * t["output"]
```

- `token_estimasi(teks) = ceil(len(teks)/4)` — cukup untuk **anggaran**, bukan tagihan.
- Angka terkunci: 1k in / 300 out → `mini` $0.00033, `besar` $0.0095 (~29x).
- **Penjaga anggaran** berhenti *memanggil* saat anggaran habis; sisanya dilewati:

```
anggaran $0.00003 untuk 4 pekerjaan -> diproses 3, dilewati 1, berhenti_di 3
```

Lima cara menekan biaya: model yang tepat · kurangi token input · kurangi token output ·
cache · routing per tingkat kesulitan tugas.

---

## 7. Function Calling: Model Mengusulkan, Kode Memutuskan

```
model -> tool_calls: [{id, nama, argumen}]    USULAN (data tak tepercaya)
kode  -> validasi_argumen -> jalankan_tool    KEPUTUSAN
kode  -> {role: 'tool', tool_call_id, content} OBSERVASI
```

- Pesan error validasi: `missing:<k>`, `type:<k>`, `tak_dikenal:<k>` (stabil → bahan retry).
- `bool` adalah subclass `int` → kecualikan eksplisit di cek tipe.
- **Jangan pernah** `eval(argumen["expression"])` — itu memberi model kendali atas kode.
- De-dup tool dari pesan assistant `tool_calls`, **bukan** dari substring prompt.
- `maks_iterasi` **wajib**: agent tanpa batas = tagihan tanpa batas.
- Angka terkunci: 2 tool → 3 iterasi (`['cari_catatan','hitung']`), 0 tool → 1 iterasi,
  `maks_iterasi=1` → `berhenti='maks_iterasi'`, `jawaban=None`.

---

## 8. Orkestrasi: Pipeline yang Bisa Diaudit

```
klasifikasi (mini,   T=0.0)   184 ms   $1.53e-05
ekstraksi   (mini,   T=0.0)   187 ms   $1.55e-05
balasan     (sedang, T=0.4)  1249 ms   $1.30e-04   ← satu-satunya butuh kreativitas
--------------------------------------------------
total                         1620 ms   (7 langkah->3 langkah, attempts [1,1,2])
```

- Tiap langkah: model + suhu + biaya + latensi sendiri → **kegagalan terlokalisasi**.
- Langkah gagal dilaporkan (`galat='semua_gagal'`), biaya 0 di mock, dan alert bisa
  menyala di langkah yang benar.
- Biaya & latency naik (lebih banyak panggilan) — trade-off yang **diukur**, bukan dirasa.

---

## 9. Anti-pattern

| Anti-pattern | Gejala | Solusi |
|---|---|---|
| Retry untuk semua galat | `bad_request` diulang 5x | klasifikasikan transien vs fatal |
| Backoff tanpa batas atas | percobaan ke-10 menunggu menit | `maks_ms` + jitter |
| Tunggu setelah percobaan terakhir | jeda sia-sia sebelum error | cek urutan sebelum `sleep` |
| Kunci cache kurang lengkap | jawaban basi / salah model | model+pesan+params+versi prompt+konteks+tenant |
| Fallback disembunyikan | kualitas turun tanpa jejak | log `dicoba` & `lompatan` |
| Loop tool tanpa batas | biaya/latensi tak terbatas | `maks_iterasi` + anggaran + timeout |
| `eval` argumen tool | pintu RCE | validasi skema → whitelist handler |
| API key hardcode | kebocoran kredensial | env var + `.env` di-gitignore |
| Streaming untuk JSON pendek | kompleksitas tanpa manfaat | streaming hanya untuk UX |
| Biaya tidak dihitung | tagihan sebagai kejutan | `biaya_hasil` + penjaga anggaran |

---

## 10. Checklist Review 30 Detik

- [ ] Galat fatal vs transien dibedakan **di kode**, bukan di komentar.
- [ ] Retry punya batas percobaan, batas jeda, dan `sleep_fn` yang bisa disuntik.
- [ ] Fallback chain melaporkan tautan yang menang & alasan tautan sebelumnya gagal.
- [ ] Cache di-key lengkap; `hit`, `miss`, `expired` dihitung terpisah.
- [ ] TTFT dan total_ms dilaporkan sebagai dua metrik berbeda.
- [ ] Argumen tool divalidasi; `maks_iterasi` ada; de-dup dari `tool_calls`.
- [ ] Pipeline melaporkan biaya/latensi **per langkah**.
- [ ] Ada penjaga anggaran yang **melewati** pekerjaan, bukan memaksa jalan.
- [ ] Angka-angka ini diuji **tanpa jaringan** (provider palsu deterministik).
