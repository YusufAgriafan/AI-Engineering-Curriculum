# Bab 11 — Bekerja dengan LLM API & Orkestrasi

> Dari "prompt yang jalan di playground" ke "kode yang bertahan saat provider
> gagal": biaya yang bisa dihitung sebelum memanggil, retry yang tahu kapan
> berhenti, fallback yang sadar harga, cache yang tidak membayar dua kali, dan
> pipeline yang bisa diaudit per langkah.

## 🎯 Tujuan Belajar

- Memanggil LLM API dengan benar: peran pesan, parameter, timeout, API key dari
  environment — dan tahu **kapan** streaming dipakai.
- Menghitung **biaya & anggaran** sebelum memanggil, lalu menegakkannya sebagai
  fitur produk (melewati pekerjaan, bukan memaksa jalan).
- Menjalankan **retry yang berdisiplin**: membedakan galat transien vs fatal,
  menjadwalkan backoff, dan tidak menunggu setelah percobaan terakhir.
- Merancang **fallback chain multi-provider** dan mengukur apa yang dikorbankan
  (kualitas, konsistensi) demi ketersediaan.
- Merancang **cache yang tidak berbohong**: kunci lengkap, TTL, dan hit-rate yang
  dipisahkan dari expired.
- Mengukur **TTFT** vs total latency, dan memutuskan streaming berdasarkan itu.
- Mengimplementasikan **function calling** dengan validasi argumen di kode kita,
  plus batas iterasi yang wajib.
- Memecah satu tugas besar menjadi **pipeline** berlangkah dengan laporan biaya,
  latency, dan attempts per langkah.

## 🗂️ Materi Pendukung Bab Ini

| Materi | File | Isi |
|---|---|---|
| 🧪 Lab | [`01_lab_llm_api.ipynb`](01_lab_llm_api.ipynb) | 8 bagian: provider palsu & rencana kegagalan → biaya & anggaran → retry/backoff → fallback → cache → streaming → tools → pipeline |
| 📝 Kuis | [`02_kuis_llm_api.ipynb`](02_kuis_llm_api.ipynb) | 10 soal (6 PG + 4 coding), 22 poin, dinilai otomatis, **mandiri** (tanpa jaringan) |
| 🔑 Kunci | [`03_kunci_jawaban_kuis_llm_api.ipynb`](03_kunci_jawaban_kuis_llm_api.ipynb) | Jawaban + **kenapa**, kode referensi tiap soal coding |
| 🏗️ Project | [`project-llmkit/`](project-llmkit/README.md) | Paket `llmkit` TDD **107 test**: cost, retry, fallback, cache, stream, tools, pipeline + 2 notebook |
| 📄 Cheatsheet | [`cheatsheet-llm-api.md`](cheatsheet-llm-api.md) | Ringkas 1 halaman untuk review cepat |

Materi lab & project **tanpa API key**: "provider" diganti
`project-llmkit/llmkit/transport.py` — provider palsu deterministik yang bisa
**disuruh gagal** (`rate_limit` di panggilan ke-1, `server_error` di ke-2, pulih
di ke-3). Angka terkunci seed 11 dan sudah diverifikasi otomatis
(`_verify_lab.py`, `_verify_quiz.py`, `project-llmkit/_verify_project.py`).

---

## 1. Materi Inti

### 1.1 Peta Lapisan: Panggilan API Itu Bagian Terkecil

```
┌───────────────────────────────────────────────────────────────┐
│  APLIKASI         produk, endpoint, kuota per tenant          │
├───────────────────────────────────────────────────────────────┤
│  ORKESTRASI       prompt chaining, laporan per langkah (§1.10)│
├───────────────────────────────────────────────────────────────┤
│  KEAMANAN/TOOLS   validasi argumen, least-privilege (§1.9)    │
├───────────────────────────────────────────────────────────────┤
│  TAHAN GAGAL      retry (§1.5) · fallback (§1.6)              │
├───────────────────────────────────────────────────────────────┤
│  EFISIENSI        cache (§1.7) · anggaran (§1.3)              │
├───────────────────────────────────────────────────────────────┤
│  PANGGILAN        client.chat.completions.create(...)  ← §1.2 │
└───────────────────────────────────────────────────────────────┘
```

Baris paling bawah biasanya bagian termudah — dan paling sedikit baris kodenya.
Sisa bab ini tentang lapisan di atasnya, karena **itulah yang membedakan prototipe
dari sistem**.

Bab ini melatih lapisan tengah dengan provider palsu yang deterministik (mereka
diuji tepat di titik kegagalannya), lalu menunjukkan bentuknya pada provider
nyata.

---

### 1.2 Anatomi Satu Panggilan

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])   # JANGAN hardcode

response = client.chat.completions.create(
    model="gpt-4o-mini",              # pilih per tugas, bukan per gengsi
    messages=[
        {"role": "system", "content": "Kamu asisten yang menjawab ringkas dan akurat."},
        {"role": "user",   "content": "Jelaskan apa itu machine learning dalam 2 kalimat."},
    ],
    temperature=0.0,                  # ekstraksi/klasifikasi: 0–0.2
    max_tokens=500,                   # batasi output = batasi biaya
    timeout=30,                       # timeout per request
)
print(response.choices[0].message.content)
print(response.usage)                 # prompt_tokens, completion_tokens
```

| Peran | Isi | Aturan yang tidak boleh dilanggar |
|---|---|---|
| `system` | aturan, persona, format output, larangan | **jangan pernah** menyambung data pengguna ke sini |
| `user` | permintaan + data tak tepercaya | bungkus dengan delimiter + leash (Bab 10 §1.6) |
| `assistant` | jawaban sebelumnya / `tool_calls` | jadi bukti tool sudah dipanggil (§1.9) |
| `tool` | hasil eksekusi tool | `tool_call_id` harus cocok dengan usulan model |

Semua provider memakai pola yang sama (OpenAI, Anthropic, Gemini, DeepSeek);
LiteLLM menyeragamkan antarmukanya. **API key selalu dari environment variable** —
`.env` masuk `.gitignore`, dan rotasi kunci kalau pernah ter-commit.

> Di lab & project, "client" ini digantikan `ProviderPalsu`. Ia mengembalikan
> metadata yang sama (`tokens_in`, `tokens_out`, `latency_ms`) sehingga **seluruh
> kode lapisan atas di bawah ini bisa diuji tanpa jaringan**.

---

### 1.3 Biaya & Anggaran: Hitung Sebelum Memanggil

```python
def token_estimasi(teks):                      # cukup untuk ANGGARAN, bukan tagihan
    return int(math.ceil(len(str(teks)) / 4.0))

def hitung_biaya(tokens_in, tokens_out, model, harga):
    t = harga[model]                           # USD per 1 JUTA token
    return tokens_in / 1e6 * t["input"] + tokens_out / 1e6 * t["output"]
```

Angka terkunci (lab Bagian 1): `hitung_biaya(1000, 300, ...)` → `mini` **$0.00033**
vs `besar` **$0.0095** — **~29x**. Model besar bukan "lebih baik"; ia kategori
harga lain.

Anggaran harus **ditegakkan di kode**, bukan diingat:

```python
def jalankan_anggaran(provider, daftar_pesan, batas_usd, model="mini"):
    diproses, dilewati, total, berhenti_di = 0, 0, 0.0, None
    for i, pesan in enumerate(daftar_pesan):
        if total >= batas_usd:                 # ← STOP MEMANGGIL, bukan "paksa jalan"
            berhenti_di = i if berhenti_di is None else berhenti_di
            dilewati += 1
            continue
        r = provider.panggil(model, pesan)
        total += hitung_biaya(r["tokens_in"], r["tokens_out"], model)
        diproses += 1
    return {"diproses": diproses, "dilewati": dilewati,
            "total_biaya_usd": total, "berhenti_di": berhenti_di}
```

Angka terkunci: anggaran kecil untuk 4 pekerjaan → `diproses 3 | dilewati 1 |
berhenti_di 3`, total `4.005e-05`, biaya per hasil `1.335e-05`.

**Lima cara menekan biaya** (urut dampak):

| Cara | Potensi | Catatan |
|---|---|---|
| Model yang tepat | 5–50x | task mudah jangan pakai model besar |
| Kurangi token input | 2–10x | retrieval tepat (Bab 12), prompt ringkas, cache |
| Kurangi token output | 2–5x | `max_tokens`, minta output padat |
| Cache respons | 10–100x untuk query berulang | §1.7 |
| Routing per tingkat kesulitan | 3–20x | classifier kecil memilih model |

---

### 1.4 Structured Output di API Nyata

Bab 10 melatih parse → validasi → retry dengan mock. Di API nyata, tambahkan
jaminan dari provider:

```python
from pydantic import BaseModel, Field

class Sentimen(BaseModel):
    label: str = Field(..., description="positive/negative/neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    alasan: list[str] = Field(default_factory=list)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},   # atau json_schema ketat
    temperature=0.0,
)
try:
    hasil = Sentimen.model_validate_json(resp.choices[0].message.content)
except Exception as e:
    # self-healing loop: kirim error skema kembali ke model (maks 1–2 kali)
    ...
```

Empat lapis tetap sama seperti Bab 10, hanya titiknya bergeser:

```
1. PROMPT        : skema field + "balas JSON saja"
2. PROVIDER      : response_format / json_schema  ← jaminan tambahan, bukan pengganti
3. PARSER        : ekstrak_json (jaring pengaman untuk output kotor)
4. VALIDATOR+RETRY: skema + pesan error yang informatif -> model memperbaiki
```

---

### 1.5 Retry & Backoff: Dua Keputusan, Bukan Satu Loop

```
JENIS_TRANSIEN = ("rate_limit", "server_error", "timeout")   → ulangi
bad_request / model tidak ada / skema salah                  → LEMPAR sekarang
```

```python
@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)))
def panggil(**kwargs):
    return client.chat.completions.create(**kwargs)
```

Kalau kamu menulisnya sendiri (seperti di project), urutannya wajib persis:

```
1. panggil provider
2. catat (percobaan, jenis) ke riwayat
3. galat FATAL?           -> raise SEKARANG
4. percobaan TERAKHIR?    -> raise GalatSemuaPercobaan  (JANGAN menunggu dulu)
5. sisanya                -> sleep(backoff(percobaan)) lalu ulangi
```

```python
def backoff_ms(percobaan, basis_ms=1000, faktor=2, maks_ms=8000):
    return min(basis_ms * faktor ** (percobaan - 1), maks_ms)   # [1000,2000,4000,8000,8000]
```

Angka terkunci: `sedang` gagal sekali → `attempts 2`, tunggu `[1000]`, latency
240 ms → `total_ms` **1240 ms**. `bad_request` → **1 panggilan** (tidak diulang).

Dua hal yang sering lupa di produksi:

- **`sleep_fn` disuntik** — supaya test cepat; jalur produksi memakai `time.sleep`.
- **Jitter** — `jeda + rng.randint(0, jeda // 4)`; tanpa itu ribuan klien bangun
  bersamaan (thundering herd) dan memukul ulang provider di detik yang sama.
- **Circuit breaker** — kalau provider sedang down total, berhenti mencoba lebih
  murah daripada mencoba terus.

---

### 1.6 Fallback Chain & Routing Model

```python
RANTAI = [{"provider": "utama",    "model": "besar"},     # kualitas dulu
          {"provider": "utama",    "model": "mini"},      # kompromi
          {"provider": "cadangan", "model": "murah"}]     # ketersediaan

def panggil_dengan_fallback(providers, rantai, messages, settings=None):
    for i, tautan in enumerate(rantai):
        setelan = {**(settings or {}), **tautan.get("settings", {})}
        try:
            hasil = panggil_dengan_retry(providers[tautan["provider"]],
                                        tautan["model"], messages, **setelan)
        except GalatAPI as e:
            dicoba.append({..., "jenis": e.jenis})       # ← BEKAS KEGAGALAN DICATAT
            continue
        hasil["dicoba"], hasil["lompatan"] = dicoba, i    # ← audit turun-kualitas
        return hasil
    raise GalatSemuaPercobaan(riwayat)
```

Angka terkunci: `besar` gagal dua kali (`rate_limit` + `server_error`), rantai
menang di tautan ke-1 (`mini`); `dicoba` mencatat `(besar, semua_gagal)` lalu
`(mini, success)`. Rasio biaya `besar`:`mini` ≈ **26x**.

Yang dibayar: kualitas penalaran & **konsistensi output**. Untuk tugas yang
hasilnya diparse/dibandingkan (JSON ber-skema, skor), ganti model di tengah jalan
adalah sumber regresi — kadang **error 503 + retry di sisi klien lebih jujur**
daripada hasil yang tampak sukses tapi beda kualitas.

---

### 1.7 Cache: Kunci Lengkap, TTL Logis

```
kunci = sha256(model + messages + params)[:16]
       + (produksi) versi prompt · konteks retrieval (Bab 12) · tenant id
hit ≠ expired      -> keduanya menghasilkan "tidak ada cache", tapi artinya berbeda
```

```python
def ambil(self, kunci, sekarang):
    if kunci not in self._isi:
        self.miss += 1
        return None
    hasil, tick = self._isi[kunci]
    if sekarang - tick >= self.ttl:      # KEDALUWARSA ≠ MISS
        self.expired += 1
        del self._isi[kunci]
        return None
    self.hit += 1
    return hasil
```

Angka terkunci (tick 0,1,5,6 dengan `ttl=3`): pola `miss, hit, miss, hit`,
`hit_rate 0.5` (hit 2, miss 1, expired 1), dan **4 permintaan = 2 panggilan
provider**.

TTL di sini memakai **tick logis** supaya bisa diuji deterministik — pola yang
sama dipakai untuk test TTL produksi (tanpa `sleep`).

Di produksi, lapisan berikutnya:

| Tingkat | Cara kerja | Kapan |
|---|---|---|
| Tepat | hash identik (di atas) | retry, pengulangan eksak |
| Semantic | embedding + ambang kemiripan | pertanyaan mirip-mirip (FAQ) |
| Provider prompt cache | prefix prompt yang sama di-cache provider | system prompt panjang berulang |

---

### 1.8 Streaming: TTFT vs Total

```python
stream = client.chat.completions.create(..., stream=True)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

| Metrik | Artinya | Gunanya |
|---|---|---|
| **TTFT** | waktu sampai token pertama terlihat | persepsi responsif (UX) |
| **total_ms** | sampai respons selesai | throughput & biaya |

Angka terkunci: **TTFT 120 ms** dari **total 340 ms** (11 chunk) → pengguna melihat
kata pertama ~65% lebih cepat, **total prosesnya identik**.

Streaming **tidak** menghapus waktu tunggu, hanya memindahkannya. Kapan tidak
layak: output JSON pendek yang baru berguna setelah utuh, batch/async yang tidak
ditunggu manusia, atau parsing yang butuh hasil lengkap.

---

### 1.9 Function Calling: Model Mengusulkan, Kode Memutuskan

```
model -> tool_calls: [{id, nama, argumen}]        USULAN (data tak tepercaya)
kode  -> validasi_argumen -> jalankan_tool         KEPUTUSAN
kode  -> {role:'tool', tool_call_id, content}      OBSERVASI dikirim balik
```

```python
res = jalankan_tool(tc["nama"], tc["argumen"])
if res["galat"]:
    content = json.dumps({"galat": res["galat"]})     # error dikirim BALIK ke model
else:
    content = json.dumps(res["hasil"])
```

Aturan yang tidak boleh dilanggar:

1. **Validasi dulu, eksekusi kemudian.** Error stabil: `missing:<k>`, `type:<k>`,
   `tak_dikenal:<k>` (bahannya retry ke model).
2. **Jangan pernah** `eval(argumen["expression"])`. Contoh di banyak tutorial
   (termasuk versi awal README ini) adalah pintu RCE: model memegang kendali kode.
3. **`maks_iterasi` wajib.** Agent tanpa batas = tagihan tanpa batas.
4. **De-dup tool dari `tool_calls`**, bukan dari substring prompt. Prompt-mu sendiri
   memuat kata kunci tool, jadi pengecekan teks akan selalu salah.
5. **Log semua pemanggilan tool** — nama, argumen, hasil, siapa yang meminta.

Angka terkunci: dua tool → **iterasi 3** (`['cari_catatan','hitung']`), tanpa tool →
iterasi 1, `maks_iterasi=1` → `berhenti='maks_iterasi'` dan `jawaban=None`.

> Ini fondasi agent (Bab 14). Bedanya nanti: jumlah tool, memori, dan sandboxing
> (least-privilege) — bukan bentuk loop-nya.

---

### 1.10 Orkestrasi: Satu Tugas Besar → Langkah yang Bisa Diaudit

```
❌ satu prompt: "baca tiket, klasifikasi, ekstrak pesanan, tulis balasan, cek stok"
✅ pipeline:
   klasifikasi (mini,   T=0.0)   -> {intent}
   ekstraksi   (mini,   T=0.0)   -> JSON
   balasan     (sedang, T=0.4)   -> teks      ← satu-satunya yang butuh kreativitas
```

Angka terkunci (3 langkah, satu tiket):

```
langkah       model   attempts   total_ms   biaya
klasifikasi   mini    1          184        1.530e-05
ekstraksi     mini    1          187        1.545e-05
balasan       sedang  2          1249       1.300e-04
------------------------------------------------------
total                           1620       1.6075e-04
```

Dengan `maks_percobaan=1`: `berhasil 2 | gagal 1`, langkah `balasan` melaporkan
`galat='semua_gagal'` dan biaya `$0`. Itulah nilainya: **kegagalan terlokalisasi**
dan alert bisa menyala di langkah yang benar — tanpa laporan per langkah kamu hanya
tahu "totalnya lambat".

Biayanya: lebih banyak panggilan (latency & token). Trade-off yang **diukur**,
bukan dirasa.

---

### 1.11 Observability: Apa yang Wajib Dilog

```python
log = {
  "request_id": ..., "tenant": ...,
  "provider": "utama", "model": "mini",
  "attempts": 2, "riwayat": [(1, "rate_limit")],
  "fallback_lompatan": 1,                 # 0 = tautan pertama yang menang
  "cache": "miss",                        # hit | miss | expired
  "tokens_in": 13, "tokens_out": 19,
  "biaya_usd": 1.335e-05, "total_ms": 1240,
  "dipakai_retry": True,
}
```

Tanpa angka-angka ini, kamu tidak bisa menjawab pertanyaan yang paling sering
datang dari manajemen: *"biaya kita bulan ini naik kenapa?"* dan *"kenapa lambat
untuk sebagian pengguna?"*. Bab 15 & 16 melanjutkan ini menjadi metrik yang
dipantau terus-menerus dan kebijakan gateway (kuota per tenant).

---

### 1.12 Anti-Pattern

| Anti-pattern | Gejala | Solusi |
|---|---|---|
| **Retry untuk semua galat** | `bad_request` diulang 5x | klasifikasikan transien vs fatal |
| **Backoff tanpa batas atas** | percobaan ke-10 menunggu menit-an | `maks_ms` + jitter |
| **Menunggu setelah percobaan terakhir** | jeda sia-sia sebelum error | cek urutan sebelum `sleep` |
| **Kunci cache kurang lengkap** | jawaban basi / salah model | model+pesan+params+versi prompt+konteks+tenant |
| **Fallback disembunyikan** | kualitas turun tanpa jejak | log `dicoba` & `lompatan` |
| **Loop tool tanpa batas** | biaya/latensi tak terbatas | `maks_iterasi` + anggaran + timeout |
| **`eval` argumen tool** | pintu RCE | skema → validasi → whitelist handler |
| **API key hardcode** | kredensial bocor | env var + `.env` di-gitignore |
| **Streaming untuk JSON pendek** | kompleksitas tanpa manfaat | streaming hanya untuk UX |
| **Biaya tidak dihitung** | tagihan sebagai kejutan | `biaya_hasil` + penjaga anggaran |
| **Satu prompt raksasa** | sulit dilokalisasi saat gagal | pipeline berlangkah (§1.10) |
| **Tanpa timeout** | satu request menggantung di worker | timeout per request + total |

---

## 2. Latihan Praktis

Materi latihan bab ini sudah lengkap sebagai file terpisah (lihat tabel di atas).
Urutan yang disarankan:

1. **Lab** — [`01_lab_llm_api.ipynb`](01_lab_llm_api.ipynb): bangun delapan
   komponen lapisan produksi dari nol di atas provider palsu yang bisa disuruh
   gagal; selesaikan semua cell **✅ Cek** lalu jawab 6 pertanyaan analisis.
2. **Kuis** — [`02_kuis_llm_api.ipynb`](02_kuis_llm_api.ipynb): 22 poin, dinilai
   otomatis, **mandiri** (provider uji sudah disediakan). Baru setelah mencoba:
   buka [kunci jawaban](03_kunci_jawaban_kuis_llm_api.ipynb).
3. **Project** — [`project-llmkit/`](project-llmkit/README.md): paket `llmkit`
   (107 test, mode TDD) + notebook eksperimen + `RUBRIK.md`.

Tiga latihan ringkas yang bisa dikerjakan tanpa notebook:

1. **Hitung biaya sistemmu.** Ambil satu fitur LLM yang sudah kamu punya. Ukur
   `tokens_in`/`tokens_out` rata-rata, hitung biaya per request, kalikan dengan
   trafik harian. Lalu tulis **tiga** perubahan yang menurunkannya — sebelum
   menyentuh kode.
2. **Jadwalkan kegagalan.** Ambil satu fungsi pemanggil LLM di kodemu, suntik
   `sleep_fn`/provider palsu, lalu tulis test untuk: transien pulih di percobaan
   ke-2, fatal langsung gagal, dan percobaan habis. Kalau test ini sulit ditulis,
   berarti desainnya belum bisa diuji.
3. **Audit kunci cache.** Cari semua tempat di sistemmu yang men-cache keluaran
   LLM. Untuk masing-masing, tulis daftar lengkap yang masuk kunci. Setiap item
   yang lupa di daftar itu adalah bug yang belum terjadi.

---

## 📚 Referensi Online

| Sumber | Tipe | Keterangan |
|---|---|---|
| [OpenAI API Reference](https://platform.openai.com/docs/api-reference) | Dokumentasi | Referensi lengkap: chat, streaming, tools, structured output |
| [OpenAI Cookbook](https://cookbook.openai.com/) | Contoh | Pola produksi: retry, batching, evaluasi |
| [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) | Dokumentasi | Tool use & prompt caching (prefix cache) |
| [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs) | Dokumentasi | Multimodal, context window besar |
| [LiteLLM](https://docs.litellm.ai/) | Library | Antarmuka provider-agnostic: retry, fallback, anggaran |
| [Tenacity](https://tenacity.readthedocs.io/) | Library | Retry + backoff deklaratif untuk Python |
| [Pydantic](https://docs.pydantic.dev/) | Library | Validasi skema output LLM |
| [Vercel AI SDK — streaming & tool calls](https://sdk.vercel.ai/docs) | Dokumentasi | Pola streaming di sisi klien (TTFT) |

---

## ✅ Checklist Kompetensi

- [ ] API key dari environment variable; tidak ada kredensial di kode/commit
- [ ] Bisa menghitung biaya per request **sebelum** memanggil, dan menganggarkan
      per hari/per tenant
- [ ] Ada penjaga anggaran yang **melewati** pekerjaan saat batas tercapai
- [ ] Retry membedakan galat transien vs fatal, dengan backoff terbatas + jitter
- [ ] Tidak ada jeda sia-sia setelah percobaan terakhir
- [ ] Fallback chain ada, dan `dicoba`/`lompatan`-nya **dilog**
- [ ] Cache di-key lengkap (model + pesan + parameter + versi prompt + konteks);
      `hit`, `miss`, `expired` dihitung terpisah
- [ ] Melaporkan TTFT **dan** total latency sebagai dua metrik berbeda
- [ ] Output JSON dijamin valid: `response_format`/skema + parser + retry
- [ ] Argumen tool divalidasi di kode kita; tidak ada `eval` pada data dari model
- [ ] `maks_iterasi` ada di setiap loop tool; de-dup lewat assistant `tool_calls`
- [ ] Pipeline melaporkan biaya/latency/attempts **per langkah**
- [ ] Timeout per request dan timeout total dipasang
- [ ] Menjalankan lab (semua ✅ Cek), kuis 22/22, dan project 107 test hijau

Selesai? Lanjut ke **Bab 12 — RAG (Retrieval Augmented Generation)**: `konteks`
yang di sini masih placeholder akan benar-benar diambil dari dokumen nyata — dan
yang kamu bangun di bab ini (cache yang sadar konteks, anggaran, fallback, laporan
per langkah) langsung menjadi rangka operasionalnya.
