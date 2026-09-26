"""llmkit_ref — implementasi referensi project Bab 11 (LLM API & Orkestrasi).

Untuk cek mandiri SETELAH selesai. Verifier juga memakai modul ini untuk
mem-patch `llmkit` sebelum menjalankan test suite & notebook.

Semua stdlib murni; "provider" diambil dari `llmkit.transport` (palsu & deterministik).
"""
import hashlib
import json
import math

from llmkit.data import HARGA, TOOLS
from llmkit.transport import JENIS_TRANSIEN, GalatAPI

# ============================ COST ============================


def token_estimasi(teks):
    """Perkiraan token: ceil(len(teks) / 4)."""
    return int(math.ceil(len(str(teks)) / 4.0))


def hitung_biaya(tokens_in, tokens_out, model, harga=None):
    """Biaya USD dari tarif per 1 juta token. Model tak dikenal -> ValueError."""
    tabel = HARGA if harga is None else harga
    if model not in tabel:
        raise ValueError(f"model tidak ada di daftar harga: {model}")
    tarif = tabel[model]
    return (tokens_in / 1_000_000) * tarif["input"] + (tokens_out / 1_000_000) * tarif["output"]


def biaya_hasil(hasil, harga=None):
    """Biaya sebuah hasil panggilan (dict dari provider)."""
    return hitung_biaya(hasil["tokens_in"], hasil["tokens_out"], hasil["model"], harga)


def jalankan_anggaran(provider, daftar_pesan, batas_usd, *, model="mini",
                      timeout_ms=5000, harga=None):
    """Jalankan pekerjaan sampai anggaran HABIS; sisanya DILEWATI (bukan dipaksa).

    Return: diproses, dilewati, total_biaya_usd, hasil (list), berhenti_di (index|None).
    """
    diproses, dilewati = 0, 0
    total, hasil, berhenti_di = 0.0, [], None
    for i, pesan in enumerate(daftar_pesan):
        if total >= batas_usd:
            if berhenti_di is None:
                berhenti_di = i
            dilewati += 1
            continue
        r = provider.panggil(model, pesan, timeout_ms=timeout_ms)
        biaya = biaya_hasil(r, harga)
        total += biaya
        hasil.append({"index": i, "teks": r["teks"], "biaya_usd": biaya})
        diproses += 1
    return {"diproses": diproses, "dilewati": dilewati,
            "total_biaya_usd": total, "hasil": hasil, "berhenti_di": berhenti_di}


# ============================ RETRY + BACKOFF ============================


class GalatSemuaPercobaan(GalatAPI):
    """Semua percobaan gagal (atau galat fatal). `riwayat` = daftar (percobaan, jenis)."""

    def __init__(self, model, riwayat, pesan="semua percobaan gagal"):
        super().__init__("semua_gagal", model, pesan)
        self.riwayat = list(riwayat)


def backoff_ms(percobaan, basis_ms=1000, faktor=2, maks_ms=8000):
    """Exponential backoff: basis * faktor^(n-1), dibatasi maks_ms."""
    return min(basis_ms * (faktor ** (percobaan - 1)), maks_ms)


def panggil_dengan_retry(provider, model, messages, *, maks_percobaan=4, basis_ms=1000,
                         faktor=2, maks_ms=8000, timeout_ms=5000, sleep_fn=None,
                         **params):
    """Panggil provider dengan retry + backoff eksponensial.

    - Hanya galat TRANSIEN (rate_limit, server_error, timeout) yang dicoba ulang.
    - Galat fatal (bad_request / model tidak dikenal) langsung dilempar.
    - `sleep_fn(ms)` disuntik supaya test tidak benar-benar menunggu.
    """
    sleep_fn = (lambda ms: None) if sleep_fn is None else sleep_fn
    riwayat, tunggu = [], []
    for percobaan in range(1, maks_percobaan + 1):
        try:
            hasil = provider.panggil(model, messages, timeout_ms=timeout_ms, **params)
        except GalatAPI as e:
            riwayat.append((percobaan, e.jenis))
            transien = e.jenis in JENIS_TRANSIEN
            if not transien or percobaan == maks_percobaan:
                if not transien:
                    raise
                raise GalatSemuaPercobaan(model, riwayat) from e
            jeda = backoff_ms(percobaan, basis_ms, faktor, maks_ms)
            tunggu.append(jeda)
            sleep_fn(jeda)
            continue
        hasil = dict(hasil)
        hasil["attempts"] = percobaan
        hasil["timpenungguan_ms"] = list(tunggu)
        hasil["total_ms"] = hasil["latency_ms"] + sum(tunggu)
        hasil["riwayat"] = list(riwayat)
        return hasil
    raise GalatSemuaPercobaan(model, riwayat)


# ============================ FALLBACK ============================


def panggil_dengan_fallback(providers, rantai, messages, *, settings=None, **params):
    """Coba tiap tautan rantai sampai satu berhasil.

    `providers`: {"utama": ProviderPalsu, "cadangan": ProviderPalsu}
    `rantai`   : [{"provider": "utama", "model": "besar", "settings": {...}}, ...]
    """
    dasar = dict(settings or {})
    dicoba, riwayat = [], []
    for i, tautan in enumerate(rantai):
        nama, model = tautan["provider"], tautan["model"]
        provider = providers.get(nama)
        if provider is None:
            dicoba.append({"provider": nama, "model": model, "sukses": False,
                           "jenis": "provider_tidak_ada"})
            riwayat.append((i, model, "provider_tidak_ada"))
            continue
        setelan = {**dasar, **tautan.get("settings", {})}
        try:
            hasil = panggil_dengan_retry(provider, model, messages, **setelan, **params)
        except GalatAPI as e:
            dicoba.append({"provider": nama, "model": model, "sukses": False,
                           "jenis": e.jenis})
            riwayat.append((i, model, e.jenis))
            continue
        dicoba.append({"provider": nama, "model": model, "sukses": True,
                       "jenis": None, "attempts": hasil["attempts"]})
        hasil = dict(hasil)
        hasil["dicoba"] = dicoba
        hasil["lompatan"] = i
        hasil["tautan"] = tautan
        return hasil
    raise GalatSemuaPercobaan(rantai[-1]["model"] if rantai else "-", riwayat,
                              "seluruh rantai fallback gagal")


# ============================ CACHE ============================


class CachePrompt:
    """Cache konten-address dengan TTL dalam satuan 'tick' logis (bukan jam dinding)."""

    def __init__(self, ttl=3):
        self.ttl = ttl
        self._isi = {}          # kunci -> (hasil, tick_simpan)
        self.hit = 0
        self.miss = 0
        self.expired = 0

    def kunci(self, model, messages, params=None):
        bahan = json.dumps({"model": model, "messages": messages,
                            "params": params or {}}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(bahan.encode("utf-8")).hexdigest()[:16]

    def ambil(self, kunci, sekarang):
        if kunci not in self._isi:
            self.miss += 1
            return None
        hasil, tick = self._isi[kunci]
        if sekarang - tick >= self.ttl:
            self.expired += 1
            del self._isi[kunci]
            return None
        self.hit += 1
        return hasil

    def simpan(self, kunci, hasil, sekarang):
        self._isi[kunci] = (hasil, sekarang)

    def bersihkan(self, sekarang):
        kedaluwarsa = [k for k, (_, t) in self._isi.items() if sekarang - t >= self.ttl]
        for k in kedaluwarsa:
            del self._isi[k]
            self.expired += 1
        return len(kedaluwarsa)

    def statistik(self):
        total = self.hit + self.miss + self.expired
        return {"hit": self.hit, "miss": self.miss, "expired": self.expired,
                "ukuran": len(self._isi),
                "hit_rate": (self.hit / total) if total else 0.0}


def panggil_bercache(provider, model, messages, cache, sekarang, **params):
    """Panggil provider, lewati kalau cache masih segar. Hasil punya kunci 'cache'."""
    kunci = cache.kunci(model, messages, params)
    tersimpan = cache.ambil(kunci, sekarang)
    if tersimpan is not None:
        hasil = dict(tersimpan)
        hasil["cache"] = "hit"
        return hasil
    hasil = provider.panggil(model, messages, **params)
    cache.simpan(kunci, hasil, sekarang)
    hasil = dict(hasil)
    hasil["cache"] = "miss"
    return hasil


# ============================ STREAMING ============================


def konsumsi_stream(chunks):
    """Rakit chunk {'delta','t_ms'} menjadi teks + ukur TTFT.

    Return: teks, jumlah_chunk (delta non-kosong), ttft_ms, total_ms.
    """
    potongan, jumlah, ttft, terakhir = [], 0, None, 0
    for c in chunks:
        terakhir = c["t_ms"]
        if c["delta"]:
            jumlah += 1
            if ttft is None:
                ttft = c["t_ms"]
            potongan.append(c["delta"])
    return {"teks": "".join(potongan), "jumlah_chunk": jumlah,
            "ttft_ms": ttft, "total_ms": terakhir}


# ============================ TOOLS / FUNCTION CALLING ============================

TIPE_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def skema_parameter(nama, tools=None):
    """JSON-schema mini dari contoh argumen (`data.TOOLS[nama]['argumen']`)."""
    tabel = TOOLS if tools is None else tools
    argumen = tabel[nama]["argumen"]
    props = {k: {"type": TIPE_JSON.get(type(v), "string")} for k, v in argumen.items()}
    return {"type": "object", "properties": props,
            "required": list(argumen.keys())}


def skema_tools(tools=None):
    """Bentuk tools untuk API function calling."""
    tabel = TOOLS if tools is None else tools
    return [{"type": "function",
             "function": {"name": nama,
                          "description": spec["deskripsi"],
                          "parameters": skema_parameter(nama, tabel)}}
            for nama, spec in tabel.items()]


def validasi_argumen(nama, argumen, tools=None):
    """Daftar error ('missing:<k>', 'type:<k>', 'tak_dikenal:<k>')."""
    tabel = TOOLS if tools is None else tools
    if nama not in tabel:
        return [f"tak_dikenal:{nama}"]
    skema = skema_parameter(nama, tabel)
    errors = []
    if not isinstance(argumen, dict):
        return ["type:<argumen>"]
    for k in skema["required"]:
        if k not in argumen:
            errors.append(f"missing:{k}")
    for k, v in argumen.items():
        if k not in skema["properties"]:
            errors.append(f"tak_dikenal:{k}")
            continue
        tipe = skema["properties"][k]["type"]
        if not _tipe_ok(v, tipe):
            errors.append(f"type:{k}")
    return errors


def _tipe_ok(nilai, tipe):
    if tipe == "string":
        return isinstance(nilai, str)
    if tipe == "integer":
        return isinstance(nilai, int) and not isinstance(nilai, bool)
    if tipe == "number":
        return isinstance(nilai, (int, float)) and not isinstance(nilai, bool)
    if tipe == "boolean":
        return isinstance(nilai, bool)
    return False


def jalankan_tool(nama, argumen, tools=None):
    """Validasi argumen lalu eksekusi handler. Return {'hasil', 'galat', 'errors'}."""
    tabel = TOOLS if tools is None else tools
    errors = validasi_argumen(nama, argumen, tabel)
    if errors:
        return {"hasil": None, "galat": "; ".join(errors), "errors": errors}
    return {"hasil": tabel[nama]["handler"](argumen), "galat": None, "errors": []}


def loop_tool(provider, model, pesan_awal, *, tools=None, maks_iterasi=4,
              timeout_ms=5000, temperature=0.0, maks_token=200):
    """Loop function calling: model minta tool -> kita eksekusi -> hasil dikirim balik.

    Return: jawaban, iterasi, tool_dipakai, riwayat, tokens_in, tokens_out, total_ms.
    """
    skema = skema_tools(tools)
    messages = [{"role": "user", "content": pesan_awal}]
    tokens_in = tokens_out = total_ms = 0
    dipakai, riwayat = [], []

    for it in range(1, maks_iterasi + 1):
        hasil = provider.panggil(model, messages, tools=skema, timeout_ms=timeout_ms,
                                 temperature=temperature, maks_token=maks_token)
        tokens_in += hasil["tokens_in"]
        tokens_out += hasil["tokens_out"]
        total_ms += hasil["latency_ms"]

        if not hasil["tool_calls"]:
            return {"jawaban": hasil["teks"], "iterasi": it, "tool_dipakai": dipakai,
                    "riwayat": riwayat, "tokens_in": tokens_in,
                    "tokens_out": tokens_out, "total_ms": total_ms}

        for tc in hasil["tool_calls"]:
            messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
            res = jalankan_tool(tc["nama"], tc["argumen"], tools)
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(res["hasil"] if res["galat"] is None
                                                   else {"galat": res["galat"]})})
            dipakai.append(tc["nama"])
            riwayat.append({"iterasi": it, "tool": tc["nama"], "argumen": tc["argumen"],
                            "galat": res["galat"], "hasil": res["hasil"]})

    return {"jawaban": None, "iterasi": maks_iterasi, "tool_dipakai": dipakai,
            "riwayat": riwayat, "tokens_in": tokens_in, "tokens_out": tokens_out,
            "total_ms": total_ms, "berhenti": "maks_iterasi"}


# ============================ ORKESTRASI (PIPELINE) ============================


def jalankan_pipeline(provider, langkah, teks, *, maks_percobaan=3, timeout_ms=5000,
                      berhenti_saat_gagal=False, settings=None):
    """Prompt chaining: jalankan langkah berurutan, laporkan tiap langkah.

    Return: langkah (detail per langkah), berhasil, gagal, total_biaya_usd,
    total_ms, teks_akhir.
    """
    dasar = dict(settings or {})
    rincian, total_biaya, total_ms = [], 0.0, 0
    berhasil = gagal = 0
    teks_akhir = None

    for lang in langkah:
        prompt = lang["template"].format(teks=teks)
        messages = [{"role": "user", "content": prompt}]
        try:
            hasil = panggil_dengan_retry(
                provider, lang["model"], messages,
                maks_percobaan=maks_percobaan, timeout_ms=timeout_ms,
                temperature=lang.get("suhu", 0.0), maks_token=lang.get("maks_token", 200),
                **dasar)
            biaya = biaya_hasil(hasil)
            total_biaya += biaya
            total_ms += hasil["total_ms"]
            berhasil += 1
            teks_akhir = hasil["teks"]
            rincian.append({"nama": lang["nama"], "model": lang["model"], "sukses": True,
                            "teks": hasil["teks"], "attempts": hasil["attempts"],
                            "total_ms": hasil["total_ms"], "biaya_usd": biaya,
                            "galat": None})
        except GalatAPI as e:
            gagal += 1
            rincian.append({"nama": lang["nama"], "model": lang["model"], "sukses": False,
                            "teks": None, "attempts": None, "total_ms": 0,
                            "biaya_usd": 0.0, "galat": e.jenis})
            if berhenti_saat_gagal:
                break

    return {"langkah": rincian, "berhasil": berhasil, "gagal": gagal,
            "total_biaya_usd": total_biaya, "total_ms": total_ms,
            "teks_akhir": teks_akhir}
