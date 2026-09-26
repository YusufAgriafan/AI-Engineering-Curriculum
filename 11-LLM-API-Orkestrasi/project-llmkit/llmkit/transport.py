"""Provider LLM PALSU & deterministik — GIVEN, JANGAN DIUBAH.

Kenapa palsu? Karena pola produksi (retry, backoff, fallback, cache, budget)
hanya bisa dilatih kalau **kegagalan bisa dijadwalkan**. Provider ini:

- menjawab deterministik dari prompt (tanpa jaringan, tanpa API key);
- gagal sesuai `data.RENCANA` (rate limit, server error, timeout, bad request);
- mencatat biaya token & latensi yang bisa dihitung ulang oleh kodemu;
- bisa "memanggil tool" kalau prompt memuat kata kunci tool (`data.TOOLS`).

Semua perilakunya TERBUKA — tidak ada kotak hitam.
"""
import math
import time

from .data import (BASIS_LATENSI, HARGA, HARGA_CADANGAN, MS_PER_CHUNK, RENCANA,
                   RENCANA_CADANGAN, SEED, TOOLS, TTFT_MS)


# ============================ keluarga galat ============================

class GalatAPI(Exception):
    """Induk semua galat provider. `jenis` dipakai klien untuk memutuskan retry."""

    def __init__(self, jenis, model, pesan, percobaan=1):
        super().__init__(f"[{jenis}] {model}: {pesan}")
        self.jenis = jenis
        self.model = model
        self.pesan = pesan
        self.percobaan = percobaan


class GalatRateLimit(GalatAPI):
    def __init__(self, model, pesan="terlalu banyak permintaan", percobaan=1):
        super().__init__("rate_limit", model, pesan, percobaan)


class GalatServer(GalatAPI):
    def __init__(self, model, pesan="kesalahan internal provider", percobaan=1):
        super().__init__("server_error", model, pesan, percobaan)


class GalatTimeout(GalatAPI):
    def __init__(self, model, pesan="permintaan melewati batas waktu", percobaan=1):
        super().__init__("timeout", model, pesan, percobaan)


class GalatPermintaanBuruk(GalatAPI):
    def __init__(self, model, pesan="permintaan tidak valid", percobaan=1):
        super().__init__("bad_request", model, pesan, percobaan)


# jenis galat yang MASUK AKAL dicoba ulang (transient)
JENIS_TRANSIEN = ("rate_limit", "server_error", "timeout")


# ============================ provider palsu ============================

def _teks_prompt(messages):
    """Gabung seluruh pesan menjadi satu teks prompt (untuk hitung token & deteksi)."""
    potongan = []
    for m in messages:
        isi = m.get("content") or ""
        potongan.append(str(isi))
    return "\n".join(potongan)


def _inti_pertanyaan(messages):
    """Baris terakhir dari pesan USER terakhir = 'pertanyaan' yang dijawab model."""
    for m in reversed(messages):
        if m.get("role") == "user":
            baris = [b for b in str(m.get("content") or "").strip().splitlines() if b.strip()]
            if baris:
                return baris[-1].strip()[:50]
    return ""


def _balasan(model, messages):
    """Jawaban deterministik: potongan pertanyaan terakhir + nama model."""
    return f'[{model}] jawaban untuk "{_inti_pertanyaan(messages)}"'


def _sudah_dipakai(messages, nama):
    """Apakah tool `nama` sudah pernah dipanggil di percakapan ini?

    Dicek dari assistant message yang membawa `tool_calls` — BUKAN dari teks
    (teks prompt sendiri memuat kata kunci tool, jadi substring check salah).
    """
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if tc.get("nama") == nama:
                return True
        if m.get("role") == "tool" and m.get("nama_tool") == nama:
            return True
    return False


def _tool_berikut(prompt, messages):
    """Tool pertama (menurut POSISI kata kunci di prompt) yang belum dipanggil."""
    low = prompt.lower()
    kandidat = []
    for nama, spec in TOOLS.items():
        if _sudah_dipakai(messages, nama):
            continue
        posisi = [low.find(k) for k in spec["kata_kunci"] if k in low]
        if posisi:
            kandidat.append((min(posisi), nama))
    if not kandidat:
        return None
    kandidat.sort()
    return kandidat[0][1]


class ProviderPalsu:
    """Provider palsu stateful: hitungan panggilan per model = jam kegagalannya."""

    def __init__(self, nama="utama", harga=None, rencana=None, seed=SEED):
        self.nama = nama
        self.harga = dict(HARGA if harga is None else harga)
        self.rencana = dict(RENCANA if rencana is None else rencana)
        self.seed = seed
        self.jumlah_panggilan = {}   # model -> berapa kali sudah dipanggil
        self.log = []                # audit semua panggilan
        self._urut_tool = 0

    # -------------------- status kegagalan --------------------
    def status_berikut(self, model):
        """Status panggilan berikutnya untuk model ini (habis daftar -> 'ok')."""
        daftar = self.rencana.get(model)
        if daftar is None:
            return "model_tidak_dikenal"
        n = self.jumlah_panggilan.get(model, 0)
        return daftar[n] if n < len(daftar) else "ok"

    def latensi_untuk(self, model, prompt):
        """Latensi deterministik (ms)."""
        return BASIS_LATENSI.get(model, 200) + (len(prompt) % 50)

    # -------------------- panggilan utama --------------------
    def panggil(self, model, messages, *, timeout_ms=5000, tools=None,
                temperature=0.0, maks_token=200):
        """Return dict hasil. Raise salah satu galat di atas bila gagal.

        Hasil: teks, tokens_in, tokens_out, latency_ms, model, tool_calls
        """
        n = self.jumlah_panggilan.get(model, 0)
        status = self.status_berikut(model)      # status untuk panggilan KALI INI
        percobaan = n + 1
        self.jumlah_panggilan[model] = percobaan
        prompt = _teks_prompt(messages)
        latensi = self.latensi_untuk(model, prompt)

        catatan = {"model": model, "status": status, "percobaan": percobaan,
                   "latency_ms": latensi, "tokens_in": math.ceil(len(prompt) / 4)}
        self.log.append(catatan)

        if status == "model_tidak_dikenal":
            raise GalatPermintaanBuruk(model, "model tidak ada di daftar harga", percobaan)
        if status == "bad_request":
            raise GalatPermintaanBuruk(model, "permintaan tidak valid", percobaan)
        if status == "rate_limit":
            raise GalatRateLimit(model, "kuota per menit terlampaui", percobaan)
        if status == "server_error":
            raise GalatServer(model, "kesalahan internal 500", percobaan)
        if status == "timeout":
            raise GalatTimeout(model, "tidak ada respons sebelum batas waktu", percobaan)
        if latensi > timeout_ms:
            raise GalatTimeout(model, f"latensi {latensi}ms > timeout {timeout_ms}ms", percobaan)

        # ---- tool calling (deterministik, satu tool per panggilan) ----
        tool_calls = []
        if tools:
            nama = _tool_berikut(prompt, messages)
            if nama is not None:
                self._urut_tool += 1
                tool_calls.append({
                    "id": f"call_{self._urut_tool}",
                    "nama": nama,
                    "argumen": dict(TOOLS[nama]["argumen"]),
                })

        if tool_calls:
            teks = None
            tokens_out = 12
        else:
            teks = _balasan(model, messages)
            tokens_out = math.ceil(len(teks) / 4)

        return {
            "provider": self.nama,
            "model": model,
            "teks": teks,
            "tool_calls": tool_calls,
            "tokens_in": math.ceil(len(prompt) / 4),
            "tokens_out": tokens_out,
            "latency_ms": latensi,
            "attempts": percobaan,
        }

    # -------------------- streaming --------------------
    def stream(self, model, messages, *, timeout_ms=5000, temperature=0.0, maks_token=200):
        """Hasilkan chunk {'delta', 't_ms'} deterministik.

        Chunk pertama muncul di TTFT_MS, sisanya setiap MS_PER_CHUNK per kata.
        """
        hasil = self.panggil(model, messages, timeout_ms=timeout_ms,
                             temperature=temperature, maks_token=maks_token)
        t = TTFT_MS
        for i, kata in enumerate(str(hasil["teks"]).split()):
            yield {"delta": (" " if i else "") + kata, "t_ms": t}
            t += MS_PER_CHUNK
        yield {"delta": "", "t_ms": t}


def provider_utama():
    return ProviderPalsu("utama", HARGA, RENCANA)


def provider_cadangan():
    return ProviderPalsu("cadangan", HARGA_CADANGAN, RENCANA_CADANGAN)


def jam():
    """Sumber waktu — dipakai bila kodemu butuh mengukur latensi nyata."""
    return time.perf_counter()
