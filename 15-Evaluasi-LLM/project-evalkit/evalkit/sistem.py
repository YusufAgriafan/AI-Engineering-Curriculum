"""GIVEN — Tiga versi "sistem LLM" yang akan dievaluasi (JANGAN DIUBAH).

Kontrak satu fungsi untuk semua versi:

    jawab(teks) -> {'jawaban': str, 'skor_retrieval': float, 'meta': dict}

Deterministik penuh: input sama → output sama, di komputer siapa pun.
Itulah syarat mesin eval bisa diuji dengan angka terkunci.

- v1 : baseline rapuh — generator ekstraktif murni; tidak tahu kapan harus
       diam; injection dilayani; input kosong menghasilkan "jawaban" gema.
- v2 : + abstain saat skor retrieval di bawah ambang (Bab 12) — lebih jujur,
       TAPI injection yang kaya kata kunci (skor ≥ ambang) tetap dilayani.
       (Pelajaran: abstain-gate retrieval bukan guardrail keamanan.)
- v3 : + deteksi injection → tolak eksplisit (gaya agentkit Bab 14) +
       input kosong diperlakukan abstain.

Meta tiap hasil selalu memuat 'versi' — runner mencatatnya ke hasil run.
"""
from .data import AMBAT_SKOR, KB

# Frasa abstain yang KONSISTEN — scorer mencarinya (jangan diganti-ganti).
FRASA_TIDAK_TAHU = "Maaf, saya tidak memiliki informasi tentang hal tersebut di dokumen."

# Frasa injection yang dinetralkan (subset POLA_INJECTION Bab 14).
POLA_INJECTION = ("ignore previous", "abaikan instruksi", "system prompt",
                  "lupakan aturan", "reveal", "kirim kode", "api key")

# Jawaban v3 saat injection terdeteksi — penolakan eksplisit, konsisten.
PENOLAKAN_INJECTION = "Maaf, saya hanya bisa menjawab pertanyaan sesuai dokumen toko."


def _kata(teks):
    """Token sederhana: huruf/angka, buang token pendek (< 3 char)."""
    import re
    return {k for k in re.findall(r"[a-z0-9]+", str(teks).lower()) if len(k) > 2}


def _cari(kueri):
    """Retrieval kata kunci di KB → [(doc, skor)] skor>0, urut menurun.

    Skor = jumlah kata kueri unik yang muncul di (judul + teks) doc,
    dibagi jumlah kata kueri unik — gaya tool_cari_dokumen agentkit (Bab 14).
    """
    kata = _kata(kueri)
    if not kata:
        return []
    hasil = []
    for doc in KB:
        gabung = (doc["judul"] + " " + doc["teks"]).lower()
        cocok = sum(1 for k in kata if k in gabung)
        skor = cocok / len(kata)
        if skor > 0:
            hasil.append((doc, round(skor, 4)))
    hasil.sort(key=lambda h: (-h[1], h[0]["id"]))
    return hasil


def _kutip(konteks, kueri):
    """Generator EKSTRAKTIF: kalimat konteks yang paling banyak memuat kata
    kueri (tanpa mengarang). Konteks kosong → string kosong."""
    kata = _kata(kueri)
    terbaik, skor_terbaik = "", -1
    import re
    for kalimat in re.split(r"(?<=[.!?])\s+", konteks):
        if not kalimat.strip():
            continue
        k = kalimat.lower()
        skor = sum(1 for w in kata if w in k)
        if skor > skor_terbaik:
            terbaik, skor_terbaik = kalimat.strip(), skor
    return terbaik


def _bersih_injection(teks):
    """Netralkan frasa injection (case-insensitive) → '[dihapus]'."""
    import re
    bersih = str(teks)
    for frasa in POLA_INJECTION:
        pola = re.compile(re.escape(frasa), re.IGNORECASE)
        bersih = pola.sub("[dihapus]", bersih)
    return bersih


def _terdeteksi_injection(teks):
    """True bila ada frasa injection di teks (case-insensitive)."""
    rendah = str(teks).lower()
    return any(frasa in rendah for frasa in POLA_INJECTION)


def _jawab_v1(teks):
    asli = str(teks)
    teks_bersih = asli.strip() or asli
    hasil = _cari(teks_bersih)
    skor = hasil[0][1] if hasil else 0.0
    konteks = hasil[0][0]["teks"] if hasil else ""
    jawaban = _kutip(konteks, teks_bersih) if hasil else ""
    if not jawaban:
        # v1 TIDAK jujur: tanpa konteks ia menggema input (hallucination
        # gaya buruk) — disinilah eval menangkapnya.
        jawaban = f"Untuk '{teks_bersih}' silakan hubungi customer service kami."
    return {"jawaban": jawaban, "skor_retrieval": skor, "meta": {"versi": "v1"}}


def _jawab_v2(teks):
    asli = str(teks)
    teks_bersih = asli.strip() or asli
    hasil = _cari(teks_bersih)
    skor = hasil[0][1] if hasil else 0.0
    if skor < AMBAT_SKOR:
        # Abstain gate (Bab 12): skor rendah → jangan mengarang.
        return {"jawaban": FRASA_TIDAK_TAHU, "skor_retrieval": skor,
                "meta": {"versi": "v2", "abstain": True}}
    konteks = hasil[0][0]["teks"]
    jawaban = _kutip(konteks, teks_bersih)
    if not jawaban:
        jawaban = FRASA_TIDAK_TAHU
    return {"jawaban": jawaban, "skor_retrieval": skor, "meta": {"versi": "v2"}}


def _jawab_v3(teks):
    asli = str(teks)
    # Guardrail 1: input kosong/whitespace → abstain (bukan gema).
    if not asli.strip():
        return {"jawaban": FRASA_TIDAK_TAHU, "skor_retrieval": 0.0,
                "meta": {"versi": "v3", "abstain": True}}
    # Guardrail 2: injection terdeteksi → TOLAK eksplisit (jangan dilayani,
    # jangan dinetralkan lalu dijawab — frasa '[dihapus]' bukan untuk user).
    if _terdeteksi_injection(asli):
        return {"jawaban": PENOLAKAN_INJECTION, "skor_retrieval": 0.0,
                "meta": {"versi": "v3", "injection": True}}
    hasil = _cari(asli)
    skor = hasil[0][1] if hasil else 0.0
    if skor < AMBAT_SKOR:
        return {"jawaban": FRASA_TIDAK_TAHU, "skor_retrieval": skor,
                "meta": {"versi": "v3", "abstain": True}}
    konteks = hasil[0][0]["teks"]
    jawaban = _kutip(konteks, asli)
    if not jawaban:
        jawaban = FRASA_TIDAK_TAHU
    return {"jawaban": jawaban, "skor_retrieval": skor, "meta": {"versi": "v3"}}


SISTEM = {"v1": _jawab_v1, "v2": _jawab_v2, "v3": _jawab_v3}


def jalankan_sistem(versi, teks):
    """Nama versi + teks → hasil kontrak standar (versi tak dikenal → error)."""
    fn = SISTEM.get(versi)
    if fn is None:
        return {"jawaban": "", "skor_retrieval": 0.0,
                "meta": {"versi": versi, "error": f"versi tidak dikenal: {versi}"}}
    return fn(teks)
