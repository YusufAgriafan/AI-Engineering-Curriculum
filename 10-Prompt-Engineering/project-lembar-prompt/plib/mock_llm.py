"""Mock LLM deterministik — GIVEN, JANGAN DIUBAH.

Kenapa ada mock? Prompt engineering biasanya butuh API berbayar dan hasilnya
acak, sehingga sulit dipakai sebagai latihan ber-test. Di sini "model" diganti
simulator yang **kepatuhannya ditentukan kualitas prompt-mu**. Jadi kamu bisa
mengukur prompt tanpa membakar token, dan angkanya reproducible (seed 10).

Aturan mainnya TERBUKA (bukan kotak hitam):

  1. Tanpa delimiter + leash, instruksi di dalam data dituruti -> injeksi sukses.
  2. Tanpa field berskema (`has_schema`), model menjawab prosa bebas.
  3. Tanpa kebijakan null, nilai kosong DIKARANG (halusinasi terkalibrasi).
  4. Tanpa contoh yang menunjukkan kasus TEPI (`"extracted": false`), model
     memakai default aman `extracted=true, confidence=0.5` — contoh yang tidak
     menambah informasi tidak menambah akurasi.
  5. Tanpa instruksi "hanya JSON", output dibungkus basa-basi + code fence.
  6. temperature > 0: dengan probabilitas T, output diawali basa-basi.

Konsekuensinya: satu-satunya variabel yang mengubah skor adalah prompt-mu.
"""
import json
import random
import re

from .data import FIELD_ORDER, GOLDEN, HALLUCINASI, POLA_INJEKSI, SEED

PAT_STRICT = re.compile(
    r"(hanya|only|cukup)[^\n]{0,40}json"
    r"|json[^\n]{0,30}(saja|only)"
    r"|tanpa (teks|penjelasan|kata)[^\n]{0,20}lain"
    r"|no other text",
    re.I,
)
PAT_NULL_ADA = re.compile(r"null", re.I)
PAT_NULL_KONTEKS = re.compile(
    r"(jika|bila|apabila)[^\n]{0,60}(tidak ada|tidak disebutkan|tidak ditemukan|unknown)"
    r"|(tidak ada|tidak disebutkan|tidak ditemukan)[^\n]{0,40}null"
    r"|jangan mengarang"
    r"|(isi|gunakan|beri|pakai)\s+(nilai\s+)?null"
    r"|\bunknown\b",
    re.I,
)
PAT_CONTOH_NEGATIF = re.compile(r'"extracted"\s*:\s*false', re.I)
PAT_DELIMITER = re.compile(r"<(dokumen|konteks|data|input|chat)>[\s\S]*</\1>", re.I)
PAT_LEASH = re.compile(
    r"(abaikan|jangan (ikuti|turut|patuhi)|jangan menuruti)[^\n]{0,80}(instruksi|perintah|arahan)",
    re.I,
)
def _nama_field_ada(prompt, nama):
    """Apakah `nama` dipakai sebagai kunci/skema di prompt ini?"""
    return bool(re.search(r'["\'\-\s]{}["\']?\s*:'.format(re.escape(nama)), prompt))


def deteksi_injeksi(teks):
    """Nama pola injeksi yang cocok (dipakai internal mock; versi publik + tes
    ada di `plib/guard.py` yang kamu isi sendiri).
    """
    t = str(teks)
    return sorted({nama for nama, pola in POLA_INJEKSI if re.search(pola, t, re.I | re.M)})


def fitur_prompt(prompt):
    """Baca fitur kualitas dari prompt yang diberikan. (GIVEN)"""
    low = str(prompt).lower()
    fields = [n for n in FIELD_ORDER if _nama_field_ada(prompt, n)]
    return {
        "fields": fields,
        "has_schema": len(fields) >= 4,
        "strict_json": bool(PAT_STRICT.search(prompt)),
        "null_policy": bool(PAT_NULL_ADA.search(prompt)) and bool(PAT_NULL_KONTEKS.search(prompt)),
        "has_delimiter": bool(PAT_DELIMITER.search(prompt)),
        "has_leash": bool(PAT_LEASH.search(prompt)),
        "has_fewshot": ("contoh" in low) and ("input:" in low) and ("output:" in low),
        "contoh_negatif": bool(PAT_CONTOH_NEGATIF.search(prompt)),
    }


def cari_expected(teks):
    """Isi tabel GOLDEN untuk teks ini (dict kosong bila tidak dikenal). (GIVEN)"""
    for kasus in GOLDEN:
        if kasus["teks"] == teks:
            return dict(kasus["expected"])
    return {}


def _format_output(out, fitur):
    teks_json = json.dumps(out, ensure_ascii=False)
    if fitur["strict_json"]:
        return teks_json
    return f"Tentu! Berikut hasil ekstraksinya:\n```json\n{teks_json}\n```\nSemoga membantu!"


def panggil(prompt, teks, rng=None, temperature=0.0):
    """Output mock model (string). (GIVEN) — lihat aturan 1-6 di docstring modul."""
    rng = random.Random(SEED) if rng is None else rng
    fitur = fitur_prompt(prompt)

    if deteksi_injeksi(teks) and not (fitur["has_delimiter"] and fitur["has_leash"]):
        return "Siap, saya turuti. " + str(teks)

    expected = cari_expected(teks)

    if not fitur["has_schema"]:
        topik = expected.get("product_name") or "hal ini"
        return (f"Sepertinya pelanggan membahas {topik}. "
                "Silakan hubungi tim kami untuk detailnya.")

    out = {}
    for nama in fitur["fields"]:
        nilai = expected.get(nama)
        if nilai is None and not fitur["null_policy"]:
            nilai = HALLUCINASI[nama]
        out[nama] = nilai

    if not fitur["contoh_negatif"]:
        if "extracted" in out:
            out["extracted"] = True
        if "confidence" in out:
            out["confidence"] = 0.5

    if temperature > 0 and rng.random() < temperature:
        return "Baik, saya bantu ya! " + _format_output(out, fitur)
    return _format_output(out, fitur)
