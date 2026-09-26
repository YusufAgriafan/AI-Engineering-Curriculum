"""Generator ekstraktif deterministik — GIVEN, JANGAN DIUBAH.

"LLM" di sini adalah generator ekstraktif: ia HANYA mengutip kalimat dari
konteks yang diberikan. Tidak ada pengetahuan luar — jadi kalau jawabannya
salah, penyebabnya retrieval atau prompt, bukan "model mengarang".

Heuristik pemilih kalimat (deterministik, sengaja sederhana — supaya terlihat
BAGAIMANA generator bekerja dan KAPAN ia butuh bantuan retrieval):

1. Kalimat yang berakhir "?" dibuang (itu pertanyaan FAQ, bukan jawaban).
2. Kalimat tanpa kata apa pun dari pertanyaan dibuang (tidak nyambung).
3. Skor kalimat = jumlah kata pertanyaan yang muncul + 2 x jumlah ANGKA yang
   muncul (angka penting: pertanyaan "berapa hari" butuh angka).
4. Skor nol / konteks kosong / tidak ada konteks → ABSTAIN:
   "tidak ada di dokumen".

Bagian penting untuk didik: generator ini TIDAK MENGARANG. Model nyata bisa.
Karena itu prompt RAG + abstain gate di sisi kita tetap wajib — itu yang
dilatih di Bagian 6.
"""
import re

from .data import SEP_DETEKSI_TIDAK_ADA, TEMPLATE_PROMPT_RAG
from .embed import tokenisasi

_KATA_BENDUNG = {"berapa", "bagaimana", "apa", "siapa", "kapan", "yang", "untuk",
                 "dengan", "adalah", "tanpa", "dari", "ke", "di", "dan", "atau",
                 "jika", "bila", "bisa", "dapat", "hari"}

# Skor kalimat minimum supaya dianggap menjawab. 1 kata kebetulan (mis. kueri
# "tiket pesawat" vs kalimat "tiket bantuan IT") BUKAN bukti — di situlah
# abstain gate menyelamatkan: lebih baik "tidak ada di dokumen" daripada
# menjawab dengan kebetulan kosakata.
SKOR_MINIMUM = 2


class GeneratorEkstraktif:
    """Generator jawaban dari konteks. API-nya mirip panggilan LLM:

    `generate(pertanyaan, konteks_list)` →
        {"jawaban": str, "sitasi": [n], "abstain": bool, "sumber": [...]}
    """

    def generate(self, pertanyaan, konteks_list):
        kata_tanya = {t for t in tokenisasi(pertanyaan) if t not in _KATA_BENDUNG}
        kandidat = []          # (skor, index_konteks, kalimat)
        for ci, konteks in enumerate(konteks_list):
            for kalimat in self._kalimat(konteks):
                if kalimat.endswith("?"):
                    continue   # pertanyaan FAQ bukan jawaban
                kata_kal = set(tokenisasi(kalimat))
                irisan = kata_tanya & kata_kal
                if not irisan:
                    continue
                angka = 2 if re.search(r"\d", kalimat) else 0
                kandidat.append((len(irisan) + angka, ci, kalimat))
        if not kandidat or max(k[0] for k in kandidat) < SKOR_MINIMUM:
            return {"jawaban": SEP_DETEKSI_TIDAK_ADA, "sitasi": [],
                    "abstain": True, "sumber": []}
        kandidat.sort(key=lambda t: (-t[0], t[1]))
        terbaik = kandidat[0]
        sitasi = sorted({ci for s, ci, _ in kandidat if s == terbaik[0]})
        teks = " ".join(k for s, _, k in kandidat if s == terbaik[0])
        return {"jawaban": teks, "sitasi": sitasi, "abstain": False,
                "sumber": []}   # diisi `jawab()` dari metadata chunk

    def _kalimat(self, teks):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(teks)) if s.strip()]


def bangun_konteks(chunks):
    """Chunk terpilih → teks konteks bernomor + daftar metadata sitasi.

    [1] (sumber, halaman X)
        isi chunk
    """
    bagian = []
    meta_list = []
    for i, c in enumerate(chunks, start=1):
        bagian.append(f"[{i}] ({c['source']}, halaman {c['page']})\n{c['text']}")
        meta_list.append({"n": i, "source": c["source"], "page": c["page"]})
    return "\n\n".join(bagian), meta_list


def prompt_rag(pertanyaan, chunks, template=TEMPLATE_PROMPT_RAG):
    """Prompt lengkap siap kirim ke LLM (di project ini: ke generator)."""
    konteks, meta = bangun_konteks(chunks)
    return template.format(konteks=konteks, pertanyaan=pertanyaan), meta


def jawab(generator, pertanyaan, chunks):
    """Gabungan prompt → generate. Return jawaban + sitasi + prompt (audit)."""
    prompt, meta = prompt_rag(pertanyaan, chunks)
    hasil = generator.generate(pertanyaan, [c["text"] for c in chunks])
    hasil["prompt"] = prompt
    hasil["sumber"] = [meta[ci] for ci in hasil["sitasi"]]
    return hasil
