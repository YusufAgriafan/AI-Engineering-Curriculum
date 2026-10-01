"""Implementasi REFERENSI agentkit — untuk cek mandiri & verifier.

JANGAN dibaca sebelum mencoba; ini kunci jawaban seluruh modul TODO.
Bentuk & kontrak tiap fungsi = yang diuji tests/.
"""
import copy
import json
import re

from agentkit.data import (AKSI_BERISIKO, AMBAT_SKOR_RAG, DOKUMEN, HARGA,
                           MAKS_ITER, PESANAN, TOOLS_SCHEMA)
from agentkit.guardrail import POLA_INJECTION
from agentkit.nlu import klasifikasi_intent, rencana_untuk, token_estimasi


# ============================ 1. tools ============================

def validasi_skema(nama_tool, args):
    """Validasi args terhadap skema. Return (ok, pesan_error).

    Subset JSON Schema yang dipakai: required, type 'string', min_panjang,
    pola (regex, dicari via re.search). Args non-dict → tidak valid.
    """
    skema = TOOLS_SCHEMA.get(nama_tool)
    if skema is None:
        return False, f"tool tidak dikenal: {nama_tool}"
    if not isinstance(args, dict):
        return False, "argumen harus berupa object/dict"
    spec = skema["argumen"]
    for nama in spec.get("required", []):
        if nama not in args:
            return False, f"argumen wajib hilang: {nama}"
    for nama, aturan in spec.get("properties", {}).items():
        if nama not in args:
            continue
        nilai = args[nama]
        if aturan.get("type") == "string" and not isinstance(nilai, str):
            return False, f"argumen {nama} harus string"
        if "min_panjang" in aturan and len(nilai) < aturan["min_panjang"]:
            return False, f"argumen {nama} minimal {aturan['min_panjang']} karakter"
        if "pola" in aturan and not re.search(aturan["pola"], nilai):
            return False, f"argumen {nama} tidak cocok pola {aturan['pola']}"
    return True, ""


def tool_cari_dokumen(kueri):
    """Pencarian kata kunci di KB: skor = kata unik yang cocok / total kata unik."""
    kata = {k for k in re.findall(r"[a-z0-9]+", str(kueri).lower()) if len(k) > 2}
    if not kata:
        return []
    hasil = []
    for doc in DOKUMEN["kb"]:
        gabung = (doc["judul"] + " " + doc["teks"]).lower()
        cocok = sum(1 for k in kata if k in gabung)
        skor = cocok / len(kata)
        if skor > 0:
            hasil.append({"id": doc["id"], "judul": doc["judul"],
                          "teks": doc["teks"], "skor": round(skor, 4)})
    hasil.sort(key=lambda h: (-h["skor"], h["id"]))
    return hasil


def tool_cek_pesanan(invoice_id):
    """INV-### → salinan pesanan TANPA kunci 'user' (PII tidak boleh ke LLM)."""
    pesanan = PESANAN.get(str(invoice_id).upper())
    if pesanan is None:
        return {"error": "pesanan tidak ditemukan"}
    keluar = dict(pesanan)
    keluar.pop("user", None)
    return keluar


def tool_refund(invoice_id, alasan):
    """Refund tidak dieksekusi: selalu menghasilkan rencana konfirmasi manusia."""
    return {"butuh_konfirmasi": True, "invoice_id": invoice_id,
            "alasan": alasan,
            "pesan": "Refund butuh konfirmasi manusia sebelum dieksekusi."}


REGISTRY = {
    "cari_dokumen": tool_cari_dokumen,
    "cek_pesanan": tool_cek_pesanan,
    "refund": tool_refund,
}


def eksekusi_tool(nama_tool, args):
    """Dispatch registry: validasi dulu, lalu panggil. Error → {'error': ...}."""
    ok, pesan = validasi_skema(nama_tool, args)
    if not ok:
        return {"error": pesan}
    fn = REGISTRY.get(nama_tool)
    if fn is None:
        return {"error": f"tool tidak dikenal: {nama_tool}"}
    try:
        return fn(**args)
    except Exception as e:  # error tool = observasi, bukan crash
        return {"error": f"{type(e).__name__}: {e}"}


# ============================ 2. loop_react ============================

def format_langkah(thought, action=None, observation=None):
    """Satu langkah → blok teks ReAct terkunci."""
    baris = [f"Thought: {thought}"]
    if action is not None:
        nama, args = action
        baris.append(f"Action: {nama}({json.dumps(args, ensure_ascii=False)})")
    if observation is not None:
        baris.append(f"Observation: {observation}")
    return "\n".join(baris)


def format_trace(langkah_list):
    """List langkah → trace penuh, blok dipisah baris kosong."""
    return "\n\n".join(langkah_list)


def jalankan_langkah(langkah):
    """Satu langkah rencana → (observasi, nama_tool|None)."""
    nama = langkah.get("tool")
    if nama is None:
        return None, None
    return eksekusi_tool(nama, langkah.get("args") or {}), nama


def jalankan_rencana(rencana, maks_iter=MAKS_ITER):
    """Eksekusi rencana + guardrail iterasi. Return dict hasil final.

    Rencana habis tanpa final ATAU mencapai maks_iter → escallate dengan
    berhenti_dini=True (loop guardrail, bukan loop tak berujung).
    """
    langkah_tereksekusi = []
    for n, langkah in enumerate(rencana[:maks_iter], start=1):
        observasi, dipakai = jalankan_langkah(langkah)
        catatan = {"thought": langkah["thought"], "tool": dipakai,
                   "args": langkah.get("args"), "observasi": observasi,
                   "teks": format_langkah(langkah["thought"],
                                          (dipakai, langkah.get("args")) if dipakai else None,
                                          observasi)}
        langkah_tereksekusi.append(catatan)
        if langkah.get("tool") is None:
            return {"final": dict(langkah.get("args") or {}),
                    "langkah": langkah_tereksekusi,
                    "berhenti_dini": False,
                    "jumlah_langkah": len(langkah_tereksekusi)}
    escallate = {"aksi_final": "jawab",
                 "jawaban": "Saya belum bisa menyelesaikan permintaan ini. "
                            "Mohon hubungi agen manusia."}
    return {"final": escallate, "langkah": langkah_tereksekusi,
            "berhenti_dini": True, "jumlah_langkah": len(langkah_tereksekusi)}


# ============================ 3. guardrail ============================

def butuh_konfirmasi(nama_aksi, aksi_berisiko=None):
    """Aksi berisiko → True (perbandingan persis, bukan substring)."""
    daftar = AKSI_BERISIKO if aksi_berisiko is None else aksi_berisiko
    return nama_aksi in daftar


def sanitasi(teks):
    """Netralkan perintah injection + baris baru; return salinan bersih."""
    bersih = str(teks)
    for frasa in POLA_INJECTION:
        pola = re.compile(re.escape(frasa), re.IGNORECASE)
        bersih = pola.sub("[dihapus]", bersih)
    bersih = bersih.replace("\n", " ")
    return bersih.strip()


def amankan_observasi(observasi):
    """Sanitasi rekursif: string di-sanitasi, dict/list di-salin & dibersihkan."""
    if isinstance(observasi, str):
        return sanitasi(observasi)
    if isinstance(observasi, dict):
        return {k: amankan_observasi(v) for k, v in observasi.items()}
    if isinstance(observasi, list):
        return [amankan_observasi(v) for v in observasi]
    return observasi


# ============================ 4. memory ============================

class MemoriSesi:
    """Riwayat giliran + cache retrieval per sesi (persisten antar panggilan)."""

    def __init__(self, maks_giliran=4):
        self.maks_giliran = maks_giliran
        self.riwayat = []
        self.cache = {}
        self.cache_hit = 0
        self.cache_miss = 0

    def tambah(self, peran, teks):
        giliran = {"peran": peran, "teks": teks, "token": token_estimasi(teks)}
        self.riwayat.append(giliran)
        return giliran

    def konteks(self):
        """max_giliran TERAKHIR, urutan ASLI (bukan dibalik)."""
        return list(self.riwayat[-self.maks_giliran:])

    def ringkas(self):
        return "\n".join(f"{g['peran']}: {g['teks']}" for g in self.riwayat)

    def cache_get(self, kueri):
        nilai = self.cache.get(str(kueri).strip().lower())
        if nilai is None:
            self.cache_miss += 1
        else:
            self.cache_hit += 1
        return nilai

    def cache_put(self, kueri, nilai):
        self.cache[str(kueri).strip().lower()] = nilai


def hitung_biaya(pesan, model="api_mini"):
    """Total token riwayat → USD sesuai tarif HARGA (in+out digabung sederhana)."""
    tarif = HARGA[model]
    total_token = sum(g["token"] for g in pesan)
    return total_token / 1e6 * tarif["input"]


# ============================ 5. agent ============================

def jawab(teks, maks_iter=MAKS_ITER, skor_rag_min=AMBAT_SKOR_RAG):
    """Pipeline end-to-end: intent → rencana → loop ReAct → + intent."""
    intent = klasifikasi_intent(teks)
    hasil = jalankan_rencana(rencana_untuk(intent, teks), maks_iter)
    hasil["intent"] = intent
    return hasil


def planner_bertahap(intent, teks):
    """Planner 'kalau-perlu': langkah tambah hanya bila prasyarat ada."""
    import re as _re
    punya_id = _re.search(r"INV-\d{3}", str(teks).upper()) is not None
    if intent == "status_pesanan" and not punya_id:
        return [{"thought": "User tanya status tapi tidak menyebut ID invoice.",
                 "tool": None, "args": {"aksi_final": "tanya_balik",
                                        "jawaban": "Boleh sebutkan nomor invoice Anda (format INV-XXX)?"}}]
    if intent == "status_pesanan":
        inv = _re.search(r"INV-\d{3}", str(teks).upper()).group(0)
        return [{"thought": f"Saya perlu cek status pesanan {inv} di sistem.",
                 "tool": "cek_pesanan", "args": {"invoice_id": inv}},
                {"thought": "Status sudah didapat — jawab user.",
                 "tool": None, "args": {"aksi_final": "jawab"}}]
    if intent == "refund" and punya_id:
        inv = _re.search(r"INV-\d{3}", str(teks).upper()).group(0)
        return [{"thought": f"Cek dulu status {inv} supaya refund kontekstual.",
                 "tool": "cek_pesanan", "args": {"invoice_id": inv}},
                {"thought": "Refund adalah aksi berisiko — minta konfirmasi manusia.",
                 "tool": None, "args": {"aksi_final": "minta_konfirmasi"}}]
    if intent == "refund":
        return [{"thought": "Refund butuh nomor invoice yang valid.",
                 "tool": None, "args": {"aksi_final": "tanya_balik",
                                        "jawaban": "Untuk refund, sebutkan nomor invoice (format INV-XXX) dan alasannya."}}]
    if intent == "kebijakan":
        return [{"thought": "Saya carikan kebijakan relevan di KB.",
                 "tool": "cari_dokumen", "args": {"kueri": str(teks)}},
                {"thought": "Kebijakan ditemukan — rangkum untuk user.",
                 "tool": None, "args": {"aksi_final": "jawab"}}]
    return [{"thought": "Saya tidak yakin maksud user — lebih jujur bertanya balik.",
             "tool": None, "args": {"aksi_final": "tanya_balik",
                                    "jawaban": "Maaf, saya belum paham. Bisa dijelaskan terkait status pesanan, kebijakan, atau refund?"}}]


def evaluasi_trace(hasil):
    """Skor kualitas hasil agent: 1.0 bersih | 0.5 tanya_balik | 0.0 error/dini."""
    langkah_list = hasil.get("langkah", []) if isinstance(hasil, dict) else []
    temuan = []
    ada_error = False
    ada_tanya = False
    for l in langkah_list:
        final = l.get("args") or {}
        if isinstance(final, dict) and final.get("aksi_final") == "tanya_balik":
            ada_tanya = True
        if isinstance(l.get("observasi"), dict) and "error" in l["observasi"]:
            ada_error = True
            temuan.append(f"tool error: {l['observasi']['error']}")
    if ada_tanya:
        temuan.append("ada klarifikasi ke user (tanya_balik)")
    if isinstance(hasil, dict) and hasil.get("berhenti_dini"):
        temuan.append("berhenti dini (guardrail iterasi)")
        ada_error = True
    if ada_error:
        skor = 0.0
    elif ada_tanya:
        skor = 0.5
    else:
        skor = 1.0
    if not temuan:
        temuan = ["bersih"]
    return {"skor": skor, "temuan": temuan}


def verifikasi_tools():
    """Health check: semua tool harus mengembalikan hasil tanpa 'error'."""
    coba = [("cari_dokumen", {"kueri": "pengembalian barang"}),
            ("cek_pesanan", {"invoice_id": "INV-001"}),
            ("refund", {"invoice_id": "INV-001", "alasan": "barang rusak"})]
    return all("error" not in eksekusi_tool(n, a) for n, a in coba)
