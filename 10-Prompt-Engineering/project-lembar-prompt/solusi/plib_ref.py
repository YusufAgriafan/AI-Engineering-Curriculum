"""plib_ref — implementasi referensi project Bab 10 (Prompt Engineering).

Untuk cek mandiri SETELAH selesai. Verifier juga memakai modul ini untuk
mem-patch `plib` sebelum menjalankan test suite & notebook.

Semua fungsi murni stdlib (tanpa numpy) — prompt engineering bukan soal numerik,
tapi soal DISIPLIN: template, skema, parsing, guard, dan evaluasi.

Catatan: `mock_llm.py` (GIVEN) dipakai ulang dari package aslinya supaya mock di
`plib` dan di referensi tidak pernah berbeda perilaku.
"""
import json
import math
import random
import re

from plib.data import (BAGIAN, BAGIAN_WAJIB, FIELD_ORDER, GOLDEN,
                       INJECTION_CASES, LEASH_MARKER_TERLARANG, POLA_INJEKSI,
                       SEED, SKEMA_PENUH)
from plib.mock_llm import cari_expected, fitur_prompt, panggil  # noqa: F401

# ============================ SECTIONS ============================

URUTAN_BAGIAN = BAGIAN


def bagian_ditemukan(prompt):
    """Bagian mana saja yang benar-benar ada (urutan kanonik)."""
    return [b for b in URUTAN_BAGIAN if f"[{b}]" in prompt]


def bagian_hilang(prompt):
    """Bagian WAJIB yang belum ada (urutan kanonik)."""
    ada = set(bagian_ditemukan(prompt))
    return [b for b in BAGIAN_WAJIB if b not in ada]


def prompt_lengkap(prompt):
    return not bagian_hilang(prompt)


def build_prompt(*, role, konteks, tugas, format_rules, constraints,
                 pertanyaan, contoh=""):
    """Rakit prompt produksi dari bagian-bagiannya.

    Bagian dengan isi kosong DILEWATI (jadi `bagian_ditemukan` jujur melaporkan
    apa yang ada, bukan apa yang seharusnya ada).
    """
    bagian = [
        ("ROLE", role), ("KONTEKS", konteks), ("TUGAS", tugas),
        ("FORMAT", format_rules), ("CONSTRAINT", constraints),
        ("CONTOH", contoh), ("INPUT", pertanyaan),
    ]
    potongan = []
    for nama, isi in bagian:
        if isi is None or str(isi).strip() == "":
            continue
        potongan.append(f"[{nama}]\n{str(isi).strip()}")
    return "\n\n".join(potongan)


def estimasi_token(teks, rasio=4.0):
    """Perkiraan jumlah token: ceil(len(teks) / rasio). Rasio 4.0 ~ teks Inggris."""
    return int(math.ceil(len(str(teks)) / float(rasio)))


def potong_budget(teks, maks_token, rasio=4.0):
    """Potong teks ke anggaran token tanpa memotong kata di tengah."""
    maks_karakter = int(maks_token * rasio)
    t = str(teks)
    if len(t) <= maks_karakter:
        return t
    return t[:maks_karakter].rsplit(" ", 1)[0]


# ============================ TEMPLATE ============================

PAT_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def placeholder(template):
    """Nama placeholder unik, terurut (kanonik)."""
    return sorted(set(PAT_PLACEHOLDER.findall(template)))


def render(template, **nilai):
    """Isi placeholder {{nama}}. Placeholder tanpa nilai -> ValueError.

    Sengaja BUKAN str.format: prompt penuh `{ }` JSON, dan `.format` akan
    meledak di sana. Single-pass supaya isi variabel tidak pernah di-render ulang.
    """
    kurang = [n for n in placeholder(template) if n not in nilai]
    if kurang:
        raise ValueError(f"placeholder tanpa nilai: {kurang}")
    return PAT_PLACEHOLDER.sub(lambda m: str(nilai[m.group(1)]), template)


def render_aman(template, **nilai):
    """Versi yang tidak melempar: return (teks, daftar placeholder kurang)."""
    kurang = [n for n in placeholder(template) if n not in nilai]
    teks = PAT_PLACEHOLDER.sub(
        lambda m: str(nilai[m.group(1)]) if m.group(1) in nilai else m.group(0),
        template,
    )
    return teks, kurang


def escape_delimiter(teks, tag="dokumen"):
    """Netralkan tag pembuka/penutup milik delimiter (cegah 'breakout' tag)."""
    t = str(teks).replace(f"</{tag}>", "").replace(f"<{tag}>", "")
    return t


# ============================ FEW-SHOT ============================

def format_contoh(teks, expected):
    """Satu pasangan contoh: Input: "..." / Output: {...}."""
    return f'Input: "{teks}"\nOutput: {json.dumps(expected, ensure_ascii=False)}'


def pilih_contoh(contoh, k):
    """Pilih k contoh yang tersebar MERATA di daftar (deterministik)."""
    n = len(contoh)
    if k >= n:
        return list(contoh)
    if k <= 1:
        return [contoh[0]]
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    unik = []
    for i in idx:
        if i not in unik:
            unik.append(i)
    return [contoh[i] for i in unik]


def urutkan_contoh(contoh):
    """Contoh paling sedikit-terisi dulu, paling kaya TERAKHIR (recency bias).

    Model cenderung meniru contoh terakhir — jadi taruh pola tersulit di sana.
    """
    return sorted(
        contoh,
        key=lambda c: sum(1 for v in c["expected"].values() if v is not None),
    )


def bangun_fewshot(contoh, k):
    """Blok [CONTOH] siap tempel: pilih k, urutkan, format."""
    terpilih = urutkan_contoh(pilih_contoh(contoh, k))
    return "\n\n".join(format_contoh(c["teks"], c["expected"]) for c in terpilih)


# ============================ SCHEMA ============================

TIPE_VALID = ("string", "integer", "number", "boolean", "null", "object", "array")


def _cek_tipe(nilai, tipe):
    if tipe == "string":
        return isinstance(nilai, str)
    if tipe == "integer":
        return isinstance(nilai, int) and not isinstance(nilai, bool)
    if tipe == "number":
        return isinstance(nilai, (int, float)) and not isinstance(nilai, bool)
    if tipe == "boolean":
        return isinstance(nilai, bool)
    if tipe == "null":
        return nilai is None
    if tipe == "object":
        return isinstance(nilai, dict)
    if tipe == "array":
        return isinstance(nilai, list)
    raise ValueError(f"tipe skema tidak dikenal: {tipe}")


def validate(data, skema=None):
    """Validasi JSON-schema mini. Return DAFTAR error (kosong = valid).

    Format error (stabil, dipakai tests/): "missing:<field>", "type:<field>",
    "minimum:<field>", "maximum:<field>", "enum:<field>", "length:<field>",
    "extra:<field>".
    """
    skema = SKEMA_PENUH if skema is None else skema
    errors = []
    if not isinstance(data, dict):
        return ["type:<root>"]
    props = skema.get("properties", {})
    for nama in skema.get("required", []):
        if nama not in data:
            errors.append(f"missing:{nama}")
    for nama, aturan in props.items():
        if nama not in data:
            continue
        nilai = data[nama]
        tipe = aturan.get("type")
        if tipe is not None:
            daftar = tipe if isinstance(tipe, list) else [tipe]
            if not any(_cek_tipe(nilai, t) for t in daftar):
                errors.append(f"type:{nama}")
                continue
        if "enum" in aturan and nilai not in aturan["enum"]:
            errors.append(f"enum:{nama}")
        if nilai is None:
            continue
        if "minimum" in aturan and nilai < aturan["minimum"]:
            errors.append(f"minimum:{nama}")
        if "maximum" in aturan and nilai > aturan["maximum"]:
            errors.append(f"maximum:{nama}")
        if isinstance(nilai, str):
            if "minLength" in aturan and len(nilai) < aturan["minLength"]:
                errors.append(f"length:{nama}")
            if "maxLength" in aturan and len(nilai) > aturan["maxLength"]:
                errors.append(f"length:{nama}")
    if skema.get("additionalProperties") is False:
        for nama in data:
            if nama not in props:
                errors.append(f"extra:{nama}")
    return errors


def valid(data, skema=None):
    """True bila tidak ada error."""
    return not validate(data, skema)


# ============================ PARSE ============================

def perbaiki_json(teks):
    """Betulkan pelanggaran JSON yang paling umum di output LLM.

    Ditangani: komentar `//`, kutip tunggal, True/False/None ala Python,
    dan koma menggantung (trailing comma). Sengaja naif — lihat catatan di README.
    """
    t = str(teks).strip()
    t = re.sub(r"//[^\n]*", "", t)
    t = t.replace("True", "true").replace("False", "false").replace("None", "null")
    t = re.sub(r"'([^']*)'", r'"\1"', t)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return json.loads(t)


def ekstrak_json(teks):
    """Ambil objek JSON PERTAMA dari teks berisik (prosa, code fence, dll).

    Pemindai brace-matching yang sadar string & escape — jadi `{` di dalam
    string tidak menghancurkan penghitungan kedalaman.
    """
    t = str(teks)
    pos = t.find("{")
    while pos != -1:
        depth, dalam_string, escape = 0, False, False
        for j in range(pos, len(t)):
            c = t[j]
            if dalam_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    dalam_string = False
                continue
            if c == '"':
                dalam_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    kandidat = t[pos:j + 1]
                    try:
                        return json.loads(kandidat)
                    except json.JSONDecodeError:
                        try:
                            return perbaiki_json(kandidat)
                        except Exception:
                            break
        pos = t.find("{", pos + 1)
    raise ValueError("tidak ada objek JSON valid di output")


def parse_output(teks, skema=None):
    """Output mentah model -> dict tervalidasi. Lempar ValueError bila gagal."""
    data = ekstrak_json(teks)
    errors = validate(data, skema)
    if errors:
        raise ValueError(f"output tidak sesuai skema: {errors}")
    return data


def naive_json_ok(teks):
    """Apakah `json.loads` MENTAH berhasil (tanpa pembersihan apa pun)?

    Ini metrik penting: output yang butuh pembersihan = output yang rapuh.
    """
    try:
        json.loads(str(teks).strip())
        return True
    except Exception:
        return False


# ============================ GUARD ============================

def deteksi_injection(teks):
    """Nama-nama pola injeksi yang cocok, terurut & unik."""
    t = str(teks)
    return sorted({nama for nama, pola in POLA_INJEKSI if re.search(pola, t, re.I | re.M)})


def bungkus_data(teks, tag="dokumen"):
    """Bungkus data tak tepercaya dengan delimiter yang sudah di-escape."""
    return f"<{tag}>\n{escape_delimiter(teks, tag)}\n</{tag}>"


def langgar_leash(output, marker=None):
    """True bila output keluar dari tugas (mengandung marker terlarang)."""
    daftar = LEASH_MARKER_TERLARANG if marker is None else [marker]
    out = str(output).lower()
    return any(str(m).lower() in out for m in daftar)


# ============================ EVALUATE ============================

METRIK = ["naive_json_rate", "parse_ok_rate", "exact_rate", "field_accuracy", "injeksi_rate"]


def jalankan_eval(buat_prompt, kasus=None, temperature=0.0, seed=SEED):
    """Jalankan satu konfigurasi prompt terhadap seluruh dataset.

    `buat_prompt(teks) -> prompt` — pemanggil yang menentukan bagaimana prompt
    dirakit (render template, bungkus data, pilih few-shot, dll).

    Mengembalikan dict metrik + rincian per kasus (untuk audit, bukan cuma angka).
    """
    kasus = GOLDEN if kasus is None else kasus
    rng = random.Random(seed)
    rincian = []
    n = len(kasus)
    naive_json = parse_ok = exact = 0
    field_benar = field_total = 0

    for item in kasus:
        teks, expected = item["teks"], item["expected"]
        prompt = buat_prompt(teks)
        out = panggil(prompt, teks, rng=rng, temperature=temperature)

        nj = naive_json_ok(out)
        naive_json += int(nj)

        data = None
        po = True
        try:
            data = parse_output(out)
        except Exception:
            po = False
        parse_ok += int(po)

        benar = 0
        if po:
            benar = sum(1 for k in FIELD_ORDER if data.get(k, "<kosong>") == expected[k])
        field_benar += benar
        field_total += len(FIELD_ORDER)

        ex = bool(po and data == expected)
        exact += int(ex)

        rincian.append({
            "teks": teks, "prompt": prompt, "output": out, "expected": expected,
            "parsed": data, "naive_json": nj, "parse_ok": po, "exact": ex,
            "field_benar": benar,
        })

    injeksi_diblokir = 0
    rincian_injeksi = []
    for item in INJECTION_CASES:
        prompt = buat_prompt(item["teks"])
        out = panggil(prompt, item["teks"], rng=rng, temperature=temperature)
        diblokir = item["marker"].lower() not in str(out).lower()
        injeksi_diblokir += int(diblokir)
        rincian_injeksi.append({
            "teks": item["teks"], "marker": item["marker"],
            "output": out, "diblokir": diblokir,
        })

    return {
        "n": n,
        "naive_json": naive_json,
        "naive_json_rate": naive_json / n,
        "parse_ok": parse_ok,
        "parse_ok_rate": parse_ok / n,
        "exact": exact,
        "exact_rate": exact / n,
        "field_benar": field_benar,
        "field_total": field_total,
        "field_accuracy": field_benar / field_total,
        "injeksi_diblokir": injeksi_diblokir,
        "injeksi_total": len(INJECTION_CASES),
        "injeksi_rate": injeksi_diblokir / len(INJECTION_CASES),
        "temperature": temperature,
        "rincian": rincian,
        "rincian_injeksi": rincian_injeksi,
    }


def bandingkan(hasil_a, hasil_b):
    """Bandingkan dua hasil eval: delta per metrik + pemenang."""
    delta = {k: round(hasil_b[k] - hasil_a[k], 6) for k in METRIK}
    skor = lambda h: (h["field_accuracy"] + h["exact_rate"]
                      + h["injeksi_rate"] + h["naive_json_rate"])
    sa, sb = skor(hasil_a), skor(hasil_b)
    if sb > sa:
        pemenang = "b"
    elif sa > sb:
        pemenang = "a"
    else:
        pemenang = "seri"
    return {"delta": delta, "skor_a": round(sa, 6), "skor_b": round(sb, 6),
            "pemenang": pemenang}
