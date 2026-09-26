"""Verify 01_lab_prompt_engineering.ipynb: exec every cell; TODO cells get reference impls.

Usage (dari folder ini):
    python _verify_lab.py
"""
import contextlib  # noqa: E402
import io  # noqa: E402
import json
import sys
import time  # noqa: E402
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

NB = Path(__file__).resolve().parent / "01_lab_prompt_engineering.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "bagian_ditemukan": (
        "def bagian_ditemukan(prompt):\n"
        "    return [b for b in BAGIAN if f'[{b}]' in prompt]\n"
    ),
    "bagian_hilang": (
        "def bagian_hilang(prompt):\n"
        "    ada = set(bagian_ditemukan(prompt))\n"
        "    return [b for b in BAGIAN_WAJIB if b not in ada]\n"
    ),
    "placeholder": (
        "def placeholder(template):\n"
        "    return sorted(set(PAT.findall(template)))\n"
    ),
    "render": (
        "def render(template, **nilai):\n"
        "    kurang = [n for n in placeholder(template) if n not in nilai]\n"
        "    if kurang:\n"
        "        raise ValueError(f'placeholder tanpa nilai: {kurang}')\n"
        "    return PAT.sub(lambda m: str(nilai[m.group(1)]), template)\n"
    ),
    "escape_delimiter": (
        "def escape_delimiter(teks, tag='dokumen'):\n"
        "    return str(teks).replace(f'</{tag}>', '').replace(f'<{tag}>', '')\n"
    ),
    "format_contoh": (
        "def format_contoh(teks, expected):\n"
        "    return f'Input: \"{teks}\"\\nOutput: ' + json.dumps(expected, ensure_ascii=False)\n"
    ),
    "pilih_contoh": (
        "def pilih_contoh(contoh, k):\n"
        "    n = len(contoh)\n"
        "    if k >= n:\n"
        "        return list(contoh)\n"
        "    if k <= 1:\n"
        "        return [contoh[0]]\n"
        "    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]\n"
        "    unik = []\n"
        "    for i in idx:\n"
        "        if i not in unik:\n"
        "            unik.append(i)\n"
        "    return [contoh[i] for i in unik]\n"
    ),
    "urutkan_contoh": (
        "def urutkan_contoh(contoh):\n"
        "    return sorted(contoh, key=lambda c: sum(1 for v in c['expected'].values() if v is not None))\n"
    ),
    "fitur_prompt": (
        "def fitur_prompt(prompt):\n"
        "    low = str(prompt).lower()\n"
        "    fields = [n for n in FIELD if _field_ada(prompt, n)]\n"
        "    return {\n"
        "        'fields': fields,\n"
        "        'has_schema': len(fields) >= 4,\n"
        "        'strict_json': bool(PAT_STRICT.search(prompt)),\n"
        "        'null_policy': bool(PAT_NULL_ADA.search(prompt)) and bool(PAT_NULL_KONTEKS.search(prompt)),\n"
        "        'has_delimiter': bool(PAT_DELIM.search(prompt)),\n"
        "        'has_leash': bool(PAT_LEASH.search(prompt)),\n"
        "        'has_fewshot': ('contoh' in low) and ('input:' in low) and ('output:' in low),\n"
        "        'contoh_negatif': bool(PAT_NEGATIF.search(prompt)),\n"
        "    }\n"
    ),
    "deteksi_injection": (
        "def deteksi_injection(teks):\n"
        "    return sorted({nama for nama, pola in POLA if re.search(pola, str(teks), re.I | re.M)})\n"
    ),
    "bungkus_data": (
        "def bungkus_data(teks, tag='dokumen'):\n"
        "    return f'<{tag}>\\n' + escape_delimiter(teks, tag) + f'\\n</{tag}>'\n"
    ),
    "validate": (
        "def validate(data):\n"
        "    errors = []\n"
        "    if not isinstance(data, dict):\n"
        "        return ['type:<root>']\n"
        "    props = SKEMA['properties']\n"
        "    for nama in SKEMA['required']:\n"
        "        if nama not in data:\n"
        "            errors.append(f'missing:{nama}')\n"
        "    for nama, aturan in props.items():\n"
        "        if nama not in data:\n"
        "            continue\n"
        "        nilai = data[nama]\n"
        "        tipe = aturan['type']\n"
        "        daftar = tipe if isinstance(tipe, list) else [tipe]\n"
        "        if not any(_tipe_ok(nilai, t) for t in daftar):\n"
        "            errors.append(f'type:{nama}')\n"
        "            continue\n"
        "        if nilai is None:\n"
        "            continue\n"
        "        if 'minimum' in aturan and nilai < aturan['minimum']:\n"
        "            errors.append(f'minimum:{nama}')\n"
        "        if 'maximum' in aturan and nilai > aturan['maximum']:\n"
        "            errors.append(f'maximum:{nama}')\n"
        "    return errors\n"
    ),
    "perbaiki_json": (
        "def perbaiki_json(teks):\n"
        "    t = str(teks).strip()\n"
        "    t = re.sub(r'//[^\\n]*', '', t)\n"
        "    t = t.replace('True', 'true').replace('False', 'false').replace('None', 'null')\n"
        "    t = re.sub(r\"'([^']*)'\", r'\"\\1\"', t)\n"
        "    t = re.sub(r',\\s*([}\\]])', r'\\1', t)\n"
        "    return json.loads(t)\n"
    ),
    "ekstrak_json": (
        "def ekstrak_json(teks):\n"
        "    t = str(teks)\n"
        "    pos = t.find('{')\n"
        "    while pos != -1:\n"
        "        depth, dalam_string, escape = 0, False, False\n"
        "        for j in range(pos, len(t)):\n"
        "            c = t[j]\n"
        "            if dalam_string:\n"
        "                if escape:\n"
        "                    escape = False\n"
        "                elif c == '\\\\':\n"
        "                    escape = True\n"
        "                elif c == '\"':\n"
        "                    dalam_string = False\n"
        "                continue\n"
        "            if c == '\"':\n"
        "                dalam_string = True\n"
        "            elif c == '{':\n"
        "                depth += 1\n"
        "            elif c == '}':\n"
        "                depth -= 1\n"
        "                if depth == 0:\n"
        "                    kandidat = t[pos:j + 1]\n"
        "                    try:\n"
        "                        return json.loads(kandidat)\n"
        "                    except json.JSONDecodeError:\n"
        "                        try:\n"
        "                            return perbaiki_json(kandidat)\n"
        "                        except Exception:\n"
        "                            break\n"
        "        pos = t.find('{', pos + 1)\n"
        "    raise ValueError('tidak ada objek JSON valid di output')\n"
    ),
    "jalankan_eval": (
        "def jalankan_eval(buat_prompt, temperature=0.0, seed=10):\n"
        "    rng = random.Random(seed)\n"
        "    n = len(LAB_KASUS)\n"
        "    naive_json = parse_ok = exact = field_benar = 0\n"
        "    rincian = []\n"
        "    for kasus in LAB_KASUS:\n"
        "        teks, expected = kasus['teks'], kasus['expected']\n"
        "        out = panggil(buat_prompt(teks), teks, rng=rng, temperature=temperature)\n"
        "        try:\n"
        "            json.loads(out.strip())\n"
        "            nj = True\n"
        "        except Exception:\n"
        "            nj = False\n"
        "        naive_json += int(nj)\n"
        "        data, po = None, True\n"
        "        try:\n"
        "            data = ekstrak_json(out)\n"
        "            if validate(data):\n"
        "                po = False\n"
        "        except Exception:\n"
        "            po = False\n"
        "        parse_ok += int(po)\n"
        "        benar = 0\n"
        "        if po:\n"
        "            benar = sum(1 for k in FIELD if data.get(k, '<kosong>') == expected[k])\n"
        "        field_benar += benar\n"
        "        ex = bool(po and data == expected)\n"
        "        exact += int(ex)\n"
        "        rincian.append({'teks': teks, 'output': out, 'expected': expected,\n"
        "                        'parsed': data if po else None, 'naive_json': nj,\n"
        "                        'parse_ok': po, 'exact': ex, 'field_benar': benar})\n"
        "    diblokir = 0\n"
        "    for kasus in LAB_INJEKSI:\n"
        "        out = panggil(buat_prompt(kasus['teks']), kasus['teks'], rng=rng, temperature=temperature)\n"
        "        diblokir += int(kasus['marker'].lower() not in str(out).lower())\n"
        "    return {\n"
        "        'n': n, 'naive_json': naive_json, 'naive_json_rate': naive_json / n,\n"
        "        'parse_ok': parse_ok, 'parse_ok_rate': parse_ok / n,\n"
        "        'exact': exact, 'exact_rate': exact / n,\n"
        "        'field_benar': field_benar, 'field_total': n * len(FIELD),\n"
        "        'field_accuracy': field_benar / (n * len(FIELD)),\n"
        "        'injeksi_diblokir': diblokir, 'injeksi_total': len(LAB_INJEKSI),\n"
        "        'injeksi_rate': diblokir / len(LAB_INJEKSI),\n"
        "        'rincian': rincian,\n"
        "    }\n"
    ),
}

ctx = {"__name__": "__main__", "__builtins__": __builtins__}

t0 = time.time()
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src:
        names = [n for n in REF if f"def {n}(" in src]
        if not names:
            print(f"[cell {i:>2}] SKIPPED (TODO tanpa fungsi dikenal)")
            continue
        for n in names:
            exec(REF[n], ctx)
        print(f"[cell {i:>2}] injected: {', '.join(names)}")
        continue
    cap = io.StringIO()
    try:
        with contextlib.redirect_stdout(cap):
            exec(compile(src, f"<lab cell {i}>", "exec"), ctx)
    except Exception as e:
        print(f"FAILED di cell {i}: {type(e).__name__}: {e}")
        print(cap.getvalue()[-900:])
        sys.exit(1)
    out = cap.getvalue()
    if "✅" in out:
        last = [l for l in out.splitlines() if "✅" in l]
        print(f"[cell {i:>2}] OK  {last[0][:88] if last else ''}")
    else:
        print(f"[cell {i:>2}] OK ({len(out)} chars)")

print(f"\nSemua cell lab dieksekusi dalam {time.time() - t0:.1f} detik.")
print("LAB VERIFY OK")
