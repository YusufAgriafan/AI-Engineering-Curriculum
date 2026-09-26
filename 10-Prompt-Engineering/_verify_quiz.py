"""Verify quiz notebook: inject perfect answers, run grader, expect full score.

Usage (dari folder ini):
    python _verify_quiz.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_kuis_prompt_engineering.ipynb"
EXPECTED = 22

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "b", 2: "b", 3: "c", 4: "c", 5: "a", 6: "b"},
    "bagian_hilang": (
        "def bagian_hilang(prompt):\n"
        "    return [b for b in BAGIAN_WAJIB if f'[{b}]' not in prompt]\n"
    ),
    "render": (
        "PAT = re.compile(r'\\{\\{\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\}\\}')\n"
        "\n"
        "def render(template, **nilai):\n"
        "    kurang = [n for n in sorted(set(PAT.findall(template))) if n not in nilai]\n"
        "    if kurang:\n"
        "        raise ValueError(f'placeholder tanpa nilai: {kurang}')\n"
        "    return PAT.sub(lambda m: str(nilai[m.group(1)]), template)\n"
    ),
    "ekstrak_json": (
        "def _perbaiki(teks):\n"
        "    t = str(teks).strip()\n"
        "    t = re.sub(r'//[^\\n]*', '', t)\n"
        "    t = t.replace('True', 'true').replace('False', 'false').replace('None', 'null')\n"
        "    t = re.sub(r\"'([^']*)'\", r'\"\\1\"', t)\n"
        "    t = re.sub(r',\\s*([}\\]])', r'\\1', t)\n"
        "    return json.loads(t)\n"
        "\n"
        "\n"
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
        "                            return _perbaiki(kandidat)\n"
        "                        except Exception:\n"
        "                            break\n"
        "        pos = t.find('{', pos + 1)\n"
        "    raise ValueError('tidak ada objek JSON valid di output')\n"
    ),
    "bungkus_data": (
        "def bungkus_data(teks, tag='dokumen'):\n"
        "    bersih = str(teks).replace(f'</{tag}>', '').replace(f'<{tag}>', '')\n"
        "    return f'<{tag}>\\n{bersih}\\n</{tag}>'\n"
    ),
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ctx = {"__name__": "__main__"}

for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "JAWABAN = {" in src and "None" in src:
        exec("JAWABAN = " + repr(ANSWERS["JAWABAN"]), ctx)
        print(f"[cell {i}] injected JAWABAN")
        continue
    if "TODO" in src:
        names = [n for n in ANSWERS if n != "JAWABAN" and f"def {n}(" in src]
        if not names:
            print(f"[cell {i}] SKIPPED (TODO tanpa fungsi dikenal)")
            continue
        # cell TODO asli TIDAK dieksekusi; timpa dengan implementasi referensi
        for n in names:
            exec(ANSWERS[n], ctx)
            print(f"[cell {i}] injected {n}")
        continue
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, ctx)
    out = buf.getvalue()
    print(f"[cell {i}] executed ({len(out)} chars)")
    if "SKOR AKHIR" in out:
        print(out)
        m = re.search(r"SKOR AKHIR:\s*(\d+)/(\d+)", out)
        skor, total = int(m.group(1)), int(m.group(2))
        status = "PASS" if (skor == total == EXPECTED) else "FAIL"
        print(f">>> {status}: {skor}/{total} (expected {EXPECTED})")
        sys.exit(0 if status == "PASS" else 1)
