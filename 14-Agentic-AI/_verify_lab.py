"""Verify 01_lab_agentic.ipynb: exec every cell; TODO cells get reference impls.

Usage (dari folder ini):
    python _verify_lab.py
"""
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

BASE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.chdir(BASE)  # notebook memakai Path.cwd() untuk menemukan project-agentkit

NB = BASE / "01_lab_agentic.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "langkah_final": (
        "def langkah_final(aksi_final, jawaban=None):\n"
        "    args = {'aksi_final': aksi_final}\n"
        "    if jawaban is not None:\n"
        "        args['jawaban'] = jawaban\n"
        "    return {'thought': 'langkah final', 'tool': None, 'args': args}\n"
    ),
    "rencana_status": (
        "def rencana_status(teks):\n"
        "    inv = ekstrak_invoice(teks)\n"
        "    if inv is None:\n"
        "        return [langkah_final('tanya_balik',\n"
        "                              'Boleh sebutkan nomor invoice Anda (format INV-XXX)?')]\n"
        "    return [{'thought': f'Saya perlu cek status pesanan {inv} di sistem.',\n"
        "             'tool': 'cek_pesanan', 'args': {'invoice_id': inv}},\n"
        "            langkah_final('jawab')]\n"
    ),
    "validasi_skema": (
        "def validasi_skema(nama_tool, args):\n"
        "    skema = TOOLS_SCHEMA.get(nama_tool)\n"
        "    if skema is None:\n"
        "        return False, f'tool tidak dikenal: {nama_tool}'\n"
        "    if not isinstance(args, dict):\n"
        "        return False, 'argumen harus berupa object/dict'\n"
        "    spec = skema['argumen']\n"
        "    for nama in spec.get('required', []):\n"
        "        if nama not in args:\n"
        "            return False, f'argumen wajib hilang: {nama}'\n"
        "    for nama, aturan in spec.get('properties', {}).items():\n"
        "        if nama not in args:\n"
        "            continue\n"
        "        nilai = args[nama]\n"
        "        if aturan.get('type') == 'string' and not isinstance(nilai, str):\n"
        "            return False, f'argumen {nama} harus string'\n"
        "        if 'min_panjang' in aturan and len(nilai) < aturan['min_panjang']:\n"
        "            return False, f'argumen {nama} minimal {aturan[\"min_panjang\"]} karakter'\n"
        "        if 'pola' in aturan and not re.search(aturan['pola'], nilai):\n"
        "            return False, f'argumen {nama} tidak cocok pola {aturan[\"pola\"]}'\n"
        "    return True, ''\n"
    ),
    "tool_cek_pesanan": (
        "def tool_cek_pesanan(invoice_id):\n"
        "    pesanan = PESANAN.get(str(invoice_id).upper())\n"
        "    if pesanan is None:\n"
        "        return {'error': 'pesanan tidak ditemukan'}\n"
        "    keluar = dict(pesanan)\n"
        "    keluar.pop('user', None)\n"
        "    return keluar\n"
    ),
    "format_langkah": (
        "def format_langkah(thought, action=None, observation=None):\n"
        "    baris = [f'Thought: {thought}']\n"
        "    if action is not None:\n"
        "        nama, args = action\n"
        "        baris.append(f'Action: {nama}({json.dumps(args, ensure_ascii=False)})')\n"
        "    if observation is not None:\n"
        "        baris.append(f'Observation: {observation}')\n"
        "    return '\\n'.join(baris)\n"
    ),
    "format_trace": (
        "def format_trace(langkah_list):\n"
        "    return '\\n\\n'.join(langkah_list)\n"
    ),
    "jalankan_rencana": (
        "def jalankan_rencana(rencana, maks_iter=MAKS_ITER):\n"
        "    langkah_tereksekusi = []\n"
        "    for langkah in rencana[:maks_iter]:\n"
        "        if langkah.get('tool') is None:\n"
        "            catatan = {'thought': langkah['thought'], 'tool': None,\n"
        "                       'args': langkah.get('args'), 'observasi': None}\n"
        "            langkah_tereksekusi.append(catatan)\n"
        "            return {'final': dict(langkah.get('args') or {}),\n"
        "                    'langkah': langkah_tereksekusi, 'berhenti_dini': False,\n"
        "                    'jumlah_langkah': len(langkah_tereksekusi)}\n"
        "        catatan = {'thought': langkah['thought'], 'tool': langkah['tool'],\n"
        "                   'args': langkah.get('args'), 'observasi': 'OK'}\n"
        "        langkah_tereksekusi.append(catatan)\n"
        "    return {'final': {'aksi_final': 'jawab',\n"
        "                      'jawaban': 'Saya belum bisa menyelesaikan permintaan ini. '\n"
        "                                 'Mohon hubungi agen manusia.'},\n"
        "            'langkah': langkah_tereksekusi, 'berhenti_dini': True,\n"
        "            'jumlah_langkah': len(langkah_tereksekusi)}\n"
    ),
    "sanitasi": (
        "def sanitasi(teks):\n"
                "    bersih = str(teks)\n"
        "    for frasa in POLA_INJECTION:\n"
        "        pola = re.compile(re.escape(frasa), re.IGNORECASE)\n"
        "        bersih = pola.sub('[dihapus]', bersih)\n"
        "    bersih = bersih.replace('\\n', ' ')\n"
        "    return bersih.strip()\n"
    ),
    "eksekusi_tool": (
        "def eksekusi_tool(nama_tool, args):\n"
        "    ok, pesan = validasi_skema(nama_tool, args)\n"
        "    if not ok:\n"
        "        return {'error': pesan}\n"
        "    fn = REGISTRY.get(nama_tool)\n"
        "    if fn is None:\n"
        "        return {'error': f'tool tidak dikenal: {nama_tool}'}\n"
        "    try:\n"
        "        return fn(**args)\n"
        "    except Exception as e:\n"
        "        return {'error': f'{type(e).__name__}: {e}'}\n"
    ),
    "MemoriSesi": (
        "class MemoriSesi:\n"
        "    def __init__(self, maks_giliran=4):\n"
        "        self.maks_giliran = maks_giliran\n"
        "        self.riwayat = []\n"
        "        self.cache = {}\n"
        "        self.cache_hit = 0\n"
        "        self.cache_miss = 0\n"
        "\n"
        "    def tambah(self, peran, teks):\n"
        "        giliran = {'peran': peran, 'teks': teks, 'token': token_estimasi(teks)}\n"
        "        self.riwayat.append(giliran)\n"
        "        return giliran\n"
        "\n"
        "    def konteks(self):\n"
        "        return list(self.riwayat[-self.maks_giliran:])\n"
        "\n"
        "    def ringkas(self):\n"
        "        return '\\n'.join(f\"{g['peran']}: {g['teks']}\" for g in self.riwayat)\n"
        "\n"
        "    def cache_get(self, kueri):\n"
        "        nilai = self.cache.get(str(kueri).strip().lower())\n"
        "        if nilai is None:\n"
        "            self.cache_miss += 1\n"
        "        else:\n"
        "            self.cache_hit += 1\n"
        "        return nilai\n"
        "\n"
        "    def cache_put(self, kueri, nilai):\n"
        "        self.cache[str(kueri).strip().lower()] = nilai\n"
    ),
    "hitung_biaya": (
        "def hitung_biaya(pesan, model='api_mini'):\n"
        "    if not pesan:\n"
        "        return 0.0\n"
        "    tarif = HARGA[model]\n"
        "    total_token = sum(g['token'] for g in pesan)\n"
        "    return total_token / 1e6 * tarif['input']\n"
    ),
    "planner_bertahap_status": (
        "def planner_bertahap_status(teks):\n"
        "    inv = ekstrak_invoice(teks)\n"
        "    if inv is None:\n"
        "        return [langkah_final('tanya_balik',\n"
        "                              'Boleh sebutkan nomor invoice Anda (format INV-XXX)?')]\n"
        "    return [{'thought': f'Saya perlu cek status pesanan {inv} di sistem.',\n"
        "             'tool': 'cek_pesanan', 'args': {'invoice_id': inv}},\n"
        "            langkah_final('jawab')]\n"
    ),
    "evaluasi_trace": (
        "def evaluasi_trace(hasil):\n"
        "    temuan = []\n"
        "    ada_error = bool(hasil.get('berhenti_dini'))\n"
        "    ada_tanya = False\n"
        "    if ada_error:\n"
        "        temuan.append('berhenti dini (guardrail iterasi)')\n"
        "    for l in hasil.get('langkah', []):\n"
        "        final = l.get('args') or {}\n"
        "        if isinstance(final, dict) and final.get('aksi_final') == 'tanya_balik':\n"
        "            ada_tanya = True\n"
        "        if isinstance(l.get('observasi'), dict) and 'error' in l['observasi']:\n"
        "            ada_error = True\n"
        "            temuan.append(f\"tool error: {l['observasi']['error']}\")\n"
        "    if ada_tanya:\n"
        "        temuan.append('ada klarifikasi ke user (tanya_balik)')\n"
        "    if ada_error:\n"
        "        skor = 0.0\n"
        "    elif ada_tanya:\n"
        "        skor = 0.5\n"
        "    else:\n"
        "        skor = 1.0\n"
        "    if not temuan:\n"
        "        temuan = ['bersih']\n"
        "    return {'skor': skor, 'temuan': temuan}\n"
    ),
}


def nama_ref(src):
    """Nama REF yang didefinisikan di cell ini."""
    return [n for n in REF if f"def {n}(" in src or f"class {n}" in src]


ctx = {"__name__": "__main__", "__builtins__": __builtins__}

t0 = time.time()
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src:
        names = nama_ref(src)
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
        print(cap.getvalue()[-1500:])
        sys.exit(1)
    out = cap.getvalue()
    if "\u2705" in out:
        baris = [l for l in out.splitlines() if "\u2705" in l]
        print(f"[cell {i:>2}] OK  {baris[0][:88] if baris else ''}")
    else:
        print(f"[cell {i:>2}] OK ({len(out)} chars)")

print(f"\nSemua cell lab dieksekusi dalam {time.time() - t0:.1f} detik.")
print("LAB VERIFY OK")
