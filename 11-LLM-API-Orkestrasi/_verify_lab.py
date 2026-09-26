"""Verify 01_lab_llm_api.ipynb: exec every cell; TODO cells get reference impls.

Usage (dari folder ini):
    python _verify_lab.py
"""
import contextlib  # noqa: E402
import io  # noqa: E402
import json
import os
import sys
import time  # noqa: E402
from pathlib import Path

BASE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.chdir(BASE)  # notebook memakai Path.cwd() untuk menemukan project-llmkit

NB = BASE / "01_lab_llm_api.ipynb"
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

REF = {
    "token_estimasi": (
        "def token_estimasi(teks):\n"
        "    return int(math.ceil(len(str(teks)) / 4.0))\n"
    ),
    "hitung_biaya": (
        "def hitung_biaya(tokens_in, tokens_out, model, harga=None):\n"
        "    tabel = HARGA if harga is None else harga\n"
        "    if model not in tabel:\n"
        "        raise ValueError(f'model tidak ada di daftar harga: {model}')\n"
        "    tarif = tabel[model]\n"
        "    return ((tokens_in / 1_000_000) * tarif['input']\n"
        "            + (tokens_out / 1_000_000) * tarif['output'])\n"
    ),
    "jalankan_anggaran": (
        "def jalankan_anggaran(provider, daftar_pesan, batas_usd, model='mini', timeout_ms=5000):\n"
        "    diproses, dilewati = 0, 0\n"
        "    total, hasil, berhenti_di = 0.0, [], None\n"
        "    for i, pesan in enumerate(daftar_pesan):\n"
        "        if total >= batas_usd:\n"
        "            if berhenti_di is None:\n"
        "                berhenti_di = i\n"
        "            dilewati += 1\n"
        "            continue\n"
        "        r = provider.panggil(model, pesan, timeout_ms=timeout_ms)\n"
        "        biaya = hitung_biaya(r['tokens_in'], r['tokens_out'], model)\n"
        "        total += biaya\n"
        "        hasil.append({'index': i, 'teks': r['teks'], 'biaya_usd': biaya})\n"
        "        diproses += 1\n"
        "    return {'diproses': diproses, 'dilewati': dilewati, 'total_biaya_usd': total,\n"
        "            'hasil': hasil, 'berhenti_di': berhenti_di}\n"
    ),
    "backoff_ms": (
        "def backoff_ms(percobaan, basis_ms=1000, faktor=2, maks_ms=8000):\n"
        "    return min(basis_ms * (faktor ** (percobaan - 1)), maks_ms)\n"
    ),
    "panggil_dengan_retry": (
        "def panggil_dengan_retry(provider, model, messages, maks_percobaan=4,\n"
        "                         basis_ms=1000, faktor=2, maks_ms=8000,\n"
        "                         timeout_ms=5000, sleep_fn=None, **params):\n"
        "    sleep_fn = (lambda ms: None) if sleep_fn is None else sleep_fn\n"
        "    riwayat, tunggu = [], []\n"
        "    for percobaan in range(1, maks_percobaan + 1):\n"
        "        try:\n"
        "            hasil = provider.panggil(model, messages, timeout_ms=timeout_ms, **params)\n"
        "        except GalatAPI as e:\n"
        "            riwayat.append((percobaan, e.jenis))\n"
        "            transien = e.jenis in JENIS_TRANSIEN\n"
        "            if not transien or percobaan == maks_percobaan:\n"
        "                if not transien:\n"
        "                    raise\n"
        "                raise GalatSemuaPercobaan(model, riwayat) from e\n"
        "            jeda = backoff_ms(percobaan, basis_ms, faktor, maks_ms)\n"
        "            tunggu.append(jeda)\n"
        "            sleep_fn(jeda)\n"
        "            continue\n"
        "        hasil = dict(hasil)\n"
        "        hasil['attempts'] = percobaan\n"
        "        hasil['timpenungguan_ms'] = list(tunggu)\n"
        "        hasil['total_ms'] = hasil['latency_ms'] + sum(tunggu)\n"
        "        hasil['riwayat'] = list(riwayat)\n"
        "        return hasil\n"
        "    raise GalatSemuaPercobaan(model, riwayat)\n"
    ),
    "panggil_dengan_fallback": (
        "def panggil_dengan_fallback(providers, rantai, messages, settings=None, **params):\n"
        "    dasar = dict(settings or {})\n"
        "    dicoba, riwayat = [], []\n"
        "    for i, tautan in enumerate(rantai):\n"
        "        nama, model = tautan['provider'], tautan['model']\n"
        "        provider = providers.get(nama)\n"
        "        if provider is None:\n"
        "            dicoba.append({'provider': nama, 'model': model, 'sukses': False,\n"
        "                           'jenis': 'provider_tidak_ada'})\n"
        "            riwayat.append((i, model, 'provider_tidak_ada'))\n"
        "            continue\n"
        "        setelan = {**dasar, **tautan.get('settings', {})}\n"
        "        try:\n"
        "            hasil = panggil_dengan_retry(provider, model, messages, **setelan, **params)\n"
        "        except GalatAPI as e:\n"
        "            dicoba.append({'provider': nama, 'model': model, 'sukses': False,\n"
        "                           'jenis': e.jenis})\n"
        "            riwayat.append((i, model, e.jenis))\n"
        "            continue\n"
        "        dicoba.append({'provider': nama, 'model': model, 'sukses': True,\n"
        "                       'jenis': None, 'attempts': hasil['attempts']})\n"
        "        hasil = dict(hasil)\n"
        "        hasil['dicoba'] = dicoba\n"
        "        hasil['lompatan'] = i\n"
        "        hasil['tautan'] = tautan\n"
        "        return hasil\n"
        "    raise GalatSemuaPercobaan(rantai[-1]['model'] if rantai else '-', riwayat,\n"
        "                              'seluruh rantai fallback gagal')\n"
    ),
    "CachePrompt": (
        "class CachePrompt:\n"
        "    def __init__(self, ttl=3):\n"
        "        self.ttl = ttl\n"
        "        self._isi = {}\n"
        "        self.hit = 0\n"
        "        self.miss = 0\n"
        "        self.expired = 0\n"
        "\n"
        "    def kunci(self, model, messages, params=None):\n"
        "        bahan = json.dumps({'model': model, 'messages': messages,\n"
        "                            'params': params or {}}, sort_keys=True, ensure_ascii=False)\n"
        "        return hashlib.sha256(bahan.encode('utf-8')).hexdigest()[:16]\n"
        "\n"
        "    def ambil(self, kunci, sekarang):\n"
        "        if kunci not in self._isi:\n"
        "            self.miss += 1\n"
        "            return None\n"
        "        hasil, tick = self._isi[kunci]\n"
        "        if sekarang - tick >= self.ttl:\n"
        "            self.expired += 1\n"
        "            del self._isi[kunci]\n"
        "            return None\n"
        "        self.hit += 1\n"
        "        return hasil\n"
        "\n"
        "    def simpan(self, kunci, hasil, sekarang):\n"
        "        self._isi[kunci] = (hasil, sekarang)\n"
        "\n"
        "    def statistik(self):\n"
        "        total = self.hit + self.miss + self.expired\n"
        "        return {'hit': self.hit, 'miss': self.miss, 'expired': self.expired,\n"
        "                'ukuran': len(self._isi),\n"
        "                'hit_rate': (self.hit / total) if total else 0.0}\n"
    ),
    "panggil_bercache": (
        "def panggil_bercache(provider, model, messages, cache, sekarang, **params):\n"
        "    kunci = cache.kunci(model, messages, params)\n"
        "    tersimpan = cache.ambil(kunci, sekarang)\n"
        "    if tersimpan is not None:\n"
        "        hasil = dict(tersimpan)\n"
        "        hasil['cache'] = 'hit'\n"
        "        return hasil\n"
        "    hasil = provider.panggil(model, messages, **params)\n"
        "    cache.simpan(kunci, hasil, sekarang)\n"
        "    hasil = dict(hasil)\n"
        "    hasil['cache'] = 'miss'\n"
        "    return hasil\n"
    ),
    "konsumsi_stream": (
        "def konsumsi_stream(chunks):\n"
        "    potongan, jumlah, ttft, terakhir = [], 0, None, 0\n"
        "    for c in chunks:\n"
        "        terakhir = c['t_ms']\n"
        "        if c['delta']:\n"
        "            jumlah += 1\n"
        "            if ttft is None:\n"
        "                ttft = c['t_ms']\n"
        "            potongan.append(c['delta'])\n"
        "    return {'teks': ''.join(potongan), 'jumlah_chunk': jumlah,\n"
        "            'ttft_ms': ttft, 'total_ms': terakhir}\n"
    ),
    "skema_parameter": (
        "TIPE_JSON = {str: 'string', int: 'integer', float: 'number', bool: 'boolean'}\n"
        "\n"
        "\n"
        "def skema_parameter(nama, tools=TOOLS):\n"
        "    tabel = TOOLS if tools is None else tools\n"
        "    argumen = tabel[nama]['argumen']\n"
        "    props = {k: {'type': TIPE_JSON.get(type(v), 'string')} for k, v in argumen.items()}\n"
        "    return {'type': 'object', 'properties': props,\n"
        "            'required': list(argumen.keys())}\n"
    ),
    "skema_tools": (
        "def skema_tools(tools=TOOLS):\n"
        "    tabel = TOOLS if tools is None else tools\n"
        "    return [{'type': 'function',\n"
        "             'function': {'name': nama,\n"
        "                          'description': spec['deskripsi'],\n"
        "                          'parameters': skema_parameter(nama, tabel)}}\n"
        "            for nama, spec in tabel.items()]\n"
    ),
    "validasi_argumen": (
        "def _tipe_ok(nilai, tipe):\n"
        "    if tipe == 'string':\n"
        "        return isinstance(nilai, str)\n"
        "    if tipe == 'integer':\n"
        "        return isinstance(nilai, int) and not isinstance(nilai, bool)\n"
        "    if tipe == 'number':\n"
        "        return isinstance(nilai, (int, float)) and not isinstance(nilai, bool)\n"
        "    if tipe == 'boolean':\n"
        "        return isinstance(nilai, bool)\n"
        "    return False\n"
        "\n"
        "\n"
        "def validasi_argumen(nama, argumen, tools=TOOLS):\n"
        "    tabel = TOOLS if tools is None else tools\n"
        "    if nama not in tabel:\n"
        "        return [f'tak_dikenal:{nama}']\n"
        "    skema = skema_parameter(nama, tabel)\n"
        "    errors = []\n"
        "    if not isinstance(argumen, dict):\n"
        "        return ['type:<argumen>']\n"
        "    for k in skema['required']:\n"
        "        if k not in argumen:\n"
        "            errors.append(f'missing:{k}')\n"
        "    for k, v in argumen.items():\n"
        "        if k not in skema['properties']:\n"
        "            errors.append(f'tak_dikenal:{k}')\n"
        "            continue\n"
        "        if not _tipe_ok(v, skema['properties'][k]['type']):\n"
        "            errors.append(f'type:{k}')\n"
        "    return errors\n"
    ),
    "jalankan_tool": (
        "def jalankan_tool(nama, argumen, tools=TOOLS):\n"
        "    tabel = TOOLS if tools is None else tools\n"
        "    errors = validasi_argumen(nama, argumen, tabel)\n"
        "    if errors:\n"
        "        return {'hasil': None, 'galat': '; '.join(errors), 'errors': errors}\n"
        "    return {'hasil': tabel[nama]['handler'](argumen), 'galat': None, 'errors': []}\n"
    ),
    "loop_tool": (
        "def loop_tool(provider, model, pesan_awal, tools=TOOLS, maks_iterasi=4):\n"
        "    skema = skema_tools(tools)\n"
        "    messages = [{'role': 'user', 'content': pesan_awal}]\n"
        "    tokens_in = tokens_out = total_ms = 0\n"
        "    dipakai, riwayat = [], []\n"
        "    for it in range(1, maks_iterasi + 1):\n"
        "        hasil = provider.panggil(model, messages, tools=skema)\n"
        "        tokens_in += hasil['tokens_in']\n"
        "        tokens_out += hasil['tokens_out']\n"
        "        total_ms += hasil['latency_ms']\n"
        "        if not hasil['tool_calls']:\n"
        "            return {'jawaban': hasil['teks'], 'iterasi': it, 'tool_dipakai': dipakai,\n"
        "                    'riwayat': riwayat, 'tokens_in': tokens_in,\n"
        "                    'tokens_out': tokens_out, 'total_ms': total_ms}\n"
        "        for tc in hasil['tool_calls']:\n"
        "            messages.append({'role': 'assistant', 'content': None, 'tool_calls': [tc]})\n"
        "            res = jalankan_tool(tc['nama'], tc['argumen'], tools)\n"
        "            messages.append({'role': 'tool', 'tool_call_id': tc['id'],\n"
        "                             'content': json.dumps(res['hasil'] if res['galat'] is None\n"
        "                                                   else {'galat': res['galat']})})\n"
        "            dipakai.append(tc['nama'])\n"
        "            riwayat.append({'iterasi': it, 'tool': tc['nama'], 'argumen': tc['argumen'],\n"
        "                            'galat': res['galat'], 'hasil': res['hasil']})\n"
        "    return {'jawaban': None, 'iterasi': maks_iterasi, 'tool_dipakai': dipakai,\n"
        "            'riwayat': riwayat, 'tokens_in': tokens_in, 'tokens_out': tokens_out,\n"
        "            'total_ms': total_ms, 'berhenti': 'maks_iterasi'}\n"
    ),
    "jalankan_pipeline": (
        "def jalankan_pipeline(provider, langkah, teks, maks_percobaan=3, timeout_ms=5000):\n"
        "    rincian, total_biaya, total_ms = [], 0.0, 0\n"
        "    berhasil = gagal = 0\n"
        "    teks_akhir = None\n"
        "    for lang in langkah:\n"
        "        prompt = lang['template'].format(teks=teks)\n"
        "        messages = [{'role': 'user', 'content': prompt}]\n"
        "        try:\n"
        "            hasil = panggil_dengan_retry(\n"
        "                provider, lang['model'], messages,\n"
        "                maks_percobaan=maks_percobaan, timeout_ms=timeout_ms,\n"
        "                temperature=lang.get('suhu', 0.0),\n"
        "                maks_token=lang.get('maks_token', 200))\n"
        "            biaya = hitung_biaya(hasil['tokens_in'], hasil['tokens_out'], hasil['model'])\n"
        "            total_biaya += biaya\n"
        "            total_ms += hasil['total_ms']\n"
        "            berhasil += 1\n"
        "            teks_akhir = hasil['teks']\n"
        "            rincian.append({'nama': lang['nama'], 'model': lang['model'], 'sukses': True,\n"
        "                            'teks': hasil['teks'], 'attempts': hasil['attempts'],\n"
        "                            'total_ms': hasil['total_ms'], 'biaya_usd': biaya,\n"
        "                            'galat': None})\n"
        "        except GalatAPI as e:\n"
        "            gagal += 1\n"
        "            rincian.append({'nama': lang['nama'], 'model': lang['model'], 'sukses': False,\n"
        "                            'teks': None, 'attempts': None, 'total_ms': 0,\n"
        "                            'biaya_usd': 0.0, 'galat': e.jenis})\n"
        "    return {'langkah': rincian, 'berhasil': berhasil, 'gagal': gagal,\n"
        "            'total_biaya_usd': total_biaya, 'total_ms': total_ms,\n"
        "            'teks_akhir': teks_akhir}\n"
    ),
}


def nama_ref(src):
    """Nama REF yang didefinisikan di cell ini (fungsi ATAU kelas)."""
    return [n for n in REF
            if f"def {n}(" in src or f"class {n}(" in src or f"class {n}:" in src]


ctx = {"__name__": "__main__", "__builtins__": __builtins__}

t0 = time.time()
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "TODO" in src:
        names = nama_ref(src)
        if not names:
            print(f"[cell {i:>2}] SKIPPED (TODO tanpa fungsi/kelas dikenal)")
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
