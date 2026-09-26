"""Spesifikasi Bagian 5 — Streaming: merakit chunk & mengukur TTFT (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di llmkit/stream.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmkit.data import MS_PER_CHUNK, TTFT_MS
from llmkit.stream import konsumsi_stream
from llmkit.transport import provider_utama

PESAN = [{"role": "user", "content": "Jelaskan apa itu machine learning dalam 2 kalimat."}]
TEKS = '[mini] jawaban untuk "Jelaskan apa itu machine learning dalam 2 kalimat."'


class TestKonsumsiStream(unittest.TestCase):
    def test_angka_terkunci(self):
        s = konsumsi_stream(provider_utama().stream("mini", PESAN))
        self.assertEqual(s["teks"], TEKS)
        self.assertEqual(s["jumlah_chunk"], 11)
        self.assertEqual(s["ttft_ms"], TTFT_MS)
        self.assertEqual(s["total_ms"], TTFT_MS + 11 * MS_PER_CHUNK)
        self.assertEqual(s["total_ms"], 340)

    def test_teks_utuh_tanpa_spasi_hilang(self):
        s = konsumsi_stream(provider_utama().stream("mini", PESAN))
        self.assertNotIn('  "', s["teks"])
        self.assertTrue(s["teks"].startswith("[mini] jawaban untuk"))

    def test_ttft_dari_chunk_pertama(self):
        chunks = [{"delta": "", "t_ms": 10},
                  {"delta": "Halo", "t_ms": 55},
                  {"delta": " dunia", "t_ms": 90}]
        s = konsumsi_stream(chunks)
        self.assertEqual((s["teks"], s["ttft_ms"], s["total_ms"], s["jumlah_chunk"]),
                         ("Halo dunia", 55, 90, 2))

    def test_chunk_penutup_kosong_tidak_dihitung(self):
        s = konsumsi_stream([{"delta": "a", "t_ms": 5}, {"delta": "", "t_ms": 9}])
        self.assertEqual(s["jumlah_chunk"], 1)
        self.assertEqual(s["total_ms"], 9)

    def test_stream_tanpa_konten(self):
        s = konsumsi_stream([{"delta": "", "t_ms": 42}])
        self.assertEqual((s["teks"], s["jumlah_chunk"], s["ttft_ms"]), ("", 0, None))

    def test_tanpa_chunk(self):
        s = konsumsi_stream([])
        self.assertEqual((s["teks"], s["ttft_ms"], s["total_ms"]), ("", None, 0))

    def test_urutan_delta_dijaga(self):
        s = konsumsi_stream([{"delta": "1", "t_ms": 1}, {"delta": "2", "t_ms": 2},
                             {"delta": "3", "t_ms": 3}])
        self.assertEqual(s["teks"], "123")


class TestProviderStream(unittest.TestCase):
    def test_stream_adalah_generator(self):
        import types
        gen = provider_utama().stream("mini", PESAN)
        self.assertIsInstance(gen, types.GeneratorType)

    def test_chunk_pertama_muncul_di_ttft(self):
        gen = provider_utama().stream("mini", PESAN)
        pertama = next(gen)
        self.assertEqual(pertama["t_ms"], TTFT_MS)

    def test_latensi_naik_per_chunk(self):
        gen = provider_utama().stream("mini", PESAN)
        waktu = [c["t_ms"] for c in gen if c["delta"]]
        self.assertEqual(waktu, sorted(waktu))
        self.assertEqual(waktu[1] - waktu[0], MS_PER_CHUNK)


if __name__ == "__main__":
    unittest.main()
