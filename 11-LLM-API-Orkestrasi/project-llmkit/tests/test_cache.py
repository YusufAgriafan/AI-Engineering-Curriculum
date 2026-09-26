"""Spesifikasi Bagian 4 — Cache konten-address + TTL (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di llmkit/cache.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmkit.cache import CachePrompt, panggil_bercache
from llmkit.transport import provider_utama

PESAN = [{"role": "user", "content": "Jelaskan apa itu machine learning dalam 2 kalimat."}]


class TestKunci(unittest.TestCase):
    def setUp(self):
        self.c = CachePrompt()

    def test_deterministik_dan_pendek(self):
        k = self.c.kunci("mini", PESAN)
        self.assertEqual(len(k), 16)
        self.assertEqual(k, self.c.kunci("mini", PESAN))
        self.assertEqual(k, "f833386edc2ecf9c")

    def test_model_berbeda_kunci_berbeda(self):
        self.assertNotEqual(self.c.kunci("mini", PESAN), self.c.kunci("sedang", PESAN))

    def test_param_berbeda_kunci_berbeda(self):
        a = self.c.kunci("mini", PESAN, {"temperature": 0.0})
        b = self.c.kunci("mini", PESAN, {"temperature": 1.0})
        self.assertNotEqual(a, b)

    def test_urutan_key_param_tidak_mengubah_kunci(self):
        self.assertEqual(self.c.kunci("mini", PESAN, {"a": 1, "b": 2}),
                         self.c.kunci("mini", PESAN, {"b": 2, "a": 1}))


class TestAmbilSimpan(unittest.TestCase):
    def setUp(self):
        self.c = CachePrompt(ttl=3)

    def test_miss_lalu_hit(self):
        self.assertIsNone(self.c.ambil("k", 0))
        self.c.simpan("k", {"teks": "hai"}, 0)
        self.assertEqual(self.c.ambil("k", 1), {"teks": "hai"})

    def test_kedaluwarsa_pada_batas_ttl(self):
        self.c.simpan("k", {"teks": "hai"}, 0)
        self.assertEqual(self.c.ambil("k", 2), {"teks": "hai"})    # 2 < 3: masih segar
        self.assertIsNone(self.c.ambil("k", 3))                    # 3 >= 3: kedaluwarsa

    def test_kedaluwarsa_menghapus_entri(self):
        self.c.simpan("k", {"x": 1}, 0)
        self.c.ambil("k", 5)
        self.assertEqual(self.c.statistik()["ukuran"], 0)

    def test_bersihkan(self):
        self.c.simpan("a", {"x": 1}, 0)
        self.c.simpan("b", {"x": 2}, 4)
        self.assertEqual(self.c.bersihkan(5), 1)

    def test_statistik(self):
        self.c.ambil("x", 0)                     # miss
        self.c.simpan("x", {"v": 1}, 0)
        self.c.ambil("x", 1)                     # hit
        self.c.ambil("x", 9)                     # expired
        s = self.c.statistik()
        self.assertEqual((s["hit"], s["miss"], s["expired"]), (1, 1, 1))
        self.assertAlmostEqual(s["hit_rate"], 1 / 3, places=9)

    def test_hit_rate_kosong(self):
        self.assertEqual(CachePrompt().statistik()["hit_rate"], 0.0)


class TestPanggilBercache(unittest.TestCase):
    def setUp(self):
        self.c = CachePrompt(ttl=3)
        self.p = provider_utama()

    def test_miss_lalu_hit(self):
        r1 = panggil_bercache(self.p, "mini", PESAN, self.c, sekarang=0)
        r2 = panggil_bercache(self.p, "mini", PESAN, self.c, sekarang=1)
        self.assertEqual((r1["cache"], r2["cache"]), ("miss", "hit"))
        self.assertEqual(r1["teks"], r2["teks"])

    def test_hit_tidak_memanggil_provider(self):
        panggil_bercache(self.p, "mini", PESAN, self.c, sekarang=0)
        panggil_bercache(self.p, "mini", PESAN, self.c, sekarang=1)
        self.assertEqual(self.p.jumlah_panggilan["mini"], 1, "hit tidak boleh menyentuh provider")

    def test_expired_memanggil_provider_lagi(self):
        panggil_bercache(self.p, "mini", PESAN, self.c, sekarang=0)
        r = panggil_bercache(self.p, "mini", PESAN, self.c, sekarang=9)
        self.assertEqual(r["cache"], "miss")
        self.assertEqual(self.p.jumlah_panggilan["mini"], 2)

    def test_pola_empat_panggilan(self):
        pola = [panggil_bercache(self.p, "mini", PESAN, self.c, sekarang=t)["cache"]
                for t in (0, 1, 5, 6)]
        self.assertEqual(pola, ["miss", "hit", "miss", "hit"])
        s = self.c.statistik()
        self.assertEqual((s["hit"], s["miss"], s["expired"]), (2, 1, 1))
        self.assertAlmostEqual(s["hit_rate"], 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
