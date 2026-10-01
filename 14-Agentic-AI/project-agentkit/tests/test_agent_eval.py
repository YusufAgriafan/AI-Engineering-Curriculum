"""Spesifikasi Bagian 4+5 — Memory, agent end-to-end, planner, eval (TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di agentkit/memory.py dan
agentkit/agent.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit.agent import (evaluasi_trace, jawab, planner_bertahap,
                            verifikasi_tools)
from agentkit.memory import MemoriSesi, hitung_biaya
from agentkit.nlu import token_estimasi


class TestMemoriSesi(unittest.TestCase):
    def test_tambah_mencatat_token(self):
        m = MemoriSesi()
        g = m.tambah("user", "abcd")     # token_estimasi('abcd') = 1
        self.assertEqual(g, {"peran": "user", "teks": "abcd", "token": 1})

    def test_angka_terkunci_konteks_maks3(self):
        m = MemoriSesi(maks_giliran=3)
        for p, t in [("user", "satu"), ("agent", "dua"),
                     ("user", "tiga"), ("agent", "empat")]:
            m.tambah(p, t)
        konteks = m.konteks()
        self.assertEqual([g["teks"] for g in konteks], ["dua", "tiga", "empat"])
        self.assertEqual([g["peran"] for g in konteks], ["agent", "user", "agent"])

    def test_ringkas_format(self):
        m = MemoriSesi()
        m.tambah("user", "halo")
        m.tambah("agent", "selamat datang")
        self.assertEqual(m.ringkas(), "user: halo\nagent: selamat datang")

    def test_cache_normalisasi_kunci(self):
        m = MemoriSesi()
        m.cache_put("  Status INV-001 ", {"ok": 1})
        self.assertEqual(m.cache_get("status inv-001"), {"ok": 1})
        self.assertIsNone(m.cache_get("kueri lain"))

    def test_cache_hit_miss_tercatat(self):
        m = MemoriSesi()
        m.cache_put("cod", {"d3": True})
        m.cache_get("cod")          # hit
        m.cache_get("cod")          # hit lagi
        m.cache_get("beli")         # miss
        self.assertEqual((m.cache_hit, m.cache_miss), (2, 1))


class TestHitungBiaya(unittest.TestCase):
    def test_angka_terkunci_api_mini(self):
        m = MemoriSesi()
        for p, t in [("user", "abcd"), ("agent", "abcdefgh")]:   # 1 + 2 token
            m.tambah(p, t)
        # total 3 token × 0.15/1e6 = 4.5e-07
        self.assertAlmostEqual(hitung_biaya(m.riwayat), 4.5e-07, places=12)

    def test_riwayat_kosong_nol(self):
        self.assertEqual(hitung_biaya([]), 0.0)


class TestJawab(unittest.TestCase):
    def test_angka_terkunci_status_2_langkah(self):
        hasil = jawab("Bagaimana status pesanan INV-001?")
        self.assertEqual(hasil["intent"], "status_pesanan")
        self.assertEqual(hasil["jumlah_langkah"], 2)
        self.assertFalse(hasil["berhenti_dini"])
        self.assertEqual(hasil["final"]["aksi_final"], "jawab")

    def test_refund_butuh_konfirmasi(self):
        hasil = jawab("Saya mau refund INV-002 barang rusak")
        self.assertEqual(hasil["intent"], "refund")
        self.assertEqual(hasil["final"]["aksi_final"], "minta_konfirmasi")

    def test_tanpa_id_tanya_balik(self):
        hasil = jawab("status pesanan dong")
        self.assertEqual(hasil["final"]["aksi_final"], "tanya_balik")
        self.assertIn("invoice", hasil["final"]["jawaban"])

    def test_lainnya_tidak_menebak(self):
        hasil = jawab("halo halo")
        self.assertEqual(hasil["intent"], "lainnya")
        self.assertEqual(hasil["final"]["aksi_final"], "tanya_balik")


class TestPlannerBertahap(unittest.TestCase):
    def test_angka_terkunci_enam_kasus(self):
        kasus = [("status_pesanan", "status pesanan dong", 1),
                 ("status_pesanan", "cek INV-003", 2),
                 ("refund", "refund INV-002 rusak", 2),
                 ("refund", "mau refund", 1),
                 ("kebijakan", "berapa lama pengiriman?", 2),
                 ("lainnya", "halo", 1)]
        for intent, teks, jumlah in kasus:
            self.assertEqual(len(planner_bertahap(intent, teks)), jumlah,
                             f"{intent} / {teks!r}")

    def test_bentuk_langkah_konsisten(self):
        rencana = planner_bertahap("status_pesanan", "cek INV-003")
        self.assertEqual(rencana[0]["tool"], "cek_pesanan")
        self.assertEqual(rencana[0]["args"], {"invoice_id": "INV-003"})
        self.assertIsNone(rencana[1]["tool"])


class TestEvaluasiTrace(unittest.TestCase):
    def test_trace_bersih_skor_1(self):
        hasil = jawab("Bagaimana status pesanan INV-001?")
        self.assertEqual(evaluasi_trace(hasil),
                         {"skor": 1.0, "temuan": ["bersih"]})

    def test_tanya_balik_skor_05(self):
        hasil = jawab("status pesanan dong")
        ev = evaluasi_trace(hasil)
        self.assertEqual(ev["skor"], 0.5)
        self.assertIn("ada klarifikasi ke user (tanya_balik)", ev["temuan"])

    def test_berhenti_dini_skor_0(self):
        from agentkit.loop_react import jalankan_rencana
        rencana = [{"thought": "loop", "tool": "cari_dokumen",
                    "args": {"kueri": "cod"}} for _ in range(10)]
        hasil = jalankan_rencana(rencana, maks_iter=3)
        ev = evaluasi_trace(hasil)
        self.assertEqual(ev["skor"], 0.0)
        self.assertIn("berhenti dini (guardrail iterasi)", ev["temuan"])

    def test_error_tool_skor_0(self):
        hasil = jawab("cek status INV-999 dong")
        ev = evaluasi_trace(hasil)
        self.assertEqual(ev["skor"], 0.0)
        self.assertTrue(any(t.startswith("tool error") for t in ev["temuan"]))


class TestVerifikasiTools(unittest.TestCase):
    def test_semua_tool_sehat(self):
        self.assertTrue(verifikasi_tools())


if __name__ == "__main__":
    unittest.main()
