"""Spesifikasi Bagian 2 — Loop ReAct + guardrail iterasi (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di agentkit/loop_react.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit.data import MAKS_ITER
from agentkit.loop_react import (format_langkah, format_trace,
                                 jalankan_langkah, jalankan_rencana)
from agentkit.nlu import rencana_untuk


class TestFormatLangkah(unittest.TestCase):
    def test_format_terkunci(self):
        teks = format_langkah("Saya cek dulu.", ("cek_pesanan", {"invoice_id": "INV-001"}),
                              {"status": "dikirim"})
        self.assertEqual(
            teks,
            'Thought: Saya cek dulu.\n'
            'Action: cek_pesanan({"invoice_id": "INV-001"})\n'
            'Observation: {\'status\': \'dikirim\'}')

    def test_langkah_final_tanpa_observasi(self):
        teks = format_langkah("Selesai, jawab user.")
        self.assertEqual(teks, "Thought: Selesai, jawab user.")


class TestFormatTrace(unittest.TestCase):
    def test_dua_blok_dipisah_kosong(self):
        trace = format_trace(["Thought: A", "Thought: B"])
        self.assertEqual(trace, "Thought: A\n\nThought: B")


class TestJalankanLangkah(unittest.TestCase):
    def test_final_tanpa_tool(self):
        langkah = {"thought": "final", "tool": None, "args": {"aksi_final": "jawab"}}
        observasi, dipakai = jalankan_langkah(langkah)
        self.assertIsNone(observasi)
        self.assertIsNone(dipakai)

    def test_tool_dipanggil(self):
        langkah = {"thought": "cek", "tool": "cek_pesanan",
                   "args": {"invoice_id": "INV-001"}}
        observasi, dipakai = jalankan_langkah(langkah)
        self.assertEqual(dipakai, "cek_pesanan")
        self.assertEqual(observasi["status"], "dikirim")


class TestJalankanRencana(unittest.TestCase):
    def test_angka_terkunci_status_2_langkah(self):
        rencana = rencana_untuk("status_pesanan", "status INV-001 dong")
        hasil = jalankan_rencana(rencana)
        self.assertEqual(hasil["jumlah_langkah"], 2)
        self.assertFalse(hasil["berhenti_dini"])
        self.assertEqual(hasil["final"]["aksi_final"], "jawab")
        self.assertEqual(hasil["langkah"][0]["tool"], "cek_pesanan")
        self.assertEqual(hasil["langkah"][0]["observasi"]["status"], "dikirim")

    def test_rencana_refund_berakhir_konfirmasi(self):
        rencana = rencana_untuk("refund", "refund INV-002 barang rusak")
        hasil = jalankan_rencana(rencana)
        self.assertEqual(hasil["final"]["aksi_final"], "minta_konfirmasi")
        self.assertEqual(hasil["langkah"][1]["tool"], None)   # langkah final
        self.assertTrue(hasil["langkah"][0]["observasi"]["id"], "INV-002")

    def test_guardrail_iterasi(self):
        """Rencana tanpa langkah final → berhenti di maks_iter + escallate."""
        rencana = [{"thought": f"langkah {i}", "tool": "cari_dokumen",
                    "args": {"kueri": "pengembalian"}} for i in range(10)]
        hasil = jalankan_rencana(rencana, maks_iter=3)
        self.assertTrue(hasil["berhenti_dini"])
        self.assertEqual(hasil["jumlah_langkah"], 3)
        self.assertEqual(hasil["final"]["aksi_final"], "jawab")

    def test_guardrail_default_6(self):
        rencana = [{"thought": "loop", "tool": "cari_dokumen",
                    "args": {"kueri": "cod"}} for _ in range(20)]
        hasil = jalankan_rencana(rencana)
        self.assertEqual(hasil["jumlah_langkah"], MAKS_ITER)

    def test_rencana_kosong(self):
        hasil = jalankan_rencana([])
        self.assertTrue(hasil["berhenti_dini"])
        self.assertEqual(hasil["jumlah_langkah"], 0)


if __name__ == "__main__":
    unittest.main()
