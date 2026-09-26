"""Spesifikasi Bagian 6 — Reranker skor kata (mode TDD).

JANGAN DIUBAH — tugasmu membuat implementasi di ragkit/rerank.py lolos.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.rerank import rerank, skorer_kata


def _r(index, teks):
    return {"chunk": {"chunk_index": index, "text": teks,
                      "source": "s.pdf", "page": 1, "doc_id": "d"},
            "score": 0.5, "index": index}


class TestSkorerKata(unittest.TestCase):
    def test_angka_bobot_tiga(self):
        self.assertAlmostEqual(skorer_kata("cuti 12 hari", "12 hari cuti tahunan"),
                               3.0 + 1.0)   # '12' ×3 + 'cuti' ×1 ('hari' stopword)

    def test_kata_tidak_ada_nol(self):
        self.assertEqual(skorer_kata("pesawat bali", "cuti tahunan 12 hari"), 0.0)

    def test_stopword_dibuang(self):
        # semua kata adalah stopword → 0.0
        self.assertEqual(skorer_kata("berapa hari yang di", "apa dan atau"), 0.0)

    def test_substring_dihitung(self):
        self.assertEqual(skorer_kata("portal", "form via portal HR"), 1.0)


class TestRerank(unittest.TestCase):
    def test_urutan_berubah(self):
        hasil = [_r(3, "Surat keterangan dokter wajib untuk cuti sakit."),
                 _r(5, "FAQ jam kantor dan wi-fi."),
                 _r(4, "Cuti melahirkan 90 hari.")]
        out = rerank("Berapa lama maksimal cuti sakit tanpa surat dokter?",
                     hasil, top_k=2)
        # skorer: 'sakit' 1 + 'surat' 1 + 'dokter' 1 + 'maksimal' 1 = 4
        # vs 'melahirkan 90 hari' → hanya '90' (angka ×3) = 3
        self.assertEqual([r["chunk"]["chunk_index"] for r in out], [3, 4])

    def test_top_k_kecil(self):
        hasil = [_r(i, f"teks {i}") for i in range(5)]
        out = rerank("teks", hasil, top_k=2)
        self.assertEqual(len(out), 2)

    def test_tie_ke_index_kecil(self):
        hasil = [_r(2, "cuti 5 hari"), _r(1, "cuti 5 hari")]
        out = rerank("cuti", hasil, top_k=2)
        self.assertEqual([r["chunk"]["chunk_index"] for r in out], [1, 2])

    def test_skorer_kustom(self):
        hasil = [_r(0, "apel"), _r(1, "jeruk")]
        out = rerank("jeruk", hasil, top_k=1,
                     skorer=lambda k, t: 1.0 if "jeruk" in t else 0.0)
        self.assertEqual(out[0]["chunk"]["chunk_index"], 1)


if __name__ == "__main__":
    unittest.main()
