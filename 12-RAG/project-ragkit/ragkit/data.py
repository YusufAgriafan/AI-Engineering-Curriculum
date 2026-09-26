"""Dataset, tarif, & konfigurasi terkunci project Bab 12 (RAG) — JANGAN DIUBAH.

TANPA API key dan TANPA jaringan — pola yang sama dengan Bab 9-11:

- "Embedding model" di sini adalah `ragkit/embed.py`: encoder **palsu
  deterministik** berbasis hashing n-gram (hashing trick). Ia punya semua sifat
  yang kita butuhkan untuk melatih RAG: semantik (teks mirip → vektor mirip),
  normalisasi L2, dimensi tetap, dan **reproducible** (teks yang sama → vektor
  yang sama, di komputer siapa pun).

- "LLM generator" di sini adalah `ragkit/generator.py`: generator ekstraktif
  deterministik yang menjawab HANYA dari konteks yang diberikan. Ia sengaja
  tidak punya pengetahuan luar — jadi halusinasi tidak bisa disalahkan ke model;
  kalau jawabannya salah, penyebabnya retrieval atau prompt.

Angka-angka (jumlah chunk, skor, hit@k, hit-rate cache, biaya) TERKUNCI SEED 12
dan diverifikasi `tests/` + `_calib.py` + `_verify_project.py`.
"""
SEED = 12

# --------------------------------------------------------------------------
# 1. Korpus dokumen kebijakan HR (3 dokumen; tiap string = satu "halaman")
#    Angka kebijakan SENGAJA dibuat jelas: 12 hari, 5 hari, 90 hari, 14 hari,
#    3 hari — supaya jawaban generator bisa di-assert secara string.
# --------------------------------------------------------------------------
DOKUMEN = [
    {
        "id": "doc1",
        "judul": "Kebijakan Ketenagakerjaan",
        "source": "kebijakan-ketenagakerjaan.pdf",
        "halaman": [
            """CUTI TAHUNAN. Karyawan tetap mendapatkan 12 hari cuti tahunan per
tahun kalender. Cuti tahunan harus digunakan dalam tahun yang sama dan tidak
dapat ditumpuk ke tahun berikutnya; sisa cuti yang tidak digunakan hangus.
Permohonan cuti diajukan minimal 14 hari sebelum tanggal cuti melalui portal
HR. Cuti tahunan tidak termasuk akhir pekan dan hari libur nasional.""",
            """GAJI DAN TUNJANGAN. Gaji dibayarkan setiap tanggal 25 melalui
transfer bank. Tunjangan kesehatan diberikan kepada karyawan tetap dan
keluarga inti. Penilaian kinerja dilakukan dua kali setahun. Kenaikan gaji
ditetapkan setelah penilaian kinerja dan disetujui direksi.""",
        ],
    },
    {
        "id": "doc2",
        "judul": "Kebijakan Cuti Kesehatan",
        "source": "kebijakan-cuti-kesehatan.pdf",
        "halaman": [
            """CUTI SAKIT. Karyawan yang sakit wajib melampirkan surat keterangan
dokter. Cuti sakit maksimal 5 hari per kejadian tanpa dokumen medis; lebih
dari itu wajib melampirkan surat dokter. Cuti sakit lebih dari 90 hari masuk
asuransi dan dikonsultasikan dengan HR.""",
            """CUTI KHUSUS. Cuti melahirkan 90 hari sesuai ketentuan
perusahaan. Cuti ayah 3 hari untuk kelahiran anak. Cuti menikah 3 hari.
Pengajuan cuti khusus dilakukan paling lambat 7 hari sebelum tanggal
kejadian melalui portal HR.""",
        ],
    },
    {
        "id": "doc3",
        "judul": "FAQ Karyawan",
        "source": "faq-karyawan.pdf",
        "halaman": [
            """FAQ CUTI. Bagaimana cara mengajukan cuti? Submit form cuti
tahunan via portal HR minimal 14 hari sebelumnya. Berapa hari cuti tahunan?
12 hari per tahun kalender untuk karyawan tetap. Apakah cuti tahunan hangus?
Ya, sisa cuti tidak dibawa ke tahun berikutnya.""",
            """FAQ KANTOR. Jam kerja kantor adalah 09.00 sampai 17.00. Portal HR
dapat diakses dari luar kantor dengan VPN. Wi-Fi kantor memakai autentikasi
perangkat. Tiket bantuan IT diajukan lewat helpdesk internal.""",
        ],
    },
]

# --------------------------------------------------------------------------
# 2. Kueri uji + jawaban emas (dipakai evaluasi hit@k & faithfulness)
# --------------------------------------------------------------------------
KUERI_UJI = [
    {
        "id": "q1",
        "teks": "Berapa hari cuti tahunan untuk karyawan tetap?",
        "jawaban_emas": "12",
        "source_emas": "kebijakan-ketenagakerjaan.pdf",
        "gold_chunk_index": None,   # diisi _calib.py setelah chunking terkunci
        "gold_chunk_text": None,
    },
    {
        "id": "q2",
        "teks": "Berapa lama maksimal cuti sakit tanpa surat dokter?",
        "jawaban_emas": "5",
        "source_emas": "kebijakan-cuti-kesehatan.pdf",
        "gold_chunk_index": None,
        "gold_chunk_text": None,
    },
    {
        "id": "q3",
        "teks": "Bagaimana cara mengajukan cuti tahunan?",
        "jawaban_emas": "portal",
        "source_emas": "faq-karyawan.pdf",
        "gold_chunk_index": None,
        "gold_chunk_text": None,
    },
    {
        "id": "q4",
        "teks": "Berapa hari cuti melahirkan?",
        "jawaban_emas": "90",
        "source_emas": "kebijakan-cuti-kesehatan.pdf",
        "gold_chunk_index": None,
        "gold_chunk_text": None,
    },
    {
        "id": "q5",
        "teks": "Berapa hari cuti ayah untuk kelahiran anak?",
        "jawaban_emas": "3",
        "source_emas": "kebijakan-cuti-kesehatan.pdf",
        "gold_chunk_index": None,
        "gold_chunk_text": None,
    },
]

# Pertanyaan yang TIDAK dijawab korpus — harus dijawab "tidak ada di dokumen"
KUERI_LUAR_CORPUS = {
    "id": "q6",
    "teks": "Berapa harga tiket pesawat ke Bali?",
    "jawaban_emas": "tidak ada di dokumen",
}

# --------------------------------------------------------------------------
# 3. Latensi cache (ms) — deterministik untuk mengukur manfaat cache retrieval
# --------------------------------------------------------------------------
LATENSI_CACHE_MS = 2

# --------------------------------------------------------------------------
# 4. Tolok ukur pipeline RAG (tiap langkah dipanggil berkali-kali)
# --------------------------------------------------------------------------
TOLOK_UKUR = {
    "n_permintaan": 4,
    "aturan": [  # teks kueri per permintaan (index 0 & 1 identik: uji cache)
        "Berapa hari cuti tahunan untuk karyawan tetap?",
        "Berapa hari cuti tahunan untuk karyawan tetap?",
        "Berapa hari cuti tahunan untuk karyawan tetap?",
        "Berapa lama maksimal cuti sakit tanpa surat dokter?",
    ],
}

# --------------------------------------------------------------------------
# 5. Prompt RAG terkunci (dipakai generator & test)
# --------------------------------------------------------------------------
TEMPLATE_PROMPT_RAG = """Anda adalah asisten yang menjawab PERTANYAAN berdasarkan
KONTEKS yang diberikan. Aturan:
1. Jawab HANYA menggunakan informasi di KONTEKS.
2. Jika KONTEKS tidak memuat jawabannya, jawab persis: "tidak ada di dokumen".
3. Sebutkan sumber [n] untuk setiap klaim.
4. Jangan gunakan pengetahuan luar dan jangan mengarang angka.

KONTEKS:
{konteks}

PERTANYAAN:
{pertanyaan}"""

SEP_DETEKSI_TIDAK_ADA = "tidak ada di dokumen"
