"""TODO Bagian 6 — End-to-end + laporan operasional: menyatukan semua lapisan.

serve() = pipeline satu permintaan persis seperti handler FastAPI nyata:
validasi (Bagian 2) → rate limit (Bagian 2) → router (Bagian 3) → jawab.
laporan_ops() = health + drift + biaya + breakeven dalam satu dict untuk dashboard.
"""
from .data import HARGA, PROFIL_TOKEN
from .konfig import config_layanan
from .monitor import deteksi_drift, health_score


def serve(body, config, waktu=None, limiter=None):
    """Satu permintaan /chat → (status, hasil).

    Alur: validasi_chat (422) → rate limit (429, {'error': 'rate limit
    tercapai'}) → router jawab (200). waktu None → detik 0.
    limiter None → RateLimiter(config['RATE_LIMIT_PER_MENIT']) baru.
    Klien tunggal: 'default'."""
    # TODO
    raise NotImplementedError


def laporan_ops(config=None):
    """Dashboard operasional harian:
    {'health': health_score(), 'drift': deteksi_drift(),
     'breakeven_req_per_hari': breakeven untuk config['MODEL_API'],
     'biaya_harian': {model: biaya_per_req × 2000 req untuk tiap model di HARGA}}
    config None → config_layanan(None). Biaya per req memakai PROFIL_TOKEN."""
    # TODO
    raise NotImplementedError
