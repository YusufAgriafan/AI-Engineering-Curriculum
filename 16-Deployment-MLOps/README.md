# Bab 16 — Deployment & MLOps untuk AI Engineering

> Model di laptop bukan produk. Bab ini membawa sistem AI Anda ke pengguna: API, cloud, monitoring, dan biaya yang terkendali.

## 🎯 Tujuan Belajar
- Menyajikan sistem AI sebagai REST API (FastAPI) + streaming.
- Deployment ke cloud; memahami opsi modern (serverless LLM APIs vs self-host GPU).
- CI/CD, Docker, secret management.
- Monitoring, drift, dan kontrol biaya produksi.

## 1. Materi Inti

### 1.1 FastAPI + Streaming
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat(req: ChatRequest):
    # stream token dari LLM (Bab 11) ke SSE
    return StreamingResponse(stream_llm(req.messages), media_type="text/event-stream")
```
- Praktik: pydantic untuk validasi (Bab 11), rate limit per user, timeout, graceful degradation (fallback model murah saat kuota habis).

### 1.2 Docker & CI/CD
- Dockerfile multi-stage; `.env` tidak masuk image — pakai secret manager (cloud) atau GitHub Secrets.
- CI: lint → unit test (Bab 1) → eval gate (Bab 15) → build image → deploy staging → smoke test → promote.

### 1.3 Opsi Infrastruktur
| Pola | Kapan | Contoh |
|---|---|---|
| Panggil LLM API dari backend kecil | Mayoritas kasus | Vercel/Railway/Cloud Run + OpenAI/Anthropic/Gemini |
| Serverless inference provider | Open-weight tanpa urus GPU | Together, Fireworks, Groq (latensi rendah) |
| Self-host GPU | Skala besar / privasi ketat | vLLM (Bab 13) di GKE/EKS |
| Semua di laptop/edge | Offline/prototipe | Ollama |

- Aturan: **mulai dari API provider** — self-host GPU hanya jika biaya/privasi/latensi menuntut.

### 1.4 Monitoring & Drift
- Metrik teknis: error rate, p50/p95 latency, token, biaya/hari per fitur.
- Metrik kualitas: sampling review mingguan, thumbs feedback, guardrail trigger rate.
- Drift: distribusi pertanyaan user berubah? volume bahasa campur naik? → perbarui evalset (Bab 15) & prompt.
- Untuk model klasik (Bab 3) yang masih Anda jalankan: data drift & retraining schedule adalah tanggung jawab MLOps klasik (bisa pakai MLflow/Evidently).

### 1.5 Kontrol Biaya
- Budget alert per fitur; cache (Bab 11); routing: pertanyaan mudah → model kecil, sulit → model besar (classifier router — pakai skill Bab 3!).
- Turunkan token: prompt ringkas, RAG dengan top-k pas, riwayat diringkas.

## 2. Latihan
1. Bungkus RAG Bab 12 jadi API Dockerized: `/ingest`, `/chat` (streaming), `/health`.
2. Deploy ke Cloud Run/Railway/Fly.io + GitHub Actions: test → build → deploy → smoke test otomatis.
3. Tambahkan Langfuse SDK + dashboard biaya harian; buat alert biaya sederhana.
4. Implementasi router 2-tier: model kecil untuk intent mudah (dicek eval), escalate ke model besar.

## 📚 Referensi Online
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Google Cloud Run + LLM tutorial](https://cloud.google.com/run/docs/tutorials)
- [vLLM production deployment](https://docs.vllm.ai/en/latest/deployment/index.html)
- [MLflow](https://mlflow.org/docs/latest/index.html) & [Evidently](https://docs.evidentlyai.com/) (drift & monitoring klasik)
- [12-Factor App](https://12factor.net/id/) (prinsip dasar aplikasi cloud)

## ✅ Checklist Kompetensi
- [ ] Sistem AI berjalan di cloud dengan CI/CD otomatis
- [ ] Monitoring: biaya, latensi, error per fitur — bukan tebakan
- [ ] Punya strategi pengendalian biaya yang terukur
