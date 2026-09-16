# Bab 14 — Agentic AI

> Agent = LLM yang diberi tujuan, alat, dan kemandekan siklus *berpikir→bertindak→mengamati*. Bab ini menyatukan tool calling (Bab 11), prompt ReAct (Bab 10), dan retrieval (Bab 12).

## 🎯 Tujuan Belajar
- Pola desain agent: ReAct, planner-executor, multi-agent.
- Tool/function calling yang aman dan tervalidasi.
- Memory: riwayat, ringkasan, state.
- Framework: LangGraph, CrewAI, OpenAI Agents SDK.
- Mengetahui kapan agent **tidak** dibutuhkan (penting!).

## 1. Materi Inti

### 1.1 Apa itu Agent (dan apa bukan)
- **Bukan agent**: satu panggilan LLM (chat, summarizer).
- **Agent**: loop `reason → act (tool) → observe` sampai tujuan tercapai — mirip loop RL dari Bab 3 (`State-action value function example.ipynb`), tapi kebijakannya ditulis bahasa natural, bukan Q-table.
- Komponen: model, **tools**, **memory**, **planning/policy**, **henti (termination)**.

### 1.2 ReAct & Tool Use
```
Thought : Saya perlu harga produk X hari ini.
Action  : get_price("X")
Observe : Rp 25.000
Thought : Cukup. Jawab user.
```
- Implementasi: tool calling (Bab 11) dengan **pydantic schema** + validasi + timeout + max-iteration (wajib, agar tidak loop tak berujung).
- Tools umum: pencarian web, SQL executor, API internal, code interpreter, RAG-retriever.

### 1.3 Memory
- **Short-term**: riwayat chat → potong/ringkas saat mendekati context window.
- **Long-term**: fakta user disimpan di DB (biasanya vektor + metadata → Bab 12).
- Episodic vs semantic memory: transkrip sesi vs fakta yang diekstrak.

### 1.4 Pola Multi-Agent
- **Planner–executor**: satu agent memecah tugas, agent lain mengeksekusi.
- **Spesialis**: agent penulis → agent kritikus → agent editor.
- **Human-in-the-loop**: aksi berisiko (transfer uang, kirim email) harus minta konfirmasi.
- Aturan: mulai **satu agent + tools yang bagus**; multi-agent hanya jika benar-benar perlu (biaya & debug melipatgandakan).

### 1.5 Framework
| Framework | Cocok untuk |
|---|---|
| **LangGraph** | Workflow graph terkontrol, state eksplisit — pilihan produksi |
| OpenAI Agents SDK | Ekosistem OpenAI, handoffs |
| CrewAI | Multi-agent cepat prototipe |
| Murni kode (loop sendiri) | Belajar & kontrol penuh ← mulai dari sini |

### 1.6 Keamanan & Kendali
- **Prompt injection via tool output** (halaman web bisa menyuruh agent): sanitasi, whitelist domain, sandbox.
- Least privilege: API key tool terpisah & terbatas; log semua aksi; budget & rate limit per agent.
- Idempotency: aksi yang sama tidak boleh dobel-eksekusi saat retry.

## 2. Latihan
1. Agent CLI dari nol (tanpa framework): loop ReAct dengan 2 tools (`get_time`, `search_notes`) + max-iterations.
2. Tambahkan tool SQL ke DB sample; pastikan query di-sanitize & hasil dibatasi.
3. Bangun ulang agent #1 dengan LangGraph; bandingkan kontrol & keterbacaan kode.
4. Buat skenario injection di tool output; buktikan proteksimu menahannya.

## 📚 Referensi Online
- [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents) (terbaik untuk pola desain)
- [ReAct paper](https://arxiv.org/abs/2210.03629)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [OpenAI — A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- [Hugging Face Agents Course (gratis)](https://huggingface.co/learn/agents-course)

## ✅ Checklist Kompetensi
- [ ] Membangun agent ReAct dari nol dan menjelaskan tiap komponennya
- [ ] Tools tervalidasi (schema, timeout, max-iterations, least privilege)
- [ ] Tahu kapan memakai agent vs satu panggilan LLM biasa
