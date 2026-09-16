# Bab 11 — Bekerja dengan LLM API & Orkestrasi

> Dari "prompt di playground" ke "kode produksi yang tangguh": API, streaming, structured output, fallback, dan biaya.

## 🎯 Tujuan Belajar
- Memakai SDK LLM dengan benar: system/user, streaming, tool calling.
- Structured output (JSON) yang tahan gagal.
- Pola produksi: retry, fallback antar-provider, caching, rate limit.
- Mengelola biaya & latensi.

## 1. Materi Inti

> Dari "prompt di playground" ke "kode produksi yang tangguh": API, streaming, structured output, fallback, dan biaya.

---

### 1.1 Memanggil LLM — Dasar yang Wajib Paham

```python
import os
from openai import OpenAI

# ⚠️ PENTING: API key dari environment variable, JANGAN hardcode!
# Kalau nggak tahu caranya:
#   export OPENAI_API_KEY="sk-..."
#   atau pakai .env + python-dotenv

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # base_url opsional: kalau pakai proxy atau provider alternatif
)

# Basic call
response = client.chat.completions.create(
    model="gpt-4o-mini",       # pilih model sesuai kebutuhan & budget
    messages=[
        {"role": "system", "content": "Kamu adalah asisten yang membantu. Jawab ringkas dan akurat."},
        {"role": "user", "content": " Jelaskan apa itu machine learning dalam 2 kalimat."},
    ],
    temperature=0.2,           # 0.0 = deterministik, tinggi = lebih kreatif
    max_tokens=500,            # batasi output agar tidak boros
)

print(response.choices[0].message.content)

# Simpan metadata untuk logging/monitoring
print(f"\nMetadata:")
print(f"  Model: {response.model}")
print(f"  Usage: {response.usage}")
print(f"  Id: {response.id}")
```

#### Semua Provider Pakai Pola yang Sama

| Provider | SDK | Model contoh | Keterangan |
|---|---|---|---|
| OpenAI | `openai` | gpt-4o, gpt-4o-mini, o1 | Standar industri; dokumentasi paling lengkap |
| Anthropic | `anthropic` | claude-3-opus, claude-3-5-sonnet | Fokus pada AI safety; output cenderung lebih terstruktur |
| Google | `google-genai` | gemini-1.5-pro, gemini-2.0-flash | Multimodal native, context window besar (1M+ token) |
| DeepSeek | `deepseek` | deepseek-chat, deepseek-reasoner | Cost-effective, performa kompetitif |

**Provider-agnostic:** [LiteLLM](https://docs.litellm.ai/) — satu interface untuk semua provider.

---

### 1.2 Streaming — UX yang Lebih Baik

```python
# Tanpa streaming: user nunggu sampai output lengkap baru muncul
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tulis artikel tentang AI"}],
    max_tokens=2000,
)
print(response.choices[0].message.content)  # Muncul sekaligus

# Dengan streaming: token muncul bertahap → UX lebih responsif
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tulis artikel tentang AI"}],
    max_tokens=2000,
    stream=True,          # ← streaming ON
)

print("Streaming output:")
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # newline di akhir
```

**Kapan pakai streaming:**

- Chat interface / CLI chatbot → harusnya streaming!
- Output panjang (artikel, kode, analisis) → user tidak perlu nunggu 10 detik
- Task singkat (< 100 token) → nggak perlu streaming, malah lebih rumit

---

### 1.3 Structured Output (JSON) — Wajib untuk Production

```python
from pydantic import BaseModel, Field
from typing import Optional

# Definisi schema output
class SentimentResult(BaseModel):
    label: str = Field(..., description="Sentimen: positive/negative/neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Keyakinan klasifikasi")
    reasons: list[str] = Field(default_factory=list, description="Alasan di balik sentimen")
    language: str = Field(default="id", description="Bahasa teks")

# Prompt dengan instruksi JSON schema
prompt = f"""
 analiz themes sentiment dari teks berikut.
Return output dalam format JSON yang sesuai dengan schema berikut:

Output Schema:
- label: string, values = [positive, negative, neutral]
- confidence: float 0.0-1.0
- reasons: array of strings, alasan utama
- language: string, bahasa teks

Teks yang harus dianalisis:
{text}

Format output: JSON object sesuai schema di atas.
"""

# Kalau pakai OpenAI, bisa panggil dengan response_format
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},  # ← minta JSON
    temperature=0.0,  # deterministic untuk structured output
)

result_json = response.choices[0].message.content

# Validasi dengan pydantic
try:
    result = SentimentResult.model_validate_json(result_json)
    print(f"Validasi berhasil!")
    print(f"  Label: {result.label}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Reasons: {result.reasons}")
except Exception as e:
    print(f"Output tidak valid: {e}")
    # Retry dengan pesan error yang informatif
    retry_prompt = f"""
    Output sebelumnya tidak valid: {e}
    
    Teks asli: {text}
    
    Silakan coba lagi dengan format JSON yang benar.
    """
```

> **Self-healing loop:** Kalau output tidak valid, kirim pesan error ke model + minta perbaiki. Bisa loop beberapa kali sebelum gagal total.

---

### 1.4 Tool/Function Calling

> Dasar dari agent (Bab 14): model bisa memilih panggil fungsi, kode yang mengeksekusi.

```python
# Definisi function/tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Dapatkan cuaca untuk kota tertentu",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Nama kota, contoh: 'Jakarta', 'Surabaya'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Satuan suhu"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Melakukan kalkulasi matematika",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Ekspresi matematika, contoh: '2+3*4'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Pesan ke model
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Berapa suhu Jakarta hari ini?"}],
    tools=tools,
    tool_choice="auto",  # model pilih apakah perlu call tool
)

# Handle response
message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        print(f"Model mau memanggil: {function_name}")
        print(f"Arguments: {arguments}")
        
        # Eksekusi fungsi (di sini dummy)
        if function_name == "get_weather":
            result = {"temperature": 28, "condition": "cerah", "unit": arguments.get("unit", "celsius")}
        elif function_name == "calculate":
            result = {"result": eval(arguments["expression"])}
        
        # Kirim kembali hasil ke model
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Berapa suhu Jakarta hari ini?"},
                message,  # assistant message dengan tool call
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            ],
        )
        print(f"\nFinal response: {second_response.choices[0].message.content}")
else:
    print(f"Model menjawab langsung: {message.content}")
```

---

### 1.5 Pola Produksi Wajib

#### Retry + Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APIError,
    )),
    before_sleep=lambda retry_state: print(f"Retry {retry_state.attempt_number}...")
)
def call_llm_with_retry(**kwargs):
    return client.chat.completions.create(**kwargs)

# Pakai fungsinya
response = call_llm_with_retry(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Halo"}],
)
```

#### Timeout & Budget

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tulis artikel panjang tentang AI"}],
    max_tokens=1000,      # ← batasi output
    timeout=30,           # ← timeout per request
)

# Log & budget tracking
import logging
logger = logging.getLogger(__name__)

logger.info(
    "LLM request completed",
    extra={
        "model": response.model,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency_ms": response.usage.completion_tokens_details?,  # tergantung SDK
        "cost_estimate": estimate_cost(response.usage),
    }
)
```

#### Fallback Chain

```python
 FallbackChain:
  Model utama (gpt-4o)         → kalau gagal/error → lanjut
  Model cadangan (gpt-4o-mini) → kalau gagal/error → lanjut
  Model fallback (claude-3-haiku) → kalau gagal/error → return error
```

```python
FALLBACK_MODELS = [
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "openai", "model": "gpt-4o-mini"},
    {"provider": "anthropic", "model": "claude-3-haiku"},
]

def call_with_fallback(messages, max_tokens=500):
    last_error = None
    
    for i, config in enumerate(FALLBACK_MODELS):
        try:
            print(f"Trying model {i+1}/{len(FALLBACK_MODELS)}: {config['model']}")
            response = call_llm(FALLBACK_MODELS[i], messages, max_tokens)
            return response
        except Exception as e:
            last_error = e
            logger.warning(f"Model {config['model']} gagal: {e}")
            continue
    
    raise Exception(f"Semua model gagal: {last_error}")
```

#### Caching

```python
import functools
import hashlib
import json

# Cache sederhana dengan hash dari prompt + paramet
@functools.lru_cache(maxsize=1000)
def cached_llm_call(model: str, prompt_hash: str, temperature: float):
    """Wrapper yang cache response berdasarkan hash dari prompt."""
    # Implementasi: simpan ke dict atau Redis
    pass

def hash_prompt(prompt: str, **params) -> str:
    """Buat hash dari prompt + parameter."""
    content = json.dumps({"prompt": prompt, "params": params}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]

# Usage
prompt_hash = hash_prompt(" Jelaskan AI", temperature=0.2, model="gpt-4o-mini")
# Kalau hash sudah ada di cache → return cached response
# Kalau belum → call API, simpan ke cache
```

---

### 1.6 Biaya & Latensi

#### Hitung Biaya

```python
def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> dict:
    """Estimasi biaya API LLM."""
    pricing = {
        "gpt-4o-mini":  {"input": 0.150, "output": 0.600},   # per 1M tokens
        "gpt-4o":      {"input": 5.000, "output": 15.000},
        "claude-3-opus": {"input": 15.000, "output": 75.000},
        "gemini-1.5-pro": {"input": 1.250, "output": 5.000},
        "deepseek-chat": {"input": 0.140, "output": 0.280},
    }
    
    rates = pricing.get(model, pricing["gpt-4o-mini"])
    
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_usd": round(input_cost + output_cost, 6),
    }

# Contoh
usage = {"input_tokens": 1500, "output_tokens": 300}
cost = estimate_cost(**usage, model="gpt-4o-mini")
print(f"Estimasi biaya per request: ${cost['total_usd']:.6f}")

# Estimasi bulanan
requests_per_day = 1000
avg_cost_per_request = cost["total_usd"]
monthly_estimate = avg_cost_per_request * requests_per_day * 30
print(f"Estimasi biaya bulanan: ${monthly_estimate:.2f}")
```

#### 3 Cara Mengecilkan Biaya

| Cara | Potensi Penghematan | Keterangan |
|---|---|---|
| **1. Gunakan model yang tepat** | 5-50x | Task sederhana → gpt-4o-mini; task kompleks → gpt-4o. Jangan pakai model besar untuk tugas mudah. |
| **2. Kurangi token input** | 2-10x | RAG dengan retrieval tepat (Bab 12), prompt yang lebih ringkas, cache untuk prompt berulang |
| **3. Kurangi token output** | 2-5x | Batasi max_tokens, minta output yang lebih padat, streaming untuk UX tanpa perlu output panjang |
| **4. Cache response** | 10-100x untuk query berulang | Cache FP untuk prompt identik; semantic cache untuk near-duplicate (Bab 11 - embedding similarity) |
| **5. Route ke model yang tepat** | 3-20x | Classifier untuk routing: task mudah → model kecil; task sulit → model besar |

---

### 1.7 Ringkasan Cepat (1 Halaman)

```
🎯 Fokus Bab 11:
  1. API call dasar: client.chat.completions.create()
  2. Semua provider pakai pola mirip;LiteLLM = provider-agnostic
  3. Streaming: token bertahap → UX lebih baik untuk output panjang
  4. Structured output: JSON schema + pydantic validation + retry
  5. Tool calling: model pilih function, kode eksekusi → dasar agent (Bab 14)
  6. Produksi pattern: retry+backoff, timeout, fallback chain, cache
  7. Biaya: hitung per request, estimasi bulanan, 5 cara menekan
```

---

## 2. Latihan Praktis

---

### Latihan 1: Summarizer API dengan Production Pattern

**Tujuan:** Bangun API yang robust dengan retry, timeout, structured output.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIConnectionError

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Summarizer API")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Schema
class SummarizerRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000, description="Teks yang akan dirangkum")
    max_length: int = Field(default=150, ge=50, le=500, description="Panjang maksimal ringkasan")
    language: str = Field(default="id", pattern="^[a-z]{2}$", description="Bahasa output (2 huruf)")
    style: str = Field(default="neutral", pattern="^(neutral|formal|casual)$") 

class SummarizerResponse(BaseModel):
    summary: str = Field(..., min_length=20)
    original_length: int
    summary_length: int
    compression_ratio: float
    model_used: str
    latency_ms: float
    cost_estimate_usd: float

# LLM call dengan retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
def call_llm(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    return response.choices[0].message.content

def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    pricing = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}}
    rates = pricing.get(model, pricing["gpt-4o-mini"])
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]

# Endpoint
@app.post("/summarize", response_model=SummarizerResponse)
async def summarize(req: SummarizerRequest):
    start_time = time.time()
    
    try:
        # Build prompt
        prompt = f"""
Ringkas teks berikut dalam {req.max_length} kata.
Gunakan bahasa {req.language} dengan gaya {req.style}.

Teks:
{req.text}

Output dalam JSON:
{{
  "summary": "ringkasan teks",
  "original_length": {len(req.text)},
  "summary_length": ...,
  "compression_ratio": ...,
  "model_used": "gpt-4o-mini"
}}
"""
        
        # Call LLM
        result_json = call_llm(prompt)
        
        # Parse & validate
        import json
        result = json.loads(result_json)
        
        # Hitung metric
        latency_ms = (time.time() - start_time) * 1000
        
        # Estimasi biaya (asumsi input ~len(text)/4 token, output ~len(summary)/4 token)
        input_tokens = len(req.text) // 4
        output_tokens = len(result["summary"]) // 4
        cost = estimate_cost(input_tokens, output_tokens)
        
        return SummarizerResponse(
            summary=result["summary"],
            original_length=result.get("original_length", len(req.text)),
            summary_length=result.get("summary_length", len(result["summary"])),
            compression_ratio=result.get("compression_ratio", len(result["summary"]) / len(req.text)),
            model_used=result.get("model_used", "gpt-4o-mini"),
            latency_ms=round(latency_ms, 2),
            cost_estimate_usd=round(cost, 6)
        )
    
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to summarize: {str(e)}")

# Test dengan curl:
# curl -X POST "http://localhost:8000/summarize" -H "Content-Type: application/json" -d '{"text": "..."}'
```

---

### Latihan 2: Streaming Chat CLI

**Tujuan:** Bangun chatbot CLI sederhana dengan streaming.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class ChatCLI:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.messages = []
        self.max_history = 20  # max pesan yang disimpan
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        # Potong history kalau terlalu panjang
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def chat(self):
        print("Chatbot CLI (ketik 'quit' untuk keluar)")
        print("="*50)
        
        while True:
            user_input = input("\nAnda: ").strip()
            
            if user_input.lower() in ["quit", "exit", "keluar"]:
                print("¡ Sampai jumpa!")
                break
            
            if not user_input:
                continue
            
            # Tambah pesan user
            self.add_message("user", user_input)
            
            # Print assistant response dengan streaming
            print("Assistant: ", end="", flush=True)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True,
                temperature=0.7,
                max_tokens=1000,
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # newline
            
            # Simpan response
            self.add_message("assistant", full_response)

# Jalankan
if __name__ == "__main__":
    cli = ChatCLI()
    cli.add_message("system", "Kamu adalah asisten yang membantu. Jawab dalam bahasa Indonesia yang santai dan akurat.")
    cli.chat()
```

---

### Latihan 3: Tool Calling dengan Validasi

**Tujuan:** Implementasi tool calling yang aman dengan validasi parameter.

```python
import json
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

# Definisi tools dengan schema pydantic
class GetWeatherInput(BaseModel):
    city: str = Field(..., min_length=2, max_length=50, description="Nama kota")
    unit: str = Field(default="celsius", pattern="^(celsius|fahrenheit)$")

class CalculateInput(BaseModel):
    expression: str = Field(..., min_length=1, max_length=100, description="Ekspresi matematika")

class SearchNotesInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=200, description="Kata kunci pencarian")
    limit: int = Field(default=5, ge=1, le=20)

TOOLS = {
    "get_weather": {
        "function": GetWeatherInput,
        "handler": lambda args: {"temperature": 28, "condition": "cerah", "city": args["city"], "unit": args["unit"]},
        "description": "Dapatkan informasi cuaca untuk kota tertentu"
    },
    "calculate": {
        "function": CalculateInput,
        "handler": lambda args: {"result": eval(args["expression"])},
        "description": "Melakukan kalkulasi matematika"
    },
    "search_notes": {
        "function": SearchNotesInput,
        "handler": lambda args: {"results": ["Catatan 1 tentang " + args["query"], "Catatan 2 tentang " + args["query"]]},
        "description": "Mencari catatan yang relevan dengan query"
    }
}

class ToolCallingAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.tools = TOOLS
        self.max_iterations = 5  # mencegah loop tak berujung
    
    def get_tool_schemas(self):
        """Convert tools ke format yang dimengerti LLM API."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["function"].model_json_schema()
                }
            }
            for name, info in self.tools.items()
        ]
    
    def run(self, user_message: str):
        """Jalankan agent loop: reason → act → observe."""
        messages = [{"role": "user", "content": user_message}]
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iterasi {iteration + 1} ---")
            
            # Call LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.get_tool_schemas(),
                tool_choice="auto",
                temperature=0.0,
            )
            
            message = response.choices[0].message
            
            # Kalau model bilang jawab langsung
            if message.content and not message.tool_calls:
                print(f"Assistant: {message.content}")
                return message.content
            
            # Kalau model mau panggil tool
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments_str = tool_call.function.arguments
                    
                    print(f"Tool call: {function_name}({arguments_str})")
                    
                    # Validasi arguments dengan pydantic
                    try:
                        args = self.tools[function_name]["function"].model_validate_json(arguments_str)
                        print(f"  Validated args: {args}")
                    except ValidationError as e:
                        print(f"  ❌ Invalid arguments: {e}")
                        # Kirim error ke model
                        messages.append(message)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": str(e)})
                        })
                        continue
                    
                    # Eksekusi tool
                    result = self.tools[function_name]["handler"](args.dict())
                    print(f"  Result: {result}")
                    
                    # Kirim result ke model
                    messages.append(message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                print("Tidak ada tool calls atau content")
                break
        
        return "Maaf, saya tidak bisa membantu dengan permintaan tersebut."

# Test
agent = ToolCallingAgent()

print("="*60)
print("TOOL CALLING AGENT DEMO")
print("="*60)

# Test 1: Weather
print("\n>> Test 1: Cuaca")
result = agent.run("Berapa suhu Jakarta hari ini?")

# Test 2: Kalkulasi
print("\n>> Test 2: Kalkulasi")
result = agent.run("Hitung 15 * 8 + 20")

# Test 3: Multi-step
print("\n>> Test 3: Multi-step")
result = agent.run("Cari catatan tentang machine learning, lalu hitung rata-rata dari 5 poin")
```

---

## 📚 Referensi Online

| Sumber | Tipe | Keterangan |
|---|---|---|
| [OpenAI API Reference](https://platform.openai.com/docs/api-reference) | Dokumentasi | Referensi lengkap API OpenAI |
| [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) | Dokumentasi | Referensi API Anthropic |
| [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs) | Dokumentasi | Referensi API Gemini |
| [LiteLLM](https://docs.litellm.ai/) | Library | Provider-agnostic interface |
| [OpenAI Cookbook](https://cookbook.openai.com/) | Contoh | Contoh produksi dari OpenAI |
| [Tenacity documentation](https://tenacity.readthedocs.io/) | Library | Retry pattern Python |

---

## ✅ Checklist Kompetensi

- [ ] API key via env var, tidak pernah di-commit
- [ ] Output JSON dijamin valid (schema + retry)
- [ ] Tahu biaya per request sistemmu dan 3 cara menekannya
- [ ] Implementasi streaming di aplikasi
- [ ] Buat fallback chain untuk multi-provider
- [ ] Handle tool calling dengan validasi parameter
- [ ] Setup logging & monitoring untuk LLM requests
