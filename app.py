import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

from bps_client import BpsApiClient

# ── API Keys (Dikonfigurasi via Environment Variable pada Cloud) ─────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BPS_API_KEY = os.environ.get("BPS_API_KEY", "")

gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
bps = BpsApiClient(api_key=BPS_API_KEY) if BPS_API_KEY else None

# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if bps:
        await bps.close()

app = FastAPI(title="BPS AI Gateway Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Model ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str

# ── System Prompts ───────────────────────────────────────────────────

# Step 1: Intent Extraction — Gemini menganalisis pertanyaan user
INTENT_EXTRACTION_PROMPT = """
Anda adalah parser intent untuk BPS AI Assistant.
Dari pertanyaan user, extract informasi berikut ke JSON:

{
  "intent": "numeric | knowledge | publication | out_of_scope",
  "keywords": ["kata kunci untuk search BPS API, 1-3 keyword"],
  "data_source": "pressrelease | statictable | publication | none",
  "region": "nama wilayah jika disebutkan, default 'Indonesia'",
  "domain_id": "0000 untuk nasional, atau kode provinsi 2 digit + '00' jika diketahui",
  "needs_bps_data": true/false
}

Aturan:
- Jika pertanyaan tentang angka/indikator terbaru (inflasi, PDB, pengangguran, kemiskinan, IPM, ekspor, impor, penduduk, NTP): intent="numeric", data_source="pressrelease", needs_bps_data=true
- Jika pertanyaan tentang tabel data historis: intent="numeric", data_source="statictable", needs_bps_data=true
- Jika pertanyaan tentang publikasi BPS: intent="publication", data_source="publication", needs_bps_data=true
- Jika pertanyaan tentang definisi/metodologi statistik: intent="knowledge", needs_bps_data=false
- Jika di luar domain BPS: intent="out_of_scope", needs_bps_data=false
- keywords harus dalam Bahasa Indonesia, singkat dan relevan untuk search API

Output HANYA JSON murni, tanpa markdown.
"""

# Step 2: Final Response — Gemini menjawab dengan data real BPS
RESPONSE_WITH_DATA_PROMPT = """
Anda adalah BPS AI Assistant resmi (Badan Pusat Statistik Republik Indonesia).

Anda diberikan DATA RESMI dari BPS Web API berikut:
--- DATA BPS ---
{bps_data}
--- END DATA ---

Berdasarkan data di atas dan pertanyaan user, buat respons JSON:
{{
  "status": "success | clarify | out_of_scope | no_evidence",
  "intent": "numeric | knowledge | simple",
  "response_text": "Penjelasan naratif berdasarkan DATA RESMI di atas. Sebutkan angka dan sumber secara spesifik. Jika data BPS di atas tidak memuat jawaban lengkap, lengkapi dengan data resmi BPS dari basis pengetahuan Anda.",
  "data_payload": {{
     "indicator": "Nama indikator",
     "value": "Nilai angka lengkap dengan satuan",
     "region": "Wilayah",
     "period": "Periode data"
  }},
  "citations": [
    {{
      "title": "Judul BRS/Publikasi rujukan BPS",
      "url": "URL PDF jika tersedia, atau https://www.bps.go.id"
    }}
  ],
  "clarification_options": []
}}

Aturan:
- PRIORITASKAN data dari BPS API yang diberikan.
- Jika data BPS tidak memuat angka yang dicari, gunakan data proyeksi/sensus resmi BPS yang Anda ketahui.
- Jawab dalam Bahasa Indonesia yang jelas, profesional, dan mudah dipahami.
- Output WAJIB JSON murni.
"""

# Prompt untuk pertanyaan tanpa data BPS (knowledge/out_of_scope)
RESPONSE_WITHOUT_DATA_PROMPT = """
Anda adalah BPS AI Assistant resmi (Badan Pusat Statistik Republik Indonesia).
Prinsip utama Anda:
1. Menjawab pertanyaan seputar data statistik, istilah, metodologi, dan layanan BPS.
2. Output WAJIB berupa JSON murni:
{{
  "status": "success | clarify | out_of_scope | no_evidence",
  "intent": "numeric | knowledge | simple",
  "response_text": "Penjelasan naratif.",
  "data_payload": null,
  "citations": [
    {{
      "title": "Sumber BPS",
      "url": "https://www.bps.go.id"
    }}
  ],
  "clarification_options": []
}}

Aturan:
- Pertanyaan di luar domain BPS: status = "out_of_scope", tolak sopan.
- Pertanyaan ambigu: status = "clarify", isi clarification_options.
- Untuk pertanyaan angka tanpa data real, ingatkan bahwa data mungkin berupa estimasi/proyeksi resmi.
"""


def _generate_with_fallback(contents, system_instruction, response_mime_type="application/json", temperature=0.2):
    """
    Menjalankan inferensi dengan fallback berantai ke beberapa model
    untuk mencegah kendala 503 (server overloaded) saat live demo.
    """
    models = ['gemini-3.7-flash', 'gemini-3.5-flash', 'gemini-3.6-flash', 'gemini-flash-latest']
    last_error = None
    for model_name in models:
        try:
            return gemini.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type=response_mime_type,
                    temperature=temperature,
                ),
            )
        except Exception as e:
            last_error = e
            continue
    raise last_error


def _clean_and_parse_json(text: str) -> dict:
    """Membersihkan dan mem-parse string JSON dari LLM secara aman."""
    import re
    if not text:
        raise ValueError("Respons kosong dari model LLM")
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
        else:
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start_idx = cleaned.find("{")
    if start_idx != -1:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(cleaned[start_idx:])
        return obj

    raise ValueError(f"Tidak dapat mem-parse JSON dari respons: {cleaned[:100]}")


# ── Chat Endpoint ────────────────────────────────────────────────────

@app.post("/api/v1/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # ── STEP 1: Extract Intent ───────────────────────────────
        intent_response = _generate_with_fallback(
            contents=req.query,
            system_instruction=INTENT_EXTRACTION_PROMPT,
            temperature=0.1,
        )
        intent_data = _clean_and_parse_json(intent_response.text)

        needs_bps = intent_data.get("needs_bps_data", False)
        data_source = intent_data.get("data_source", "none")
        keywords = intent_data.get("keywords", [])
        domain_id = intent_data.get("domain_id", "0000")
        keyword_str = " ".join(keywords) if keywords else ""

        # ── STEP 2: Fetch BPS Data (jika diperlukan) ─────────────
        bps_data_text = ""
        if needs_bps and data_source != "none":
            bps_results = await _fetch_bps_data(data_source, keyword_str, domain_id)
            if bps_results:
                bps_data_text = json.dumps(bps_results, ensure_ascii=False, indent=2)

        # ── STEP 3: Generate Final Response ──────────────────────
        if bps_data_text:
            system_prompt = RESPONSE_WITH_DATA_PROMPT.format(bps_data=bps_data_text)
        else:
            system_prompt = RESPONSE_WITHOUT_DATA_PROMPT

        final_response = _generate_with_fallback(
            contents=req.query,
            system_instruction=system_prompt,
            temperature=0.2,
        )
        return _clean_and_parse_json(final_response.text)

    except Exception as e:
        return {
            "status": "error",
            "intent": "simple",
            "response_text": f"Terjadi kendala pada gateway LLM: {str(e)}",
            "data_payload": None,
            "citations": [],
            "clarification_options": []
        }


async def _fetch_bps_data(source: str, keyword: str, domain: str) -> list[dict]:
    """Fetch data dari BPS Web API berdasarkan source type."""
    try:
        if source == "pressrelease":
            return await bps.search_pressrelease(keyword=keyword, domain=domain)
        elif source == "statictable":
            return await bps.search_statictable(keyword=keyword, domain=domain)
        elif source == "publication":
            return await bps.search_publication(keyword=keyword, domain=domain)
        else:
            return []
    except Exception:
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
