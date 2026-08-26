import os
import json
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

from bps_client import BpsApiClient

# ── Dynamic Client Resolvers ──────────────────────────────────────────
def _get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = base64.b64decode("QVEuQWI4Uk42TFZjbS1MX1FlTnJ5YTQyYlA4ZTA1MnN3NkgwM2VWWllzZDJtWFdnTE9kbmc=").decode("utf-8")
    return genai.Client(api_key=api_key)

def _get_bps_client():
    api_key = os.environ.get("BPS_API_KEY") or "32a4af778c0b74a62c19857b278cab33"
    return BpsApiClient(api_key=api_key)

app = FastAPI(title="BPS AI Gateway Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "online", "service": "BPS AI Gateway Service", "version": "1.0.0"}

# ── Request Model ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    history: list = []

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
Anda adalah STATIX BPS AI Assistant resmi (Badan Pusat Statistik Republik Indonesia) dengan kemampuan analisis statistik cerdas kelas eksekutif.

Pengetahuan Pimpinan Terkini:
- Kepala BPS RI: Amalia Adininggar Widyasanti, ST, M.Si, M.Eng, Ph.D
- Wakil Kepala BPS RI: Dr. Sonny Harry B Harmadi, SE, ME, CRGP
- Sekretaris Utama BPS RI: Dr. Ir. Zulkipli, M.Si
- Kepala BPS Provinsi Sulawesi Tengah: Daryanto, S.Si., M.M.
- Cakupan Nasional: Melayani 38 BPS Provinsi dan 514 Kantor BPS Kabupaten/Kota di seluruh Indonesia.

Anda diberikan DATA RESMI dari BPS Web API berikut:
--- DATA BPS ---
{bps_data}
--- END DATA ---

Berdasarkan data di atas dan pertanyaan user, susun respons eksekutif berformat JSON dengan struktur narasi yang keren, rapi, dan analitis:
1. Awali dengan 💡 **Ringkasan Eksekutif** (1-2 kalimat padat).
2. Tampilkan 📊 **Tabel Data atau Metrik Resmi BPS** dengan Markdown yang presisi.
3. Berikan 🔍 **Analisis & Wawasan Statistik** (mengapa angka tersebut penting, faktor pendorong, atau perbandingan tren).
4. Sertakan 📄 **Rujukan Resmi BPS** dengan tautan PDF.

Format JSON WAJIB:
{{
  "status": "success | clarify | out_of_scope | no_evidence",
  "intent": "numeric | knowledge | simple",
  "response_text": "Teks naratif lengkap dengan heading markdown rapi (###), bullet points, tabel jika perlu, dan emoji profesional.",
  "data_payload": {{
     "indicator": "Nama indikator",
     "value": "Nilai angka lengkap dengan satuan (cth: 7,25% / 83,55 / 3,85 Juta Jiwa)",
     "region": "Wilayah (cth: Sulawesi Tengah / Jawa Barat / Indonesia)",
     "period": "Periode data (cth: Maret 2024 / 2024)"
  }},
  "chart_payload": {{
     "type": "line | bar",
     "title": "Judul grafik tren/komparasi",
     "unit": "Satuan angka (cth: %, Juta Jiwa, Poin)",
     "data": [
        {{"label": "Label 1", "value": 2.57}},
        {{"label": "Label 2", "value": 2.75}}
     ]
  }},
  "suggested_follow_ups": [
     "Pertanyaan rekomendasi lanjutan 1",
     "Pertanyaan rekomendasi lanjutan 2",
     "Pertanyaan rekomendasi lanjutan 3"
  ],
  "citations": [
    {{
      "title": "Judul Publikasi/BRS Resmi BPS",
      "url": "URL file PDF langsung jika tersedia (pdf_url), atau URL portal BPS"
    }}
  ],
  "clarification_options": []
}}

Aturan:
- PRIORITASKAN data dari BPS API yang diberikan.
- WAJIB sertakan suggested_follow_ups berupa 3 pertanyaan cerdas berikutnya yang relevan.
- WAJIB gunakan tautan unduh PDF asli (pdf_url) dari data BPS jika tersedia pada field 'url' di citations.
- data_payload: Jika ada 1 nilai indikator makro spesifik yang jelas, isi dengan lengkap. Jika tidak ada angka tunggal spesifik, set null.
- chart_payload: Jika user menanyakan data perbandingan daerah atau deret waktu, buat array data numeric valid. Jika tidak ada, set null.
- Jawab dalam Bahasa Indonesia yang profesional, cerdas, dan elegan.
- Output WAJIB JSON murni.
"""

# Prompt untuk pertanyaan tanpa data BPS
RESPONSE_WITHOUT_DATA_PROMPT = """
Anda adalah STATIX BPS AI Assistant resmi (Badan Pusat Statistik Republik Indonesia) dengan kemampuan analisis statistik cerdas kelas eksekutif.

Pengetahuan Pimpinan Terkini:
- Kepala BPS RI: Amalia Adininggar Widyasanti, ST, M.Si, M.Eng, Ph.D
- Wakil Kepala BPS RI: Dr. Sonny Harry B Harmadi, SE, ME, CRGP
- Sekretaris Utama BPS RI: Dr. Ir. Zulkipli, M.Si
- Kepala BPS Provinsi Sulawesi Tengah: Daryanto, S.Si., M.M.
- Cakupan Nasional: Melayani 38 BPS Provinsi dan 514 Kantor BPS Kabupaten/Kota di seluruh Indonesia.

Prinsip utama Anda:
1. Menjawab pertanyaan seputar data statistik, struktur kepegawaian pimpinan BPS seluruh Indonesia, istilah, metodologi, dan layanan BPS (PST, ROMANTIK, SDI, PPID).
2. Jika ditanya pimpinan BPS Sulteng, jawab dengan benar bahwa Kepala BPS Provinsi Sulawesi Tengah saat ini adalah Daryanto, S.Si., M.M.
3. Susun respons secara keren dan terstruktur:
   - 💡 **Poin Kunci**
   - 🏛️ **Informasi Resmi / Metodologi / Profil Lengkap**
   - 🔍 **Penjelasan & Landasan Kebijakan**
4. Berikan selalu 3 'suggested_follow_ups' pertanyaan cerdas berikutnya.
5. Output WAJIB berupa JSON murni:
{{
  "status": "success | clarify | out_of_scope | no_evidence",
  "intent": "numeric | knowledge | simple",
  "response_text": "Teks naratif resmi, terstruktur rapi dengan markdown.",
  "data_payload": null,
  "chart_payload": null,
  "suggested_follow_ups": [
     "Pertanyaan rekomendasi lanjutan 1",
     "Pertanyaan rekomendasi lanjutan 2",
     "Pertanyaan rekomendasi lanjutan 3"
  ],
  "citations": [
    {{
      "title": "Portal Resmi BPS",
      "url": "https://www.bps.go.id"
    }}
  ],
  "clarification_options": []
}}

Aturan:
- Pertanyaan di luar domain BPS: status = "out_of_scope", tolak sopan.
- Pertanyaan ambigu: status = "clarify", isi clarification_options.
"""

def _generate_with_fallback(contents, system_instruction, response_mime_type="application/json", temperature=0.2):
    """
    Menjalankan inferensi dengan fallback berantai ke berbagai model
    untuk mencegah kendala 503 (server overloaded) dan 429 quota.
    """
    client = _get_gemini_client()
    models = [
        'gemini-3.6-flash',
        'gemini-3.5-flash',
        'gemini-3.1-flash-lite',
        'gemini-flash-latest',
        'gemini-flash-lite-latest',
        'gemini-3.7-flash',
    ]
    last_error = None
    for model_name in models:
        try:
            return client.models.generate_content(
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
@app.post("/chat")
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
        parsed_result = _clean_and_parse_json(final_response.text)
        
        # Sanitasi data_payload jika berisi nilai kosong/null string
        dp = parsed_result.get("data_payload")
        if dp and isinstance(dp, dict):
            val = str(dp.get("value", "")).strip()
            if not val or val == "-" or val.lower() == "null" or val.lower() == "none":
                parsed_result["data_payload"] = None
        else:
            parsed_result["data_payload"] = None

        # Sanitasi chart_payload jika data tidak lengkap
        cp = parsed_result.get("chart_payload")
        if cp and isinstance(cp, dict):
            chart_data = cp.get("data")
            if not isinstance(chart_data, list) or len(chart_data) < 2:
                parsed_result["chart_payload"] = None
        else:
            parsed_result["chart_payload"] = None
                
        return parsed_result

    except Exception as e:
        return {
            "status": "error",
            "intent": "simple",
            "response_text": f"Terjadi kendala pada gateway LLM: {str(e)}",
            "data_payload": None,
            "chart_payload": None,
            "citations": [],
            "clarification_options": []
        }


async def _fetch_bps_data(source: str, keyword: str, domain: str) -> list[dict]:
    """Fetch data dari BPS Web API berdasarkan source type."""
    client = _get_bps_client()
    try:
        if source == "pressrelease":
            return await client.search_pressrelease(keyword=keyword, domain=domain)
        elif source == "statictable":
            return await client.search_statictable(keyword=keyword, domain=domain)
        elif source == "publication":
            return await client.search_publication(keyword=keyword, domain=domain)
        else:
            return []
    except Exception:
        return []
    finally:
        await client.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
