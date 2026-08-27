import os
import json
import base64
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

from bps_client import BpsApiClient
try:
    from rag_retriever import retrieve_rag_context, get_rag_citations
    _RAG_ENABLED = True
except Exception as _rag_err:
    print(f'[RAG] Module not available: {_rag_err}')
    _RAG_ENABLED = False
    def retrieve_rag_context(q, **kw): return ''
    def get_rag_citations(q, **kw): return []

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
    return {"status": "online", "service": "BPS AI Gateway Service", "version": "2.0.0"}

class ChatRequest(BaseModel):
    query: str
    history: list = []

REGION_DOMAIN_MAP = {
    "indonesia": "0000", "nasional": "0000", "pusat": "0000",
    "aceh": "1100", "sumatera utara": "1200", "sumut": "1200",
    "sumatera barat": "1300", "sumbar": "1300", "riau": "1400",
    "jambi": "1500", "sumatera selatan": "1600", "sumsel": "1600",
    "bengkulu": "1700", "lampung": "1800",
    "kepulauan bangka belitung": "1900", "bangka belitung": "1900", "babel": "1900",
    "kepulauan riau": "2100", "kepri": "2100",
    "dki jakarta": "3100", "jakarta": "3100",
    "jawa barat": "3200", "jabar": "3200",
    "jawa tengah": "3300", "jateng": "3300",
    "daerah istimewa yogyakarta": "3400", "yogyakarta": "3400", "diy": "3400", "jogja": "3400",
    "jawa timur": "3500", "jatim": "3500",
    "banten": "3600", "bali": "5100",
    "nusa tenggara barat": "5200", "ntb": "5200",
    "nusa tenggara timur": "5300", "ntt": "5300",
    "kalimantan barat": "6100", "kalbar": "6100",
    "kalimantan tengah": "6200", "kalteng": "6200",
    "kalimantan selatan": "6300", "kalsel": "6300",
    "kalimantan timur": "6400", "kaltim": "6400",
    "kalimantan utara": "6500", "kaltara": "6500",
    "sulawesi utara": "7100", "sulut": "7100",
    "sulawesi tengah": "7200", "sulteng": "7200", "palu": "7200",
    "sulawesi selatan": "7300", "sulsel": "7300",
    "sulawesi tenggara": "7400", "sultra": "7400",
    "gorontalo": "7500",
    "sulawesi barat": "7600", "sulbar": "7600",
    "maluku": "8100", "maluku utara": "8200",
    "papua barat": "9100", "papua": "9400",
}

def _resolve_domain(region_name: str) -> str:
    if not region_name:
        return "0000"
    return REGION_DOMAIN_MAP.get(region_name.strip().lower(), "0000")

INTENT_EXTRACTION_PROMPT = """
Anda adalah parser intent CERDAS untuk STATIX BPS AI Assistant Indonesia.
Analisis pertanyaan user dan extract ke JSON MURNI.

OUTPUT FORMAT:
{
  "intent": "numeric | comparison | knowledge | personnel | publication | out_of_scope",
  "keywords": ["kata kunci 1", "kata kunci 2"],
  "keywords_secondary": ["kata kunci alternatif"],
  "data_source": "pressrelease | statictable | publication | none",
  "regions": ["nama wilayah"],
  "domain_ids": ["kode BPS"],
  "needs_bps_data": true,
  "needs_rag_only": false,
  "is_comparison": false,
  "time_reference": "terbaru | 2025 | 2024 | historis | null"
}

ATURAN INTENT (WAJIB IKUTI URUTAN PRIORITAS):

1. PERSONNEL (needs_rag_only=true, needs_bps_data=false, data_source="none"):
   Semua pertanyaan tentang ORANG, JABATAN, PIMPINAN, PEGAWAI BPS:
   - "siapa kepala/ketua/pimpinan BPS [wilayah]"
   - "siapa yang menjabat sebagai..."
   - "siapa nama kepala BPS..."
   - "profil pejabat BPS"
   - "struktur organisasi BPS"
   CONTOH: "siapa ketua BPS Kota Palu" -> intent="personnel", needs_rag_only=true

2. KNOWLEDGE (needs_rag_only=true, needs_bps_data=false, data_source="none"):
   - Definisi/pengertian istilah statistik (IPM, TPT, inflasi, GK, dll)
   - Metodologi survei (SUSENAS, SAKERNAS, Sensus)
   - Cara menghitung indikator
   - Sejarah/profil lembaga BPS
   - Pertanyaan umum tanpa angka spesifik

3. NUMERIC (needs_bps_data=true, data_source="pressrelease"):
   - Angka/indikator TERBARU: inflasi, kemiskinan, IPM, NTP, PDRB, pengangguran, gini
   - Pertanyaan "berapa [indikator] [wilayah]"
   - Data terkini, update terakhir

4. COMPARISON (needs_bps_data=true, is_comparison=true):
   - Membandingkan 2+ wilayah: "bandingkan X dengan Y"
   - "perbedaan X dan Y"
   - "lebih tinggi mana antara..."

5. PUBLICATION (needs_bps_data=true, data_source="publication"):
   - Mencari buku/publikasi/laporan BPS
   - "download publikasi"

6. OUT_OF_SCOPE (needs_bps_data=false, data_source="none"):
   - Tidak ada hubungan dengan BPS atau statistik Indonesia

KODE DOMAIN BPS:
- Nasional: "0000", Sulawesi Tengah: "7200", Jakarta: "3100"
- Jawa Barat: "3200", Jawa Tengah: "3300", Jawa Timur: "3500"
- Bali: "5100", Sulawesi Selatan: "7300", Sumatera Utara: "1200"
- Kalimantan Timur: "6400", DIY: "3400", Riau: "1400"
- Kota Palu: "7271", Kab Donggala: "7203", Kab Poso: "7202"
- Kab Banggai: "7201", Kab Morowali: "7206"

Output HANYA JSON murni, tanpa penjelasan tambahan.
"""

RESPONSE_WITH_DATA_PROMPT = """
Anda adalah STATIX BPS AI Assistant resmi (Badan Pusat Statistik Republik Indonesia).
Asisten analitik statistik PALING CERDAS, akurat, dan profesional di Indonesia.

PIMPINAN BPS RESMI:
- Kepala BPS RI: Amalia Adininggar Widyasanti, ST, M.Si, M.Eng, Ph.D
- Wakil Kepala BPS RI: Dr. Sonny Harry B Harmadi, SE, ME, CRGP
- Kepala BPS Sulteng: Daryanto, S.Si., M.M.

DATA TERSEDIA (KOMBINASI BPS API + RAG KNOWLEDGE BASE):
--- DATA BPS ---
{bps_data}
--- END DATA ---

INSTRUKSI KETAT (WAJIB DIIKUTI):
1. GUNAKAN ANGKA NYATA dari data di atas. Prioritaskan DATA LIVE BPS API, diperkuat KONTEKS RAG.
2. KEJUJURAN TAHUN: Jika data tahun 2025 tidak ada tapi ada 2024 atau 2026, katakan JELAS "Data terbaru yang tersedia adalah [tahun]."
3. JANGAN MENGARANG ANGKA. Jika tidak ada angka spesifik, katakan "data tidak tersedia".
4. PERBANDINGAN WILAYAH: SELALU buat tabel markdown + chart_payload dengan kedua wilayah.
5. SUMBER SPESIFIK: citations = gunakan pdf_url ASLI dari data. Cantumkan judul PERSIS BRS yang relevan.
6. DATA RAG: Jika ada data historis dari RAG (misal "tren 2015-2025"), gunakan untuk memberikan konteks mendalam.
7. ANGKA DARI RAG: Jika RAG punya angka spesifik yang relevan (kemiskinan, IPM, inflasi per wilayah per tahun), SEBUTKAN dengan jelas.
8. Berikan 3 suggested_follow_ups yang spesifik dan cerdas.
9. Gunakan emoji, tabel markdown, dan heading yang menarik dalam response_text.

FORMAT JSON WAJIB (output JSON murni, tanpa code block):
{
  "status": "success",
  "intent": "numeric | comparison | knowledge",
  "response_text": "Teks naratif LENGKAP dengan markdown rapi (###, **, tabel, emoji).",
  "data_payload": {
     "indicator": "Nama indikator",
     "value": "Nilai dengan satuan",
     "region": "Wilayah",
     "period": "Periode"
  },
  "chart_payload": {
     "type": "bar",
     "title": "Judul grafik",
     "unit": "Satuan",
     "data": [{"label": "Wilayah A", "value": 2.57}, {"label": "Wilayah B", "value": 3.12}]
  },
  "suggested_follow_ups": ["Pertanyaan 1", "Pertanyaan 2", "Pertanyaan 3"],
  "citations": [
    {
      "title": "JUDUL PERSIS BRS/Publikasi dari data API",
      "url": "URL pdf_url ASLI dari data (bukan dikarang)",
      "release_date": "Tanggal rilis dari data",
      "source_name": "BPS [wilayah]"
    }
  ],
  "clarification_options": []
}
"""

RESPONSE_WITHOUT_DATA_PROMPT = """
Anda adalah STATIX BPS AI Assistant resmi (Badan Pusat Statistik Republik Indonesia).
Asisten analitik statistik profesional, akurat, dan cerdas.

PIMPINAN BPS RESMI:
- Kepala BPS RI: Amalia Adininggar Widyasanti, ST, M.Si, M.Eng, Ph.D
- Kepala BPS Sulteng: Daryanto, S.Si., M.M.

Jawab berdasarkan pengetahuan BPS resmi. Gunakan metodologi, istilah, dan profil yang benar.
Berikan 3 suggested_follow_ups spesifik dan cerdas tentang statistik BPS.

FORMAT JSON WAJIB (output JSON murni):
{
  "status": "success",
  "intent": "knowledge",
  "response_text": "Teks naratif markdown rapi.",
  "data_payload": null,
  "chart_payload": null,
  "suggested_follow_ups": ["Q1", "Q2", "Q3"],
  "citations": [{"title": "Portal Resmi BPS", "url": "https://www.bps.go.id", "release_date": "", "source_name": "Badan Pusat Statistik"}],
  "clarification_options": []
}
"""

def _generate_with_fallback(contents, system_instruction, response_mime_type="application/json", temperature=0.2):
    import time
    client = _get_gemini_client()
    models = [
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-1.5-flash',
        'gemini-2.0-flash-lite',
        'gemini-1.5-pro',
    ]
    last_error = None
    for model_name in models:
        for attempt in range(2):
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
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    time.sleep(1.5)
                else:
                    break
    raise last_error


def _clean_and_parse_json(text: str) -> dict:
    import re
    if not text:
        raise ValueError("Respons kosong")
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
    raise ValueError(f"Tidak dapat mem-parse JSON: {cleaned[:200]}")


@app.post("/api/v1/chat")
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        intent_response = _generate_with_fallback(
            contents=req.query,
            system_instruction=INTENT_EXTRACTION_PROMPT,
            temperature=0.1,
        )
        intent_data = _clean_and_parse_json(intent_response.text)

        needs_bps = intent_data.get("needs_bps_data", False)
        needs_rag_only = intent_data.get("needs_rag_only", False)
        intent_type = intent_data.get("intent", "knowledge")
        data_source = intent_data.get("data_source", "none")
        keywords = intent_data.get("keywords", [])
        keywords_secondary = intent_data.get("keywords_secondary", [])
        regions = intent_data.get("regions", ["Indonesia"])
        domain_ids = intent_data.get("domain_ids", [])

        # PERSONNEL & KNOWLEDGE: skip BPS API, use RAG only
        if intent_type in ("personnel", "knowledge") or needs_rag_only:
            needs_bps = False
            print(f"[ROUTER] Intent={intent_type} -> RAG-only mode (skip BPS API)")

        if not domain_ids:
            domain_ids = [_resolve_domain(r) for r in regions]
        if not domain_ids:
            domain_ids = ["0000"]

        keyword_str = " ".join(keywords) if keywords else ""

        bps_data_text = ""
        if needs_bps and data_source != "none":
            all_results = []
            fetch_domains = list(set(domain_ids))
            if "0000" not in fetch_domains:
                fetch_domains.append("0000")

            fetch_tasks = []
            for domain in fetch_domains[:3]:
                if data_source in ("pressrelease", "both"):
                    fetch_tasks.append(_fetch_bps_data("pressrelease", keyword_str, domain))
                if data_source in ("statictable", "both"):
                    fetch_tasks.append(_fetch_bps_data("statictable", keyword_str, domain))
                if data_source == "publication":
                    fetch_tasks.append(_fetch_bps_data("publication", keyword_str, domain))

            results_list = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for res in results_list:
                if isinstance(res, list):
                    all_results.extend(res)

            if not all_results and keywords_secondary:
                kw2 = " ".join(keywords_secondary)
                for domain in fetch_domains[:2]:
                    res2 = await _fetch_bps_data("pressrelease", kw2, domain)
                    all_results.extend(res2)

            if all_results:
                seen_titles = set()
                deduped = []
                for item in all_results:
                    t = item.get("title", "")
                    if t not in seen_titles:
                        seen_titles.add(t)
                        deduped.append(item)
                bps_data_text = json.dumps(deduped[:10], ensure_ascii=False, indent=2)

        # === RAG Context Enrichment ===
        rag_context = ''
        rag_cits = []
        if _RAG_ENABLED:
            try:
                rag_context = retrieve_rag_context(req.query, top_k=5, min_score=0.35)
                rag_cits = get_rag_citations(req.query, top_k=3, min_score=0.40)
                if rag_context:
                    print(f'[RAG] Found context ({len(rag_context)} chars) for: {req.query[:60]}')
            except Exception as _re:
                print(f'[RAG] retrieval error: {_re}')

        # === Combine BPS API + RAG into one enriched data block ===
        combined_data = ''
        if bps_data_text:
            combined_data += f'=== DATA LIVE BPS API ===\n{bps_data_text}\n\n'
        if rag_context:
            combined_data += f'=== KONTEKS RAG VECTOR DB (DATA HISTORIS + METODOLOGI) ===\n{rag_context}\n'

        if combined_data:
            system_prompt = RESPONSE_WITH_DATA_PROMPT.replace("{bps_data}", combined_data)
        elif rag_context:
            # RAG found context even if BPS API returned nothing
            system_prompt = RESPONSE_WITH_DATA_PROMPT.replace("{bps_data}", f"=== KONTEKS RAG KNOWLEDGE BASE ===\n{rag_context}")
        else:
            system_prompt = RESPONSE_WITHOUT_DATA_PROMPT

        final_response = _generate_with_fallback(
            contents=req.query,
            system_instruction=system_prompt,
            temperature=0.2,
        )
        parsed_result = _clean_and_parse_json(final_response.text)

        dp = parsed_result.get("data_payload")
        if dp and isinstance(dp, dict):
            val = str(dp.get("value", "")).strip()
            if not val or val == "-" or val.lower() in ("null", "none"):
                parsed_result["data_payload"] = None
        else:
            parsed_result["data_payload"] = None

        cp = parsed_result.get("chart_payload")
        if cp and isinstance(cp, dict):
            chart_data = cp.get("data")
            if not isinstance(chart_data, list) or len(chart_data) < 2:
                parsed_result["chart_payload"] = None
            else:
                cleaned_data = []
                for pt in chart_data:
                    try:
                        cleaned_data.append({
                            "label": str(pt.get("label", "")),
                            "value": float(pt.get("value", 0))
                        })
                    except (ValueError, TypeError):
                        pass
                if len(cleaned_data) >= 2:
                    cp["data"] = cleaned_data
                    parsed_result["chart_payload"] = cp
                else:
                    parsed_result["chart_payload"] = None
        else:
            parsed_result["chart_payload"] = None

        citations = parsed_result.get("citations", [])
        sanitized = []
        seen_titles = set()
        for c in citations:
            title = c.get("title", "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                sanitized.append({
                    "title": title,
                    "url": c.get("url", ""),
                    "release_date": c.get("release_date", ""),
                    "source_name": c.get("source_name", "Badan Pusat Statistik"),
                })
        # Merge RAG citations (tambahkan yang belum ada)
        for rc in rag_cits:
            rt = rc.get("title", "").strip()
            if rt and rt not in seen_titles:
                seen_titles.add(rt)
                sanitized.append(rc)
        parsed_result["citations"] = sanitized if sanitized else [{
            "title": "Portal Resmi BPS",
            "url": "https://www.bps.go.id",
            "release_date": "",
            "source_name": "Badan Pusat Statistik",
        }]

        return parsed_result

    except Exception as e:
        return {
            "status": "error",
            "intent": "simple",
            "response_text": f"Terjadi kendala pada gateway: {str(e)}",
            "data_payload": None,
            "chart_payload": None,
            "citations": [],
            "clarification_options": [],
            "suggested_follow_ups": [],
        }


async def _fetch_bps_data(source: str, keyword: str, domain: str) -> list[dict]:
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
