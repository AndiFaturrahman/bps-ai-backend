"""
BPS Web API Client
==================
Helper module untuk mengakses data dari BPS Web API (webapi.bps.go.id).
Digunakan oleh Gemini gateway untuk memperkaya jawaban dengan data real-time.
"""

import httpx
from typing import Optional

BPS_API_BASE = "https://webapi.bps.go.id/v1/api"


class BpsApiClient:
    """Client untuk BPS Web API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=15.0)

    # ------------------------------------------------------------------
    # PRESS RELEASE (Berita Resmi Statistik / BRS)
    # ------------------------------------------------------------------
    async def search_pressrelease(
        self,
        keyword: str = "",
        domain: str = "0000",
        page: int = 1,
    ) -> list[dict]:
        """
        Cari Berita Resmi Statistik berdasarkan keyword.
        domain '0000' = Nasional / Pusat.
        """
        params = {
            "model": "pressrelease",
            "domain": domain,
            "key": self.api_key,
            "page": str(page),
            "lang": "ind",
        }
        if keyword:
            params["keyword"] = keyword

        resp = await self._client.get(f"{BPS_API_BASE}/list/", params=params)
        data = resp.json()

        if data.get("status") != "OK" or data.get("data-availability") != "available":
            return []

        raw = data.get("data", [])
        if len(raw) < 2:
            return []

        items = raw[1] if isinstance(raw[1], list) else []
        results = []
        for item in items[:5]:  # Max 5 BRS
            results.append({
                "title": item.get("title", ""),
                "subject": item.get("subj", ""),
                "abstract": _strip_html(item.get("abstract", "")),
                "release_date": item.get("rl_date", ""),
                "pdf_url": item.get("pdf", ""),
            })
        return results

    # ------------------------------------------------------------------
    # STATIC TABLE
    # ------------------------------------------------------------------
    async def search_statictable(
        self,
        keyword: str = "",
        domain: str = "0000",
        page: int = 1,
    ) -> list[dict]:
        """Cari tabel statistik berdasarkan keyword."""
        params = {
            "model": "statictable",
            "domain": domain,
            "key": self.api_key,
            "page": str(page),
            "lang": "ind",
        }
        if keyword:
            params["keyword"] = keyword

        resp = await self._client.get(f"{BPS_API_BASE}/list/", params=params)
        data = resp.json()

        if data.get("status") != "OK" or data.get("data-availability") != "available":
            return []

        raw = data.get("data", [])
        if len(raw) < 2:
            return []

        items = raw[1] if isinstance(raw[1], list) else []
        results = []
        for item in items[:5]:
            results.append({
                "table_id": item.get("table_id", ""),
                "title": item.get("title", ""),
                "subject": item.get("subj", ""),
                "updated": item.get("updt_date", ""),
                "url": item.get("excel", "") or item.get("pdf", ""),
            })
        return results

    # ------------------------------------------------------------------
    # SUBJECT LIST
    # ------------------------------------------------------------------
    async def list_subjects(self, domain: str = "0000") -> list[dict]:
        """List semua subjek statistik untuk domain tertentu."""
        params = {
            "model": "subject",
            "domain": domain,
            "key": self.api_key,
            "lang": "ind",
        }
        resp = await self._client.get(f"{BPS_API_BASE}/list/", params=params)
        data = resp.json()

        if data.get("status") != "OK" or data.get("data-availability") != "available":
            return []

        raw = data.get("data", [])
        if len(raw) < 2:
            return []

        items = raw[1] if isinstance(raw[1], list) else []
        return [
            {"sub_id": s.get("sub_id"), "title": s.get("title", ""), "category": s.get("subcat", "")}
            for s in items
        ]

    # ------------------------------------------------------------------
    # PUBLICATION
    # ------------------------------------------------------------------
    async def search_publication(
        self,
        keyword: str = "",
        domain: str = "0000",
        page: int = 1,
    ) -> list[dict]:
        """Cari publikasi BPS berdasarkan keyword."""
        params = {
            "model": "publication",
            "domain": domain,
            "key": self.api_key,
            "page": str(page),
            "lang": "ind",
        }
        if keyword:
            params["keyword"] = keyword

        resp = await self._client.get(f"{BPS_API_BASE}/list/", params=params)
        data = resp.json()

        if data.get("status") != "OK" or data.get("data-availability") != "available":
            return []

        raw = data.get("data", [])
        if len(raw) < 2:
            return []

        items = raw[1] if isinstance(raw[1], list) else []
        results = []
        for item in items[:5]:
            results.append({
                "pub_id": item.get("pub_id", ""),
                "title": item.get("title", ""),
                "abstract": _strip_html(item.get("abstract", "")),
                "issn": item.get("issn", ""),
                "release_date": item.get("rl_date", ""),
                "pdf_url": item.get("pdf", ""),
                "cover": item.get("cover", ""),
            })
        return results

    # ------------------------------------------------------------------
    # DOMAIN (Wilayah)
    # ------------------------------------------------------------------
    async def list_domains(self, domain_type: str = "prov") -> list[dict]:
        """List semua domain (provinsi / kabupaten)."""
        params = {
            "type": domain_type,
            "key": self.api_key,
        }
        resp = await self._client.get(f"{BPS_API_BASE}/domain/", params=params)
        data = resp.json()

        if data.get("status") != "OK":
            return []

        raw = data.get("data", [])
        if len(raw) < 2:
            return []

        items = raw[1] if isinstance(raw[1], list) else []
        return [
            {"domain_id": d.get("domain_id", ""), "domain_name": d.get("domain_name", "")}
            for d in items
        ]

    async def close(self):
        await self._client.aclose()


# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------
def _strip_html(text: str) -> str:
    """Hapus tag HTML sederhana dari string."""
    import re
    clean = re.sub(r"<[^>]+>", "", text)
    clean = clean.replace("&nbsp;", " ").replace("&amp;", "&")
    return clean.strip()
