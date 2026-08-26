"""
BPS RAG Retriever - Plug-in modul untuk backend FastAPI
Melakukan semantic search di ChromaDB local dan mengembalikan konteks relevan.
"""
import os
import chromadb

# Path ke vector DB (relatif dari backend dir, atau absolute)
_VECTOR_DB_PATHS = [
    r"D:\Magang BPS\STATIX-Chatbot-BPS\ml-pipeline\data\vector_db\bps_knowledge",
    os.path.join(os.path.dirname(__file__), "..", "ml-pipeline", "data", "vector_db", "bps_knowledge"),
]

_collection = None

def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    for path in _VECTOR_DB_PATHS:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                client = chromadb.PersistentClient(path=abs_path)
                _collection = client.get_collection("bps_knowledge_base")
                print(f"[RAG] ChromaDB loaded: {_collection.count()} docs dari {abs_path}")
                return _collection
            except Exception as e:
                print(f"[RAG] Gagal load dari {abs_path}: {e}")
    print("[RAG] WARNING: Vector DB tidak ditemukan, RAG dinonaktifkan")
    return None

def retrieve_rag_context(query: str, top_k: int = 7, min_score: float = 0.32) -> str:
    """
    Ambil konteks relevan dari vector DB berdasarkan query.
    Mengutamakan PDF Full-Text chunks yang paling kaya informasi.
    Return string yang bisa dimasukkan ke system prompt.
    """
    col = _get_collection()
    if col is None:
        return ""
    try:
        # Fetch more results to allow re-ranking
        results = col.query(query_texts=[query], n_results=min(top_k * 2, 20))
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        context_parts = []
        seen_titles = set()
        
        for doc, meta, dist in zip(docs, metas, distances):
            score = round(1.0 - dist, 4)
            if score < min_score:
                continue
            
            title = meta.get("title", "Dokumen BPS")
            region = meta.get("region", "")
            category = meta.get("category", "")
            url = meta.get("url", "")
            release_date = meta.get("release_date", "")
            
            # Deduplicate by base title (remove chunk suffix)
            base_title = title.split(" [Bag.")[0].split(" [Hal.")[0]
            if base_title in seen_titles:
                continue
            seen_titles.add(base_title)
            
            is_pdf_full = "PDF Full" in category or "pdf_full" in meta.get("domain_id", "")
            pdf_badge = " [FULL PDF]" if is_pdf_full else ""
            date_badge = f" | Tanggal: {release_date}" if release_date else ""
            pdf_url_hint = f" | URL: {url}" if url and "download.php" in url else ""
            
            header = f"[{base_title} | {region}{pdf_badge} | skor={score}{date_badge}{pdf_url_hint}]"
            
            # PDF full chunks: take more content (2000 chars)
            # Abstrak/profil: take 1200 chars
            content_limit = 2000 if is_pdf_full else 1200
            context_parts.append(f"{header}\n{doc[:content_limit]}")
            
            if len(context_parts) >= top_k:
                break
        
        if not context_parts:
            return ""
        
        return "\n\n---\n".join(context_parts)
    except Exception as e:
        print(f"[RAG ERROR] {e}")
        return ""

def get_rag_citations(query: str, top_k: int = 3, min_score: float = 0.40) -> list:
    """
    Kembalikan list citation dari RAG untuk ditampilkan sebagai sumber.
    """
    col = _get_collection()
    if col is None:
        return []
    try:
        results = col.query(query_texts=[query], n_results=top_k)
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        citations = []
        seen = set()
        for meta, dist in zip(metas, distances):
            score = round(1.0 - dist, 4)
            if score < min_score:
                continue
            title = meta.get("title", "")
            url = meta.get("url", "")
            if title in seen:
                continue
            seen.add(title)
            citations.append({
                "title": title,
                "url": url,
                "release_date": meta.get("release_date", ""),
                "source_name": f"BPS {meta.get('region', 'Indonesia')}",
            })
        return citations
    except Exception:
        return []