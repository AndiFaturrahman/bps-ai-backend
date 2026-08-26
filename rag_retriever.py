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

def retrieve_rag_context(query: str, top_k: int = 5, min_score: float = 0.35) -> str:
    """
    Ambil konteks relevan dari vector DB berdasarkan query.
    Return string yang bisa dimasukkan ke system prompt.
    """
    col = _get_collection()
    if col is None:
        return ""
    try:
        results = col.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        context_parts = []
        for doc, meta, dist in zip(docs, metas, distances):
            score = round(1.0 - dist, 4)
            if score < min_score:
                continue
            title = meta.get("title", "Dokumen BPS")
            region = meta.get("region", "")
            category = meta.get("category", "")
            url = meta.get("url", "")
            pdf_hint = f" | URL PDF: {url}" if url and ("download.php" in url or url.endswith(".pdf")) else ""
            context_parts.append(
                f"[Sumber: {title} | {region} | {category} | skor={score}{pdf_hint}]\n{doc[:1200]}"
            )
        
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