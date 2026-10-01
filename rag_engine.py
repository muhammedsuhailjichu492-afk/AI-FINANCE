
import os
import requests
import chromadb

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(name="finance_records")


def is_ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def embed_text(text: str):
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def index_transaction(transaction_id: int, text: str, metadata: dict):
    """Add or update a transaction's embedding in the vector store."""
    try:
        embedding = embed_text(text)
    except requests.RequestException:
        return False  # Ollama not running — indexing skipped, rest of app still works
    _collection.upsert(
        ids=[f"txn-{transaction_id}"],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )
    return True


def query_similar(query_text: str, n_results: int = 5):
    """Retrieve the most relevant real transactions/documents for a query."""
    try:
        embedding = embed_text(query_text)
    except requests.RequestException:
        return []
    count = _collection.count()
    if count == 0:
        return []
    results = _collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, count),
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]
