"""
Shared Qdrant Cloud vectorstore module.

Replaces local Chroma persistence (data/chroma_db) so the knowledge base
survives Render free-tier spin-downs. Render wipes the local filesystem on
every restart/redeploy/idle spin-down, which is why re-uploading was
required every time - this module points at a persistent, externally
hosted vector store instead.

Requires QDRANT_URL and QDRANT_API_KEY to be set in the environment
(.env locally, Render environment variables in production).
"""
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

try:
    from hf_embeddings import HFInferenceEmbeddings
except ImportError:
    from src.hf_embeddings import HFInferenceEmbeddings

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "nimbus_docs")
# BAAI/bge-small-en-v1.5 (the model used in HFInferenceEmbeddings) outputs
# 384-dimensional vectors. If you ever change the embedding model, this
# must be updated to match, or collection creation will silently produce
# a store with the wrong vector size.
EMBEDDING_DIM = 384

# Qdrant Cloud free-tier clusters can idle-suspend and take a while to wake,
# similar to Render's old free-tier spin-down. Without an explicit timeout,
# QdrantClient's underlying HTTP calls can hang indefinitely on a suspended
# or unreachable cluster - which looks like a frozen app with nothing in
# the logs. 15s is enough for a normal request; failing loud beats hanging.
QDRANT_TIMEOUT = 15

# Embeddings now run locally (sentence-transformers) instead of calling
# HuggingFace's remote Inference API, so no api_token is needed here
# anymore. HFInferenceEmbeddings still accepts and ignores stray kwargs,
# but we no longer pass api_token at all since it has no effect.
_embeddings = HFInferenceEmbeddings()

_client: QdrantClient | None = None
_vectorstore: QdrantVectorStore | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in the environment "
            "(.env locally, Streamlit Cloud secrets in production)."
        )
    try:
        _client = QdrantClient(url=url, api_key=api_key, timeout=QDRANT_TIMEOUT)
        # Force an actual network call now (client construction alone is lazy
        # and won't surface a bad URL/host until first use) so a dead cluster
        # fails here, clearly, instead of hanging on the first real query.
        _client.get_collections()
    except Exception as e:
        _client = None
        raise RuntimeError(
            f"Could not connect to Qdrant at {url!r} within {QDRANT_TIMEOUT}s. "
            "Check that the cluster is running (not paused/suspended on the "
            f"free tier) and that QDRANT_URL/QDRANT_API_KEY are correct. Details: {e}"
        )
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def get_vectorstore() -> QdrantVectorStore:
    """Return a cached QdrantVectorStore backed by Qdrant Cloud.

    Creates the collection on first call if it doesn't exist yet. Safe to
    call from both the ingest path (adding documents) and the retrieve
    path (similarity search) - both will share the same collection.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    client = _get_client()
    _ensure_collection(client)
    _vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=_embeddings,
    )
    return _vectorstore