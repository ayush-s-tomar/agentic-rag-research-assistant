"""
Shared Qdrant Cloud vectorstore module.

Replaces local Chroma persistence (data/chroma_db) so the knowledge base
survives Render free-tier spin-downs. Render wipes the local filesystem on
every restart/redeploy/idle spin-down, which is why re-uploading was
required every time — this module points at a persistent, externally
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

_embeddings = HFInferenceEmbeddings(api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

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
            "(.env locally, Render dashboard in production)."
        )
    _client = QdrantClient(url=url, api_key=api_key)
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
    path (similarity search) — both will share the same collection.
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