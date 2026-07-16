"""
Local embeddings using sentence-transformers.

Previously this called HuggingFace's free serverless Inference API over
HTTP. That API cold-starts unloaded models and is unreliable on the free
tier - timeouts and 5xx errors were happening even with generous timeouts
and clear error surfacing, because the failure is on HF's infrastructure,
not in this code. Loading the same model locally with sentence-transformers
removes the network dependency entirely: no token, no cold start, no
timeout, no rate limit. The model (~130MB) downloads once on first run and
is cached by HuggingFace's local cache afterward.

Keeps the same class name and embed_documents/embed_query interface as
before so nothing else in the codebase (vectorstore.py) needs to change
its calling convention beyond how the class is constructed.
"""
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class HFInferenceEmbeddings(Embeddings):
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", **_ignored_kwargs):
        # **_ignored_kwargs absorbs the old api_token=... call site so
        # vectorstore.py doesn't need to change its constructor call if
        # it's still passing api_token - it'll just be silently unused.
        # Loading happens once per process (this object is created once at
        # module import time in vectorstore.py and reused).
        self._model = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]