"""
Merged Streamlit frontend + backend for the Agentic RAG Research Assistant.

This replaces the old split architecture (Streamlit frontend -> HTTP -> FastAPI
backend on Render) with a single self-contained Streamlit app. There is no
separate backend anymore, so there is nothing that can be "suspended" by
Render's free-tier hour cap — everything runs inside this one process.

Run locally:  streamlit run frontend/app.py
Deploy:       Streamlit Community Cloud, main file path = frontend/app.py

Required secrets (Streamlit Cloud -> App settings -> Secrets, or a local
.streamlit/secrets.toml):
    GROQ_API_KEY = "..."
    HUGGINGFACEHUB_API_TOKEN = "..."
    QDRANT_URL = "..."
    QDRANT_API_KEY = "..."
    RESUME_SCREENER_URL = "..."   # optional, screen_resume degrades gracefully without it
"""
import os
import sys
import time
import uuid
import pathlib

import streamlit as st

# --- Make src/ importable regardless of Streamlit Cloud's working directory ---
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --- Bridge Streamlit secrets into os.environ so agent.py / vectorstore.py
# --- (which read via os.getenv, unchanged from the Render version) keep working ---
_REQUIRED_SECRETS = ("GROQ_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "QDRANT_URL", "QDRANT_API_KEY")
_OPTIONAL_SECRETS = ("RESUME_SCREENER_URL",)

_missing = []
for _key in _REQUIRED_SECRETS + _OPTIONAL_SECRETS:
    _val = st.secrets.get(_key)
    if _val:
        os.environ[_key] = str(_val)
    elif _key in _REQUIRED_SECRETS:
        _missing.append(_key)

if _missing:
    st.error(
        "Missing required secret(s): "
        + ", ".join(_missing)
        + ". Go to your app's Settings -> Secrets on Streamlit Cloud, add them "
        "in the form KEY = \"value\" (one per line, exact names above), Save, "
        "then Manage app -> Reboot app."
    )
    st.stop()

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import io

from src.agent import run_agent, stream_agent            # noqa: E402  (import after sys.path fix)
from src.vectorstore import get_vectorstore, _get_client, COLLECTION_NAME  # noqa: E402

st.set_page_config(page_title="Research Assistant", page_icon="🔎")
st.title("🔎 Agentic RAG Research Assistant")
st.caption("Ask a question — the agent will retrieve documents, route to the right model, and answer.")

if "last_upload_status" not in st.session_state:
    st.session_state.last_upload_status = None
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# In-process replacements for what src/api.py used to do over HTTP
# ---------------------------------------------------------------------------

# Matches src/ingest.py exactly, so chunks from new sidebar uploads are the
# same granularity as chunks already in Qdrant from the original ingest run.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _basename(source: str) -> str:
    """PyPDFLoader (used in ingest.py) stores the full file path in the
    'source' metadata field, e.g. 'data/raw/report.pdf'. Normalize down to
    just the filename for display and for matching on delete.
    """
    return os.path.basename(source) if source else "unknown"


def _extract_source(payload: dict) -> str:
    """langchain-qdrant nests document metadata under payload['metadata'],
    e.g. {'page_content': ..., 'metadata': {'source': 'data/raw/x.pdf'}} —
    not as a flat top-level field. Check both, nested first.
    """
    payload = payload or {}
    metadata = payload.get("metadata") or {}
    return metadata.get("source") or payload.get("source") or ""


def ingest_pdf(filename: str, file_bytes: bytes) -> int:
    """Extract text from a PDF, chunk it, embed it, and store it in Qdrant
    tagged with its source filename — same metadata key ("source") and
    chunk settings as the original src/ingest.py, so documents uploaded via
    this sidebar are indistinguishable from ones ingested locally.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not full_text.strip():
        raise ValueError("No extractable text found in this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(full_text)
    if not chunks:
        raise ValueError("PDF produced no chunks after splitting.")

    vectorstore = get_vectorstore()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": filename} for _ in chunks]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def list_documents() -> list[dict]:
    """Return [{filename, chunks}, ...] by scanning Qdrant point payloads."""
    client = _get_client()
    counts: dict[str, int] = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            filename = _basename(_extract_source(p.payload))
            counts[filename] = counts.get(filename, 0) + 1
        if next_offset is None:
            break
    return [{"filename": f, "chunks": c} for f, c in sorted(counts.items())]


def delete_document(filename: str) -> None:
    """Deletes any point whose nested 'metadata.source' ends with this
    filename — handles both bare filenames (sidebar uploads) and full
    paths like 'data/raw/report.pdf' (original ingest.py uploads). Qdrant
    supports dot notation to filter on nested payload fields.
    """
    client = _get_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            should=[
                FieldCondition(key="metadata.source", match=MatchValue(value=filename)),
                FieldCondition(key="metadata.source", match=MatchValue(value=f"data/raw/{filename}")),
                FieldCondition(key="source", match=MatchValue(value=filename)),
                FieldCondition(key="source", match=MatchValue(value=f"data/raw/{filename}")),
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Sidebar: upload + knowledge base management (was calling UPLOAD_URL /
# DOCUMENTS_URL on the old Render backend, now calls the functions above)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Upload a document")
    uploaded_file = st.file_uploader("Add a PDF to the knowledge base", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload & Ingest"):
            with st.spinner("Uploading and embedding..."):
                try:
                    chunks_added = ingest_pdf(uploaded_file.name, uploaded_file.getvalue())
                    st.session_state.last_upload_status = (
                        "success",
                        f"Added {uploaded_file.name} — {chunks_added} chunks embedded.",
                    )
                except Exception as e:
                    st.session_state.last_upload_status = ("error", f"Upload failed: {e}")

    if st.session_state.last_upload_status:
        kind, msg = st.session_state.last_upload_status
        if kind == "success":
            st.success(msg)
        else:
            st.error(msg)

    st.divider()
    st.header("📚 Knowledge base")

    try:
        documents = list_documents()
    except Exception as e:
        documents = []
        st.caption(f"Could not load document list: {e}")

    if not documents:
        st.caption("No documents ingested yet.")
    else:
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {doc['filename']}")
                st.caption(f"{doc['chunks']} chunks")
            with col2:
                if st.button("🗑️", key=f"delete_{doc['filename']}"):
                    try:
                        delete_document(doc["filename"])
                        st.session_state.last_upload_status = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")


# ---------------------------------------------------------------------------
# Chat (was POSTing to STREAM_URL on the old Render backend, now calls
# stream_agent() directly, in-process, no HTTP)
# ---------------------------------------------------------------------------

question = st.chat_input("Ask a question...")

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        start = time.time()
        answer = st.write_stream(stream_agent(question))
        latency = round(time.time() - start, 2)
        st.caption(f"Latency: {latency}s")

    st.session_state.history.append(("assistant", answer))