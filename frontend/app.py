"""
Phase 5b — Streamlit frontend for the assistant.
Run: streamlit run frontend/app.py
(Make sure the backend is running first: uvicorn src.api:app --reload)
"""
import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv
load_dotenv()

# Derive base URL once, build all endpoints from it
API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/ask").rstrip("/")
STREAM_URL = f"{API_BASE}/ask/stream"
UPLOAD_URL = f"{API_BASE}/upload"
DOCUMENTS_URL = f"{API_BASE}/documents"

st.set_page_config(page_title="Research Assistant", page_icon="🔎")
st.title("🔎 Agentic RAG Research Assistant")
st.caption("Ask a question — the agent will retrieve documents, route to the right model, and answer.")

# --- Sidebar: PDF upload + document list ---
with st.sidebar:
    st.header("📄 Upload a document")
    uploaded_file = st.file_uploader("Add a PDF to the knowledge base", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload & Ingest"):
            with st.spinner("Uploading and embedding... (first request after idle time can take up to a minute)"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(UPLOAD_URL, files=files, timeout=300)
                    resp.raise_for_status()
                    data = resp.json()
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.success(f"Added {data['filename']} — {data['chunks_added']} chunks embedded.")
                        st.rerun()
                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")

    st.divider()
    st.header("📚 Knowledge base")

    try:
        resp = requests.get(DOCUMENTS_URL, timeout=30)
        resp.raise_for_status()
        documents = resp.json().get("documents", [])
    except requests.RequestException as e:
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
                        del_resp = requests.delete(f"{DOCUMENTS_URL}/{doc['filename']}", timeout=30)
                        del_resp.raise_for_status()
                        st.rerun()
                    except requests.RequestException as e:
                        st.error(f"Delete failed: {e}")

if "history" not in st.session_state:
    st.session_state.history = []

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

        def token_stream():
            try:
                with requests.post(
                    STREAM_URL,
                    json={"question": question},
                    stream=True,
                    timeout=100,
                ) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk
            except requests.RequestException as e:
                yield f"Error contacting backend: {e}"

        answer = st.write_stream(token_stream())
        latency = round(time.time() - start, 2)
        st.caption(f"Latency: {latency}s")

    st.session_state.history.append(("assistant", answer))