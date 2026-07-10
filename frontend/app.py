"""
Phase 5b — Streamlit frontend for the assistant.
Run: streamlit run frontend/app.py
(Make sure the backend is running first: uvicorn src.api:app --reload)
"""
import streamlit as st
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Derive base URL once, build both endpoints from it
API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/ask").rstrip("/")
ASK_URL = f"{API_BASE}/ask"
UPLOAD_URL = f"{API_BASE}/upload"

st.set_page_config(page_title="Research Assistant", page_icon="🔎")
st.title("🔎 Agentic RAG Research Assistant")
st.caption("Ask a question — the agent will retrieve documents, route to the right model, and answer.")

# --- Sidebar: PDF upload ---
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
                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")

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
        with st.spinner("Thinking... (first question after idle time can take up to a minute — waking up the backend)"):
            try:
                resp = requests.post(ASK_URL, json={"question": question}, timeout=100)
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                st.write(answer)
                st.caption(f"Latency: {data.get('latency_seconds', '?')}s")
            except requests.RequestException as e:
                answer = f"Error contacting backend: {e}"
                st.error(answer)

    st.session_state.history.append(("assistant", answer))