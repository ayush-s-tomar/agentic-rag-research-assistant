"""
Phase 5b — Streamlit frontend for the assistant.
Run: streamlit run frontend/app.py
(Make sure the backend is running first: uvicorn src.api:app --reload)
"""
import streamlit as st
import requests

import os
API_URL = os.getenv("API_URL", "http://localhost:8000/ask")

st.set_page_config(page_title="Research Assistant", page_icon="🔎")
st.title("🔎 Agentic RAG Research Assistant")
st.caption("Ask a question — the agent will retrieve documents, route to the right model, and answer.")

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
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(API_URL, json={"question": question}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                st.write(answer)
                st.caption(f"Latency: {data.get('latency_seconds', '?')}s")
            except requests.RequestException as e:
                answer = f"Error contacting backend: {e}"
                st.error(answer)

    st.session_state.history.append(("assistant", answer))
