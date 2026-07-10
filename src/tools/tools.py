"""
Phase 3a — Tools the agent can call.

- retrieve_docs: search the Qdrant Cloud vector store
- screen_resume: call your fine-tuned LoRA resume screener (llm-finetune-resume-screener project),
  expected to be running as its own local endpoint (see RESUME_SCREENER_URL in .env)
- route_query: decide cheap-model vs big-model, reusing your llm-cost-router logic
"""
import os
import requests
from langchain_core.tools import tool
try:
    from vectorstore import get_vectorstore
except ImportError:
    from src.vectorstore import get_vectorstore

_db = get_vectorstore()


@tool
def retrieve_docs(query: str) -> str:
    """Search the document knowledge base and return the most relevant chunks."""
    docs = _db.similarity_search(query, k=4)
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(d.page_content for d in docs)


@tool
def screen_resume(resume_text: str) -> str:
    """Send resume text to the fine-tuned LoRA resume screener and return its structured verdict.

    Requires the resume-screener service to be running locally
    (see llm-finetune-resume-screener project, compare_baseline_vs_finetuned.py
    or a small FastAPI wrapper around the saved adapter).
    """
    url = os.getenv("RESUME_SCREENER_URL", "http://localhost:8001/screen")
    try:
        resp = requests.post(url, json={"resume_text": resume_text}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("verdict", resp.text)
    except requests.RequestException as e:
        return f"Resume screener unavailable ({e}). Is the local service running?"


@tool
def route_query(query: str, complexity: str = "auto") -> str:
    """Route a query to the cheap fine-tuned model or a larger model based on complexity.

    complexity: 'simple', 'complex', or 'auto' (let the router decide).
    Mirrors the routing logic from the llm-cost-router project.
    """
    if complexity == "auto":
        complexity = "complex" if len(query.split()) > 40 else "simple"

    model = "fine-tuned-qwen2.5-0.5b" if complexity == "simple" else "gpt-4o-mini"
    return f"[routed to {model}] (complexity={complexity})"