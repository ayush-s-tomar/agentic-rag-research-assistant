"""
Phase 3a — Tools the agent can call.

- retrieve_docs: search the Qdrant Cloud vector store
- screen_resume: call your fine-tuned LoRA resume screener (llm-finetune-resume-screener project),
  expected to be running as its own local endpoint (see RESUME_SCREENER_URL in .env)
- route_query: classify query complexity and answer using a cheaper or larger Groq model
  accordingly — a real cost/latency tradeoff, not a label.
"""
import os
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
try:
    from vectorstore import get_vectorstore
except ImportError:
    from src.vectorstore import get_vectorstore

_db = get_vectorstore()

# Two Groq models at very different price/capability points. The router picks
# between them based on a cheap heuristic (query length) rather than paying
# gpt-oss-120b pricing/latency for questions that don't need it.
_ROUTED_MODELS = {
    "simple": "llama-3.1-8b-instant",
    "complex": "openai/gpt-oss-120b",
}

_router_llms = {
    name: ChatOpenAI(
        model=model_id,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    for name, model_id in _ROUTED_MODELS.items()
}


@tool
def retrieve_docs(query: str) -> str:
    """Search the document knowledge base and return the most relevant chunks."""
    # k=6 (was 4) so cross-document questions have a better chance of pulling
    # relevant chunks from more than one source document.
    docs = _db.similarity_search(query, k=6)
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


def _classify_complexity(query: str) -> str:
    """Word-count heuristic. Crude on purpose — the point is a near-zero-cost
    gate that avoids paying gpt-oss-120b latency/price for short factual asks.
    Replace with a trained classifier if the heuristic misroutes in practice.
    """
    return "complex" if len(query.split()) > 40 else "simple"


@tool
def route_query(query: str, complexity: str = "auto") -> str:
    """Answer a query using a cheap or large Groq model depending on complexity.

    complexity: 'simple', 'complex', or 'auto' (heuristic decides from query length).
    Returns the actual model answer, prefixed with which model handled it, so the
    routing decision is verifiable in the response rather than just claimed.
    """
    if complexity not in ("simple", "complex"):
        complexity = _classify_complexity(query)

    model_id = _ROUTED_MODELS[complexity]
    llm = _router_llms[complexity]

    try:
        response = llm.invoke(query)
        answer = response.content
    except Exception as e:
        return f"[routing to {model_id} failed: {e}]"

    return f"[routed to {model_id}, complexity={complexity}]\n{answer}"