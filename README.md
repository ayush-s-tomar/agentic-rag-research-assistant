# 🔎 Agentic RAG Research Assistant

A portfolio-grade agentic RAG system: retrieval-augmented generation, tool-routing via LangGraph, cost-aware model selection, and a fine-tuned resume screener — deployed and live.

**🔗 Live demo:** https://agentic-rag-groq.streamlit.app
**🔗 API docs:** https://agentic-rag-research-assistant-jjch.onrender.com/docs

> ⚠️ The backend runs on Render's free tier and sleeps after inactivity — the first request after idle can take 30–60s to wake up.

---

## What it does

Ask a question, and the agent:
1. Retrieves relevant chunks from a Chroma vector store built from your uploaded PDFs
2. Answers **only** from retrieved content — if nothing relevant is found, it says so instead of guessing
3. Can route queries between a cheap model and a larger one based on complexity
4. Can screen resumes via a fine-tuned LoRA model (optional, separate service)

Upload a PDF, ask about it, and see the grounding in action — including the refusal behavior when a question is out of scope.

---

## Architecture
User question
│
▼
FastAPI backend (src/api.py)  ──deployed on Render
│
▼
LangGraph ReAct agent (src/agent.py)  ──Groq / Llama 3.3 70B
│
├── retrieve_docs  → Chroma vector store (local documents)
├── screen_resume  → fine-tuned LoRA resume screener (separate service)
└── route_query    → cost router: cheap model vs. larger model
│
▼
Answer + latency logged to src/eval/request_log.csv
│
▼
Streamlit frontend (frontend/app.py)  ──deployed on Streamlit Community Cloud

## Tech stack

| Layer | Tech |
|---|---|
| LLM inference | Groq (Llama 3.3 70B Versatile) |
| Agent framework | LangGraph (ReAct agent) |
| Vector store | Chroma |
| Embeddings | HuggingFace `bge-small-en-v1.5` |
| Backend | FastAPI |
| Frontend | Streamlit |
| Backend hosting | Render (free tier) |
| Frontend hosting | Streamlit Community Cloud |

---

## Quickstart — run it locally

```powershell
# 1. Setup
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# then edit .env and add your GROQ_API_KEY (https://console.groq.com)

# 2. Add documents
# Drop PDF files into data\raw\

# 3. Build the knowledge base
python src\ingest.py

# 4. Sanity-check retrieval
python src\retrieve_test.py "a question about your documents"

# 5. Test the full agent directly
python src\agent.py "a question about your documents"

# 6. Run the backend
uvicorn src.api:app --reload
# visit http://localhost:8000/docs

# 7. Run the frontend (in a second terminal, venv activated)
streamlit run frontend\app.py
```

---

## Project phases

| Phase | What | File(s) |
|---|---|---|
| 1 | Ingest & chunk documents | `src/ingest.py` |
| 2 | Embed, store, basic RAG | `src/ingest.py`, `src/retrieve_test.py`, `src/rag_basic.py` |
| 3 | Agent + tools + routing | `src/agent.py`, `src/tools/tools.py` |
| 4 | Evaluation | `src/eval/run_eval.py`, `src/eval/cost_comparison.py` |
| 5 | Deployment | `src/api.py`, `frontend/app.py` — see below |
| 6 | Writeup | this README |

---

## Deployment

**Backend (Render):**
- Build command: `pip install -r requirements.txt && python src/ingest.py`
- Start command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
- Environment variables: `GROQ_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`
- The vector store rebuilds from `data/raw/` on every deploy, since Render's free-tier filesystem doesn't persist across restarts.

**Frontend (Streamlit Community Cloud):**
- Main file: `frontend/app.py`
- Secret: `API_URL = "https://agentic-rag-research-assistant-jjch.onrender.com"`

CORS on the backend is scoped to the deployed Streamlit origin only.

---

## Design decisions & tradeoffs

- **Groq over OpenAI** — chosen for fast, cheap inference on Llama 3.3 70B, at the cost of occasional tool-calling quirks compared to GPT-4-class models (see Known limitations).
- **Strict grounding via system prompt** — the agent is instructed to refuse rather than answer from general knowledge, prioritizing trustworthiness over coverage.
- **Chroma rebuilt on every deploy** rather than persisted, trading a few seconds of startup time for zero infra complexity on the free tier.

## Known limitations

- Llama 3.3's tool-calling via Groq occasionally malforms a function call on ambiguous questions, surfacing as a 500 error. Not yet hardened — a candidate for a stricter tool-calling system prompt or a fallback retry.
- `screen_resume` expects a separate local/deployed service (`llm-finetune-resume-screener`) and fails gracefully if it's not reachable — this feature isn't live in the deployed demo.
- `route_query`'s complexity routing is currently a placeholder heuristic (word count), not the full `llm-cost-router` logic.

---

## Before presenting this in interviews (in progress)

- [ ] Replace `src/eval/eval_set.jsonl` with 30–50 real Q&A pairs from your document set
- [ ] Fill in real pricing in `src/eval/cost_comparison.py`
- [ ] Wire `route_query` to the actual `llm-cost-router` logic
- [ ] Add a demo GIF/video below
- [ ] Fill in Results section below

## Results

**Eval scores (RAGAS)**

| Metric | Score |
|---|---|
| Faithfulness | — |
| Answer relevancy | — |
| Context precision | — |

**Cost comparison**

| Approach | $ / 1000 queries |
|---|---|
| Fine-tuned model only | — |
| Groq Llama 3.3 only | — |
| Hybrid router | — |

## What I'd improve with more time

<<<<<<< HEAD
_Fill in: e.g. persistent vector store, streaming responses, better tool-call error recovery, real cost-router integration._
=======
_Fill in: e.g. persistent vector store, streaming responses, better tool-call error recovery, real cost-router integration._
>>>>>>> b87c673099e0f77ff4f953d41c98d7a26b40c210
