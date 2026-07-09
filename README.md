# Agentic RAG Research Assistant

A portfolio-grade system combining retrieval-augmented generation, agentic tool use, cost-based model routing (reusing `llm-cost-router`), and a fine-tuned resume screener (reusing `llm-finetune-resume-screener`) — with evaluation and a deployed demo.

## Architecture

```
User question
     |
     v
FastAPI backend (src/api.py)
     |
     v
LangGraph ReAct agent (src/agent.py)
     |
     +--> retrieve_docs  -> Chroma vector store (local documents)
     +--> screen_resume  -> fine-tuned LoRA resume screener (separate service)
     +--> route_query    -> cost router: cheap fine-tuned model vs GPT-4o-mini
     |
     v
Answer + latency logged to src/eval/request_log.csv
     |
     v
Streamlit frontend (frontend/app.py)
```

## Quickstart — run everything from scratch

```powershell
# 1. Setup
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# then edit .env and add your OPENAI_API_KEY

# 2. Add documents
# Drop PDF files into data\raw\

# 3. Build the knowledge base
python src\ingest.py
python src\embed_store.py

# 4. Sanity-check retrieval
python src\retrieve_test.py "a question about your documents"

# 5. Test basic RAG (no agent yet)
python src\rag_basic.py "a question about your documents"

# 6. Test the full agent
python src\agent.py "a question about your documents"

# 7. Run the backend
uvicorn src.api:app --reload
# visit http://localhost:8000/docs

# 8. Run the frontend (in a second terminal, venv activated)
streamlit run frontend\app.py

# 9. Evaluate
python src\eval\run_eval.py
python src\eval\cost_comparison.py
```

## Project phases

| Phase | What | File(s) |
|---|---|---|
| 1 | Ingest & chunk documents | `src/ingest.py` |
| 2 | Embed, store, basic RAG | `src/embed_store.py`, `src/retrieve_test.py`, `src/rag_basic.py` |
| 3 | Agent + tools + routing | `src/agent.py`, `src/tools/tools.py` |
| 4 | Evaluation | `src/eval/run_eval.py`, `src/eval/cost_comparison.py` |
| 5 | Deployment | `src/api.py`, `frontend/app.py` |
| 6 | Writeup | this README |

## Before you deploy / present this

- [ ] Replace `src/eval/eval_set.jsonl` with 30-50 real Q&A pairs from your document set
- [ ] Fill in real pricing in `src/eval/cost_comparison.py`
- [ ] Wire `screen_resume` in `src/tools/tools.py` to your actual resume-screener service URL
- [ ] Wire `route_query` to your actual `llm-cost-router` logic instead of the placeholder heuristic
- [ ] Add an architecture diagram image and a demo GIF/video below
- [ ] Deploy backend (Render/Fly.io) and frontend (Streamlit Community Cloud)
- [ ] Add a "What I'd improve with more time" section

## Results

_Fill in after running Phase 4:_

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
| GPT-4o-mini only | — |
| Hybrid router | — |
