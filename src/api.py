"""
Phase 5a — FastAPI backend exposing the agent as /ask.
Run: uvicorn src.api:app --reload
Then open http://localhost:8000/docs
"""
import time
import csv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="Agentic RAG Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_PATH = "src/eval/request_log.csv"


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query):
    start = time.time()
    answer = run_agent(q.question)
    latency = round(time.time() - start, 2)

    _log_request(q.question, answer, latency)
    return {"answer": answer, "latency_seconds": latency}


def _log_request(question: str, answer: str, latency: float):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "question", "answer", "latency_seconds"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), question, answer, latency])