"""
Phase 5a — FastAPI backend exposing the agent as /ask.
Run: uvicorn src.api:app --reload
Then open http://localhost:8000/docs
"""
import time
import csv
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent import run_agent
from src.ingest import load_and_chunk, embed_and_store

app = FastAPI(title="Agentic RAG Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://agentic-rag-groq.streamlit.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_PATH = "src/eval/request_log.csv"
RAW_DIR = "data/raw"


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


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    os.makedirs(RAW_DIR, exist_ok=True)
    save_path = os.path.join(RAW_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = load_and_chunk(data_dir=RAW_DIR)
    if not chunks:
        return {"error": "No content could be extracted from the PDF."}

    embed_and_store(chunks)
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": len(chunks),
    }


def _log_request(question: str, answer: str, latency: float):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "question", "answer", "latency_seconds"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), question, answer, latency])