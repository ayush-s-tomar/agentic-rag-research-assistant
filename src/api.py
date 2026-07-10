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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.agent import run_agent
from src.hf_embeddings import HFInferenceEmbeddings

app = FastAPI(title="Agentic RAG Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://agentic-rag-groq.streamlit.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_PATH = "src/eval/request_log.csv"
RAW_DIR = "data/raw"
CHROMA_DIR = "data/chroma_db"


class Query(BaseModel):
    question: str


def _get_db():
    embeddings = HFInferenceEmbeddings(api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


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

    # Only chunk the newly uploaded file — not the whole folder
    loader = PyPDFLoader(save_path)
    docs = loader.load()
    if not docs:
        return {"error": "No content could be extracted from the PDF."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Tag each chunk with a clean filename so we can list/delete by document later
    for chunk in chunks:
        chunk.metadata["source_file"] = file.filename

    # Append to the existing store instead of rebuilding it from scratch
    db = _get_db()
    db.add_documents(chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": len(chunks),
    }


@app.get("/documents")
def list_documents():
    db = _get_db()
    result = db.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])

    counts: dict[str, int] = {}
    for meta in metadatas:
        name = meta.get("source_file") or meta.get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1

    documents = [{"filename": name, "chunks": count} for name, count in counts.items()]
    return {"documents": documents}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    db = _get_db()
    db._collection.delete(where={"source_file": filename})

    file_path = os.path.join(RAW_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"status": "deleted", "filename": filename}


def _log_request(question: str, answer: str, latency: float):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "question", "answer", "latency_seconds"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), question, answer, latency])