"""
Phase 2a — Embed chunks and persist them to a local Chroma vector store.
Run: python src/embed_store.py
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from ingest import load_and_chunk


def build_vectorstore():
    chunks = load_and_chunk()
    if not chunks:
        return

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory="data/chroma_db"
    )
    db.persist()
    print("Vectorstore built and persisted to data/chroma_db")


if __name__ == "__main__":
    build_vectorstore()
