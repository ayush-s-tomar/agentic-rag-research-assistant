"""
Phase 1 — Load, chunk, and embed documents from data/raw/ into Qdrant Cloud
Run: python src/ingest.py
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from vectorstore import get_vectorstore
except ImportError:
    from src.vectorstore import get_vectorstore

load_dotenv()


def load_and_chunk(data_dir="data/raw", chunk_size=800, chunk_overlap=100):
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        print(f"No PDFs found in {data_dir}. Add some .pdf files there and re-run.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks):
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks in Qdrant Cloud")
    return vectorstore


if __name__ == "__main__":
    chunks = load_and_chunk()
    if chunks:
        embed_and_store(chunks)
    else:
        print("Nothing to embed — check data/raw for PDFs.")