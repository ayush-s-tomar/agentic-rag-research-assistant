"""
Phase 1 — Load, chunk, and embed documents from data/raw/ into Chroma
Run: python src/ingest.py
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
try:
    from hf_embeddings import HFInferenceEmbeddings
except ImportError:
    from src.hf_embeddings import HFInferenceEmbeddings

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


def embed_and_store(chunks, persist_dir="data/chroma_db"):
    embeddings = HFInferenceEmbeddings(api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Stored {len(chunks)} chunks in {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    chunks = load_and_chunk()
    if chunks:
        embed_and_store(chunks)
    else:
        print("Nothing to embed — check data/raw for PDFs.")