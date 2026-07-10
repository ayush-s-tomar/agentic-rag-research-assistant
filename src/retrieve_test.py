"""
Phase 2b — Sanity-check retrieval quality before adding the agent layer.
Run: python src/retrieve_test.py "your question here"
"""
import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
try:
    from hf_embeddings import HFInferenceEmbeddings
except ImportError:
    from src.hf_embeddings import HFInferenceEmbeddings

load_dotenv()


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "your test question here"

    embeddings = HFInferenceEmbeddings(api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

    results = db.similarity_search(query, k=4)
    if not results:
        print("No results. Did you run ingest.py first?")
        return

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(r.page_content[:300])
        print()


if __name__ == "__main__":
    main()