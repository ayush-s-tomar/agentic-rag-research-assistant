"""
Phase 2b — Sanity-check retrieval quality before adding the agent layer.
Run: python src/retrieve_test.py "your question here"
"""
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "your test question here"

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

    results = db.similarity_search(query, k=4)
    if not results:
        print("No results. Did you run embed_store.py first?")
        return

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(r.page_content[:300])
        print()


if __name__ == "__main__":
    main()
