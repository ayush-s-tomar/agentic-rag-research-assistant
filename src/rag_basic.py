"""
Phase 2c — Basic retrieve-then-generate loop, no agent/tools yet.
Run: python src/rag_basic.py "your question here"
"""
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
client = OpenAI()
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)


def answer(question, k=4):
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        f"Answer using only this context. If the answer isn't in the context, say so. "
        f"Cite which chunk you used.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "your test question here"
    print(answer(q))
