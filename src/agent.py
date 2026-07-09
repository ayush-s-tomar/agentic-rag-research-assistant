"""
Phase 3b — Agent that decides which tool(s) to call, then answers.
Run directly: python src/agent.py "your question here"
"""
import sys
import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from src.tools.tools import retrieve_docs, screen_resume, route_query

load_dotenv()

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
tools = [retrieve_docs, screen_resume, route_query]

_agent = create_react_agent(llm, tools)

SYSTEM_PROMPT = """You are a document research assistant. You MUST use the retrieve_docs tool
to search the knowledge base before answering any factual question about the document(s).
Only answer from retrieved content. If retrieval returns no relevant information, say you
don't have that information in the provided documents — do not answer from your own
general knowledge, even for well-known facts."""


def run_agent(question: str) -> str:
    result = _agent.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            {"role": "user", "content": question},
        ]
    })
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "your test question here"
    print(run_agent(q))