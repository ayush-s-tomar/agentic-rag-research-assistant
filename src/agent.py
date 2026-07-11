"""
Phase 3b — Agent that decides which tool(s) to call, then answers.
Run directly: python src/agent.py "your question here"
"""
import sys
import os
import re
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessageChunk
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from src.tools.tools import retrieve_docs, screen_resume, route_query

load_dotenv()

# llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17 and shuts down
# 2026-08-16. openai/gpt-oss-120b is Groq's recommended replacement, keeps
# full tool-calling support (required for this agent), and has a higher
# free-tier daily token budget (200K TPD vs 100K TPD).
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
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

# gpt-oss-120b occasionally leaks raw tool-call citation markers like
# 【retrieve_docs】 into the final answer text. Strip these before returning
# anything to the user — they're an internal formatting artifact, not
# content meant to be displayed.
_CITATION_ARTIFACT_RE = re.compile(r"【[^】]*】")


def _clean(text: str) -> str:
    return _CITATION_ARTIFACT_RE.sub("", text).strip()


def _build_messages(question: str):
    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            {"role": "user", "content": question},
        ]
    }


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    reraise=True,
)
def _invoke_agent(question: str):
    return _agent.invoke(_build_messages(question))


def run_agent(question: str) -> str:
    """Run the agent on a question and return the full answer at once.

    Retries once on a transient rate-limit error (e.g. a short per-minute
    cap), then gives up and lets the caller (the /ask endpoint) handle it.
    A daily-cap rate limit (like the one that broke the demo earlier) has
    a multi-minute retry-after, so this won't paper over that — it'll
    still surface as an error, just after one quick retry instead of
    immediately.
    """
    try:
        result = _invoke_agent(question)
    except RateLimitError as e:
        return (
            "The model provider is currently rate-limited (this usually clears "
            f"within a few minutes on the free tier). Details: {e}"
        )
    final_message = result["messages"][-1]
    return _clean(final_message.content)


def stream_agent(question: str):
    """Yield the answer token-by-token as it's generated.

    Only yields text from the FINAL AI message (the answer), not intermediate
    tool-call chunks — those have no content to stream and would just yield
    empty strings. If a RateLimitError happens mid-stream, yields an error
    message and stops (no retry here, since partial output may already have
    been sent to the client).

    Citation artifacts like 【retrieve_docs】 are stripped per-token where
    possible, but since a marker can span multiple streamed chunks, a final
    cleanup pass also happens in the API layer on the fully assembled answer
    if needed. For most cases the per-chunk strip below is sufficient since
    the artifact usually arrives as a single contiguous chunk.
    """
    try:
        for chunk, metadata in _agent.stream(
            _build_messages(question),
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                if metadata.get("langgraph_node") == "agent":
                    cleaned = _clean(chunk.content)
                    if cleaned:
                        yield cleaned
    except RateLimitError as e:
        yield (
            "\n\n[The model provider is currently rate-limited. "
            f"Details: {e}]"
        )


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "your test question here"
    print(run_agent(q))