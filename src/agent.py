"""
Phase 3b — Agent that decides which tool(s) to call, then answers.
Run directly: python -m src.agent "your question here"
"""
import sys
import os
import re
from dotenv import load_dotenv
from langchain.agents import create_agent
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

# create_react_agent (langgraph.prebuilt) is deprecated as of LangGraph v1.0
# and removed in v2.0 — migrated to langchain.agents.create_agent, its
# direct replacement, same call signature.
_agent = create_agent(llm, tools)

SYSTEM_PROMPT = """You are a document research assistant. You MUST use the retrieve_docs tool
to search the knowledge base before answering any factual question about the document(s).
Only answer from retrieved content. If retrieval returns no relevant information, say you
don't have that information in the provided documents — do not answer from your own
general knowledge, even for well-known facts."""

# gpt-oss-120b occasionally leaks raw tool-call citation markers like
# 【retrieve_docs】 or 【source】 into the final answer text. Strip these
# before returning anything to the user — they're an internal formatting
# artifact, not content meant to be displayed.
_CITATION_ARTIFACT_RE = re.compile(r"【[^】]*】")


def _clean(text: str) -> str:
    """Remove citation artifacts from a complete string. Does NOT strip
    whitespace — safe to call on a fully-assembled answer, but NOT on
    individual streamed token chunks (see stream_agent for why).
    """
    return _CITATION_ARTIFACT_RE.sub("", text)


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
    # Safe to strip here — this is the complete, final assembled answer,
    # not an individual streamed fragment.
    return _clean(final_message.content).strip()


def stream_agent(question: str):
    """Yield the answer token-by-token as it's generated.

    Only yields text from the FINAL AI message (the answer), not intermediate
    tool-call chunks. If a RateLimitError happens mid-stream, yields an
    error message and stops.

    Citation markers like 【source】 can be split across multiple streamed
    chunks (e.g. one chunk ends "...【sour" and the next starts "ce】...").
    A per-chunk regex can't catch that, so instead we buffer text: any
    chunk starting with a partial "【" marker is held back until either
    its closing "】" arrives (then the whole marker is dropped) or we
    confirm no marker is starting. Clean text with no open marker is
    flushed immediately, so streaming still feels responsive.
    """
    buffer = ""
    try:
        for chunk, metadata in _agent.stream(
            _build_messages(question),
            stream_mode="messages",
        ):
            if not (isinstance(chunk, AIMessageChunk) and chunk.content):
                continue
            # NOTE: previously filtered on metadata.get("langgraph_node") == "agent",
            # which matched langgraph.prebuilt.create_react_agent's internal node
            # name. After migrating to langchain.agents.create_agent (ahead of the
            # v2.0 removal), that node is named differently, so the old filter
            # silently dropped every chunk and stream_agent() yielded nothing —
            # the agent ran and answered, but nothing ever reached the UI. Tool-call
            # chunks don't come through as AIMessageChunk with .content in the same
            # way final-answer text does, so the isinstance+content check above is
            # sufficient on its own without pinning to a specific node name.
            if getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None):
                continue

            buffer += chunk.content

            while True:
                start = buffer.find("【")
                if start == -1:
                    if buffer:
                        yield buffer
                        buffer = ""
                    break

                if start > 0:
                    yield buffer[:start]
                    buffer = buffer[start:]

                end = buffer.find("】")
                if end == -1:
                    # Marker not closed yet — wait for more chunks
                    break

                # Complete marker found — drop it, keep scanning the
                # remainder in case there's more clean text or another
                # marker right after it
                buffer = buffer[end + 1:]

        # Flush anything left over at the end of the stream (e.g. a
        # stray "【" that never closed — just show it rather than
        # silently eating real content)
        if buffer:
            yield buffer

    except RateLimitError as e:
        yield (
            "\n\n[The model provider is currently rate-limited. "
            f"Details: {e}]"
        )


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "your test question here"
    print(run_agent(q))