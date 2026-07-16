"""
Phase 4c — Compare $/query across the fine-tuned-only, large-model-only, and
routed hybrid paths, using ACTUAL traffic from src/eval/request_log.csv where
available, falling back to a labeled projection when there isn't enough data yet.

Run: python src/eval/cost_comparison.py
"""
import os
import csv

# Reference prices (USD per 1K tokens) — update before publishing results.
# llama-3.1-8b-instant and gpt-oss-120b are Groq-hosted; pricing changes,
# so treat these as of the date you last checked https://groq.com/pricing.
PRICES_PER_1K_TOKENS = {
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "openai/gpt-oss-120b": {"input": 0.00015, "output": 0.0006},
}

# Used only as a fallback when request_log.csv doesn't have enough rows yet.
AVG_INPUT_TOKENS = 400
AVG_OUTPUT_TOKENS = 150
FALLBACK_HYBRID_RATIO_SIMPLE = 0.7  # explicitly labeled as an assumption below

LOG_PATH = "src/eval/request_log.csv"


def cost_per_query(model: str, input_tokens: float, output_tokens: float) -> float:
    p = PRICES_PER_1K_TOKENS[model]
    return (input_tokens / 1000) * p["input"] + (output_tokens / 1000) * p["output"]


def _load_routed_traffic():
    """Read request_log.csv and count how many requests actually went through
    route_query, and to which model. Returns None if the log doesn't exist or
    has no routed traffic yet, so the caller can fall back to a labeled
    projection instead of silently making up a ratio.
    """
    if not os.path.exists(LOG_PATH):
        return None

    simple_count = 0
    complex_count = 0
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answer = row.get("answer", "")
            if "complexity=simple" in answer:
                simple_count += 1
            elif "complexity=complex" in answer:
                complex_count += 1

    total = simple_count + complex_count
    if total == 0:
        return None
    return simple_count / total, total


def main():
    print("Estimated cost per query (fixed avg token assumptions):")
    for model in PRICES_PER_1K_TOKENS:
        c = cost_per_query(model, AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS)
        print(f"  {model}: ${c:.6f}  (${c * 1000:.2f} per 1000 queries)")

    routed = _load_routed_traffic()
    if routed is None:
        ratio_simple = FALLBACK_HYBRID_RATIO_SIMPLE
        source_note = (
            f"PROJECTED — no routed traffic found in {LOG_PATH} yet. "
            f"Using an assumed {int(ratio_simple * 100)}% simple / "
            f"{int((1 - ratio_simple) * 100)}% complex split."
        )
    else:
        ratio_simple, total = routed
        source_note = (
            f"MEASURED — based on {total} routed requests logged in {LOG_PATH}."
        )

    hybrid_cost = (
        ratio_simple * cost_per_query("llama-3.1-8b-instant", AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS)
        + (1 - ratio_simple) * cost_per_query("openai/gpt-oss-120b", AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS)
    )

    print(f"\nHybrid router ({int(ratio_simple * 100)}% simple / "
          f"{int((1 - ratio_simple) * 100)}% complex): "
          f"${hybrid_cost:.6f} per query (${hybrid_cost * 1000:.2f} per 1000 queries)")
    print(f"[{source_note}]")


if __name__ == "__main__":
    main()