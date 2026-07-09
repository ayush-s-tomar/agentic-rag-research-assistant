"""
Phase 4c — Compare $/query across fine-tuned-only, GPT-4o-mini-only, and your routed hybrid.
Fill in real prices before presenting these numbers.
Run: python src/eval/cost_comparison.py
"""

# Rough reference prices (USD) — update with current pricing before publishing results.
PRICES_PER_1K_TOKENS = {
    "fine-tuned-qwen2.5-0.5b": {"input": 0.0, "output": 0.0},  # self-hosted, ~electricity/compute only
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

AVG_INPUT_TOKENS = 400
AVG_OUTPUT_TOKENS = 150


def cost_per_query(model: str) -> float:
    p = PRICES_PER_1K_TOKENS[model]
    return (AVG_INPUT_TOKENS / 1000) * p["input"] + (AVG_OUTPUT_TOKENS / 1000) * p["output"]


def main():
    print("Estimated cost per query:")
    for model in PRICES_PER_1K_TOKENS:
        c = cost_per_query(model)
        print(f"  {model}: ${c:.6f}  (${c*1000:.2f} per 1000 queries)")

    # Example hybrid: 70% of queries routed to the free fine-tuned model, 30% to GPT-4o-mini
    hybrid_ratio_finetuned = 0.7
    hybrid_cost = (
        hybrid_ratio_finetuned * cost_per_query("fine-tuned-qwen2.5-0.5b")
        + (1 - hybrid_ratio_finetuned) * cost_per_query("gpt-4o-mini")
    )
    print(f"\nHybrid router ({int(hybrid_ratio_finetuned*100)}% fine-tuned / "
          f"{int((1-hybrid_ratio_finetuned)*100)}% GPT-4o-mini): "
          f"${hybrid_cost:.6f} per query (${hybrid_cost*1000:.2f} per 1000 queries)")


if __name__ == "__main__":
    main()
