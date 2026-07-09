"""
Phase 4 — Run the agent over eval_set.jsonl and score with RAGAS.
Run from project root: python src/eval/run_eval.py
"""
import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import run_agent
from tools.tools import retrieve_docs

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


def build_dataset(eval_path="src/eval/eval_set.jsonl"):
    rows = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            answer = run_agent(question)
            contexts = [retrieve_docs.invoke({"query": question})]
            rows.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            })
    return rows


def main():
    rows = build_dataset()
    dataset = Dataset.from_list(rows)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    print(results)

    df = results.to_pandas()
    out_path = "src/eval/results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
