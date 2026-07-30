"""Evaluate the RAG assistant with a groundedness (faithfulness) score.

Groundedness asks: is the generated answer actually supported by the retrieved
sources? We use the local LLM as a judge to score 0-1 — a common lightweight
approach when there is no labelled ground truth. This maps to the "model
evaluation" requirement.

Usage:
    python -m src.rag.evaluate --config configs/rag.yaml
"""

from __future__ import annotations

import argparse
import re

from src.rag.ask import answer, load_config
from src.rag.ollama_client import OllamaClient

JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a question, an answer, and the sources "
    "used, rate from 0.0 to 1.0 how well the answer is supported by (grounded "
    "in) the sources. 1.0 means fully supported, 0.0 means unsupported or "
    "hallucinated. Reply with ONLY the number."
)


def groundedness_score(client: OllamaClient, question, answer_text, hits) -> float | None:
    sources = "\n".join(doc["text"] for doc, _ in hits)
    user = f"Question: {question}\n\nAnswer: {answer_text}\n\nSources:\n{sources}"
    raw = client.chat(JUDGE_SYSTEM, user)
    match = re.search(r"[01](?:\.\d+)?", raw)
    return float(match.group()) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG groundedness.")
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    client = OllamaClient(
        cfg["ollama"]["host"],
        cfg["ollama"]["embed_model"],
        cfg["ollama"]["gen_model"],
        cfg["ollama"]["timeout"],
    )

    scores = []
    print("Evaluating groundedness on the configured questions ...\n")
    for question in cfg["evaluation"]["questions"]:
        reply, hits = answer(question, cfg)
        score = groundedness_score(client, question, reply, hits)
        scores.append(score)
        print(f"[{score}] {question}")

    valid = [s for s in scores if s is not None]
    if valid:
        print(f"\nMean groundedness: {sum(valid) / len(valid):.2f} over {len(valid)} questions")


if __name__ == "__main__":
    main()
