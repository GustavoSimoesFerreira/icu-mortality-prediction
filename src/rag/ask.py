"""Answer a clinical question from the PubMed corpus, with source citations.

Usage:
    python -m src.rag.ask --config configs/rag.yaml --question "What predicts ICU mortality?"

Requires a running Ollama server with the configured models pulled, and an
index already built by src.rag.ingest.
"""

from __future__ import annotations

import argparse

import yaml

from src.rag.ollama_client import OllamaClient
from src.rag.store import VectorStore

SYSTEM = (
    "You are a careful clinical research assistant. Answer ONLY using the "
    "provided sources. Cite them inline as [1], [2] matching the source numbers. "
    "If the sources do not contain the answer, say so plainly. Never invent facts "
    "or citations."
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(question: str, hits: list[tuple[dict, float]]) -> str:
    lines = ["Sources:"]
    for i, (doc, _) in enumerate(hits, 1):
        lines.append(
            f"[{i}] {doc['title']} ({doc['journal']}, {doc['year']}; PMID {doc['pmid']})\n"
            f"{doc['text']}\n"
        )
    lines.append(f"\nQuestion: {question}")
    return "\n".join(lines)


def answer(question: str, cfg: dict) -> tuple[str, list[tuple[dict, float]]]:
    store = VectorStore.load(cfg["paths"]["index_file"], cfg["paths"]["docs_file"])
    client = OllamaClient(
        cfg["ollama"]["host"],
        cfg["ollama"]["embed_model"],
        cfg["ollama"]["gen_model"],
        cfg["ollama"]["timeout"],
    )
    query_vector = client.embed(question)
    hits = store.search(query_vector, k=cfg["retrieval"]["top_k"])
    reply = client.chat(SYSTEM, build_prompt(question, hits))
    return reply, hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG assistant a question.")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    reply, hits = answer(args.question, cfg)
    print("\n=== ANSWER ===\n")
    print(reply)
    print("\n=== SOURCES USED ===")
    for i, (doc, score) in enumerate(hits, 1):
        print(f"[{i}] {doc['title']} (PMID {doc['pmid']}) — similarity {score:.3f}")


if __name__ == "__main__":
    main()
