"""Build the RAG index: PubMed -> chunks -> embeddings -> local vector store.

Usage:
    python -m src.rag.ingest --config configs/rag.yaml

Requires a running Ollama server with the configured embedding model pulled.
"""

from __future__ import annotations

import argparse

import yaml

from src.rag.ollama_client import OllamaClient
from src.rag.pubmed import fetch_abstracts, search_pubmed
from src.rag.store import VectorStore


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG index from PubMed.")
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"Searching PubMed for: {cfg['pubmed']['query']!r} ...")
    pmids = search_pubmed(
        cfg["pubmed"]["query"], cfg["pubmed"]["retmax"], cfg["pubmed"]["email"]
    )
    print(f"Found {len(pmids)} articles. Fetching abstracts ...")
    docs = fetch_abstracts(pmids, cfg["pubmed"]["email"])
    print(f"Fetched {len(docs)} abstracts with usable text.")

    client = OllamaClient(
        cfg["ollama"]["host"],
        cfg["ollama"]["embed_model"],
        cfg["ollama"]["gen_model"],
        cfg["ollama"]["timeout"],
    )
    store = VectorStore()
    size, overlap = cfg["chunking"]["chunk_size"], cfg["chunking"]["overlap"]

    n_chunks = 0
    for i, doc in enumerate(docs, 1):
        text = f"{doc['title']} {doc['abstract']}"
        for chunk_id, chunk in enumerate(chunk_text(text, size, overlap)):
            vector = client.embed(chunk)
            meta = {
                "pmid": doc["pmid"],
                "title": doc["title"],
                "journal": doc["journal"],
                "year": doc["year"],
                "chunk_id": chunk_id,
                "text": chunk,
            }
            store.add([vector], [meta])
            n_chunks += 1
        if i % 10 == 0:
            print(f"  embedded {i}/{len(docs)} articles ...")

    print(f"Embedded {n_chunks} chunks from {len(docs)} articles.")
    store.save(cfg["paths"]["index_file"], cfg["paths"]["docs_file"])
    print(f"Saved index to {cfg['paths']['index_file']}")


if __name__ == "__main__":
    main()
