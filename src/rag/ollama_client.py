"""Minimal client for a local Ollama server (embeddings + chat).

Ollama runs locally and exposes an HTTP API. Using it for both embeddings and
generation keeps the pipeline free, private, and dependency-light — no torch,
no cloud API keys. Running the LLM locally also satisfies the responsible-AI
requirement of not sending data to third-party LLM services.
"""

from __future__ import annotations

import requests


class OllamaError(RuntimeError):
    """Raised when the local Ollama server can't be reached or errors out."""


class OllamaClient:
    def __init__(self, host: str, embed_model: str, gen_model: str, timeout: int = 120):
        self.host = host.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start the server with 'ollama serve' and make sure you have "
                f"pulled the models (ollama pull {self.embed_model}; "
                f"ollama pull {self.gen_model})."
            ) from exc

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a piece of text."""
        data = self._post("/api/embeddings", {"model": self.embed_model, "prompt": text})
        return data["embedding"]

    def chat(self, system: str, user: str) -> str:
        """Return the assistant reply for a system+user prompt (non-streaming)."""
        data = self._post(
            "/api/chat",
            {
                "model": self.gen_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
        )
        return data["message"]["content"]
