"""Ollama local model provider."""

from __future__ import annotations

from typing import List, Optional

from .base import EmbeddingProvider, LLMProvider


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local models (Llama, Mistral, etc.)

    Usage:
        provider = OllamaProvider(model="llama3")
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self._model_name = model
        self._base_url = base_url
        self._temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx package is required: pip install httpx")

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", self._temperature)},
        }
        response = httpx.post(f"{self._base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url

    def embed(self, texts: List[str]) -> List[List[float]]:
        import httpx
        results = []
        for text in texts:
            payload = {"model": self._model, "prompt": text}
            response = httpx.post(f"{self._base_url}/api/embeddings", json=payload, timeout=60)
            response.raise_for_status()
            results.append(response.json()["embedding"])
        return results
