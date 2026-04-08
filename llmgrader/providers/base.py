"""Base classes for LLM and embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class LLMProvider(ABC):
    """
    Base class for all LLM providers.
    Subclass this to wrap any LLM (OpenAI, Anthropic, Ollama, etc.)
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""

    def generate_raw_response(self, prompt: str, **kwargs):
        """
        Generate response and return the raw provider response object.
        Override to expose token probabilities, logprobs, etc.
        """
        return self.generate(prompt, **kwargs)

    @property
    def model_name(self) -> str:
        return getattr(self, "_model_name", "unknown")


class EmbeddingProvider(ABC):
    """Base class for embedding providers used in semantic similarity metrics."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings into vectors."""

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
