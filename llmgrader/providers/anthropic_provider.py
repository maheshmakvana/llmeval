"""Anthropic (Claude) provider."""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Usage:
        provider = AnthropicProvider(model="claude-sonnet-4-6")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._model_name = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package is required: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self._model_name,
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""
