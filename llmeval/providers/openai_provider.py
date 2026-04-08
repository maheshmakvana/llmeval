"""OpenAI LLM and embedding provider."""

from __future__ import annotations

import os
from typing import List, Optional

from .base import EmbeddingProvider, LLMProvider


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider supporting GPT-4, GPT-3.5-turbo, etc.

    Usage:
        provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ) -> None:
        self._model_name = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required: pip install openai")
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        return response.choices[0].message.content or ""

    def generate_raw_response(self, prompt: str, **kwargs):
        client = self._get_client()
        return client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            logprobs=kwargs.get("logprobs", False),
        )


class AzureOpenAIProvider(LLMProvider):
    """Azure-hosted OpenAI provider."""

    def __init__(
        self,
        model: str,
        azure_endpoint: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._model_name = model
        self._azure_endpoint = azure_endpoint
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError("openai package is required: pip install openai")
            self._client = AzureOpenAI(
                azure_endpoint=self._azure_endpoint,
                api_key=self._api_key,
                api_version=self._api_version,
            )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        return response.choices[0].message.content or ""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding provider."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        response = client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
