"""AIProvider implementation backed by Yandex AI Studio.

Uses the official `yandex-ai-studio-sdk` package (module name
`yandex_ai_studio_sdk`, entry point `AIStudio`). See:
https://github.com/yandex-cloud/yandex-ai-studio-sdk

The SDK is imported lazily inside _get_client(), and the client is only
constructed on first real use (first embed_texts()/generate() call) - never
at module import time and never in __init__. This keeps `import
backend.providers.yandex` safe even when the SDK isn't installed and lets
the whole backend be imported/tested without any Yandex credentials.

Verified against the installed yandex-ai-studio-sdk 0.22.1 by static
introspection (class MRO, method signatures, source of message/result
dataclasses) - no network calls, no client construction. Confirmed:
  - AIStudio(folder_id=..., auth=...) matches our constructor call;
  - GPTModel.run() takes messages as {"role": ..., "text": ...} dicts
    (or a plain string), matching our generate();
  - the result is a Sequence of alternatives and result[0].text /
    list(result)[0].text both work;
  - TextEmbeddingsModelResult.run(text).embedding is the correct
    attribute for the embedding vector;
  - client.models.text_embeddings(model_name) resolves model_name
    through a well-known-names table {"doc": "text-search-doc",
    "query": "text-search-query"} - i.e. Yandex embeds queries and
    documents into different spaces, selected by name, not by a
    single configurable model. That is why embed_texts() below takes
    text_type instead of reading a single EMBEDDING_MODEL value: the
    SDK's own "doc"/"query" well-known names ARE the model selector for
    this pair, there is nothing else to configure here. EMBEDDING_MODEL
    is intentionally not read by this adapter for that reason - kept in
    config only in case a future custom/tuned embedding model needs it.
"""
from typing import Dict, List

from ..config import Settings
from .base import AIProvider, EmbeddingTextType


class YandexAIProvider(AIProvider):
    def __init__(self, settings: Settings):
        if not settings.yandex_folder_id or not settings.yandex_api_key:
            raise ValueError(
                "YANDEX_FOLDER_ID and YANDEX_API_KEY must be set to use YandexAIProvider"
            )
        if not settings.ai_model:
            raise ValueError("AI_MODEL must be set to use YandexAIProvider")

        self._settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            from yandex_ai_studio_sdk import AIStudio

            self._client = AIStudio(
                folder_id=self._settings.yandex_folder_id,
                auth=self._settings.yandex_api_key,
            )
        return self._client

    def embed_texts(self, texts: List[str], *, text_type: EmbeddingTextType) -> List[List[float]]:
        if text_type not in ("query", "doc"):
            raise ValueError(f"text_type must be 'query' or 'doc', got {text_type!r}")

        client = self._get_client()
        embedder = client.models.text_embeddings(text_type)  # well-known name, see module docstring
        return [list(embedder.run(text).embedding) for text in texts]

    def generate(self, messages: List[Dict[str, str]]) -> str:
        client = self._get_client()
        model = client.models.completions(self._settings.ai_model)
        yandex_messages = [
            {"role": message["role"], "text": message["content"]} for message in messages
        ]
        result = model.run(yandex_messages)
        alternatives = list(result)
        if not alternatives:
            raise RuntimeError("YandexAIProvider.generate: empty response from model")
        return alternatives[0].text
