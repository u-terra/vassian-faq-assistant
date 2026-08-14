"""Provider registry / factory.

build_provider() is the single place that maps AI_PROVIDER -> a concrete
AIProvider implementation. Add new providers here without touching app.py,
rag_index.py, memory.py or the API layer.
"""
from ..config import Settings
from .base import AIProvider


def build_provider(settings: Settings) -> AIProvider:
    provider_name = settings.ai_provider.strip().lower()

    if provider_name == "yandex":
        from .yandex import YandexAIProvider

        return YandexAIProvider(settings)

    raise ValueError(f"Unknown AI_PROVIDER: {settings.ai_provider!r}")
