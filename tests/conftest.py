import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.providers.base import AIProvider, EmbeddingTextType  # noqa: E402


class FakeProvider(AIProvider):
    """Deterministic stand-in for a real AIProvider - no network, no API cost."""

    def __init__(self, embedding_dim: int = 4, reply: str = "fake answer"):
        self.embedding_dim = embedding_dim
        self.reply = reply
        self.generate_calls: List[List[Dict[str, str]]] = []
        self.embed_calls: List[Tuple[List[str], str]] = []

    def embed_texts(self, texts: List[str], *, text_type: EmbeddingTextType) -> List[List[float]]:
        self.embed_calls.append((texts, text_type))
        return [[float(len(t))] * self.embedding_dim for t in texts]

    def generate(self, messages: List[Dict[str, str]]) -> str:
        self.generate_calls.append(messages)
        return self.reply


@pytest.fixture
def make_fake_provider():
    def _make(reply: str = "fake answer", embedding_dim: int = 4) -> FakeProvider:
        return FakeProvider(embedding_dim=embedding_dim, reply=reply)

    return _make


@pytest.fixture
def fake_provider(make_fake_provider) -> FakeProvider:
    return make_fake_provider()
