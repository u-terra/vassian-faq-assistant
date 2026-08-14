"""Provider-agnostic interface the rest of the backend depends on.

app.py, rag_index.py and memory.py must only ever talk to this interface,
never to a concrete SDK. That is what lets us swap Yandex for another
provider later without touching them.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Literal

EmbeddingTextType = Literal["query", "doc"]


class AIProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str], *, text_type: EmbeddingTextType) -> List[List[float]]:
        """Return one embedding vector per input text, in the same order.

        text_type distinguishes a user's search query from a knowledge-base
        document: some providers (Yandex included) use different embedding
        spaces for the two, so callers must be explicit rather than relying
        on a default that could silently mix them up.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a reply for a chat transcript.

        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        in chronological order. Returns the assistant's reply text.
        """
        raise NotImplementedError
