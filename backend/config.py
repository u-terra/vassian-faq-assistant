"""Application configuration, read from environment variables / .env.

Nothing in this module talks to any external service. Settings() only reads
strings from the environment; provider clients are constructed elsewhere,
lazily, on first real use.
"""
import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _split_origins(value: str) -> List[str]:
    return [origin.strip() for origin in value.split(",") if origin.strip()]


@dataclass
class Settings:
    ai_provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "yandex"))
    yandex_folder_id: str = field(default_factory=lambda: os.getenv("YANDEX_FOLDER_ID", ""))
    yandex_api_key: str = field(default_factory=lambda: os.getenv("YANDEX_API_KEY", ""))
    ai_model: str = field(default_factory=lambda: os.getenv("AI_MODEL", ""))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", ""))
    database_path: str = field(default_factory=lambda: os.getenv("DATABASE_PATH", "data/assistant.db"))
    faiss_index_path: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin"))
    faiss_metadata_path: str = field(
        default_factory=lambda: os.getenv("FAISS_METADATA_PATH", "data/faqs_metadata.npy")
    )
    allowed_origins: List[str] = field(
        default_factory=lambda: _split_origins(os.getenv("ALLOWED_ORIGINS", ""))
    )
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))
    max_history_messages: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES", "10")))


def get_settings() -> Settings:
    """Build a fresh Settings instance from the current environment.

    Intentionally not memoized: tests may set env vars per-case and expect
    a new Settings() to pick them up.
    """
    return Settings()
