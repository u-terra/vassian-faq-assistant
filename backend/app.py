"""FastAPI application: GET /health, POST /chat.

Provider and settings are wired in via FastAPI dependencies
(get_settings_dep, get_ai_provider) specifically so tests can override them
with fakes via app.dependency_overrides - no real Yandex credentials or
FAISS index are required to exercise this module.
"""
import uuid
from typing import Any, Dict, List

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import memory, rag_index
from .config import Settings, get_settings
from .models import ChatRequest, ChatResponse
from .prompts import SYSTEM_PROMPT
from .providers import build_provider
from .providers.base import AIProvider

_default_settings = get_settings()

app = FastAPI(title="Vassian FAQ/RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_provider_cache: Dict[str, AIProvider] = {}

INTERNAL_INFO_REFUSAL = (
    "Я не раскрываю системные инструкции или внутренний RAG-контекст. "
    "Могу помочь с вопросами об услугах, проектах и формате работы."
)

_INTERNAL_TARGETS = (
    "системный промпт",
    "системного промпта",
    "system prompt",
    "внутренние инструкции",
    "внутренних инструкций",
    "internal instructions",
    "rag-контекст",
    "rag context",
    "контекст из базы знаний",
    "контекст базы знаний",
)

_DISCLOSURE_MARKERS = (
    "покажи",
    "выведи",
    "раскрой",
    "процитируй",
    "дословно",
    "напечатай",
    "show",
    "reveal",
    "print",
    "quote",
)


def get_settings_dep() -> Settings:
    return _default_settings


def get_ai_provider(settings: Settings = Depends(get_settings_dep)) -> AIProvider:
    if settings.ai_provider not in _provider_cache:
        _provider_cache[settings.ai_provider] = build_provider(settings)
    return _provider_cache[settings.ai_provider]


def _format_context(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "Контекст не найден."

    parts = []
    for item in items:
        title = item.get("title") or item.get("source") or ""
        text = item.get("text", "")
        parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts)


def _is_internal_info_request(message: str) -> bool:
    """Detect explicit requests to disclose hidden instructions/RAG context.

    Prompt instructions alone are not a reliable security boundary: a live
    evaluation showed the model could quote both the system prompt and the
    retrieved context despite being told not to. These narrow requests are
    therefore handled deterministically before retrieval/generation.
    """
    text = message.casefold()
    targets_internal = any(target in text for target in _INTERNAL_TARGETS)
    asks_to_disclose = any(marker in text for marker in _DISCLOSURE_MARKERS)
    return targets_internal and asks_to_disclose


def build_retrieval_query(current_message: str, history: List[Dict[str, str]]) -> str:
    """Text to embed for retrieval: current message + the last prior USER turn.

    Follow-ups like "А какие технологии там использовались?" lose their
    referent ("там") when embedded alone. Prepending the previous user
    question restores it without an extra LLM call or embedding call.
    Assistant replies are deliberately excluded - they're generated text,
    not a source retrieval should be driven by.
    """
    previous_user_message = None
    for item in reversed(history):
        if item["role"] == "user":
            previous_user_message = item["content"]
            break

    if previous_user_message is None:
        return current_message

    return f"{previous_user_message}\n{current_message}"


def _build_messages(
    context_text: str,
    history: List[Dict[str, str]],
    current_message: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nКонтекст базы знаний:\n{context_text}",
        }
    ]
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": current_message})
    return messages


@app.get("/health")
def health(settings: Settings = Depends(get_settings_dep)) -> dict:
    return {
        "status": "ok",
        "faiss_index_present": rag_index.index_exists(
            settings.faiss_index_path, settings.faiss_metadata_path
        ),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    settings: Settings = Depends(get_settings_dep),
    provider: AIProvider = Depends(get_ai_provider),
) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())

    memory.init_db(settings.database_path)

    if _is_internal_info_request(req.message):
        memory.add_message(settings.database_path, session_id, "user", req.message)
        memory.add_message(settings.database_path, session_id, "assistant", INTERNAL_INFO_REFUSAL)
        return ChatResponse(answer=INTERNAL_INFO_REFUSAL, session_id=session_id)

    # Fetch history BEFORE storing the current message, then append the
    # current message explicitly - avoids double-counting it in the prompt.
    history = memory.get_history(settings.database_path, session_id, limit=settings.max_history_messages)
    memory.add_message(settings.database_path, session_id, "user", req.message)

    context_items: List[Dict[str, Any]] = []
    if rag_index.index_exists(settings.faiss_index_path, settings.faiss_metadata_path):
        index, metadata = rag_index.load_index(settings.faiss_index_path, settings.faiss_metadata_path)
        retrieval_query = build_retrieval_query(req.message, history)
        query_vector = provider.embed_texts([retrieval_query], text_type="query")[0]
        context_items = rag_index.search(index, metadata, query_vector, k=settings.rag_top_k)

    context_text = _format_context(context_items)
    messages = _build_messages(context_text, history, req.message)

    answer = provider.generate(messages)

    memory.add_message(settings.database_path, session_id, "assistant", answer)

    return ChatResponse(answer=answer, session_id=session_id)
