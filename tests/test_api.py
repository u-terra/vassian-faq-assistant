from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.app import build_retrieval_query
from backend.config import Settings


def _make_client(tmp_path: Path, provider) -> TestClient:
    settings = Settings(
        ai_provider="fake",
        database_path=str(tmp_path / "assistant.db"),
        faiss_index_path=str(tmp_path / "missing_index.bin"),
        faiss_metadata_path=str(tmp_path / "missing_meta.npy"),
        allowed_origins=["https://vassian.ru"],
        rag_top_k=4,
        max_history_messages=10,
    )
    app_module.app.dependency_overrides[app_module.get_settings_dep] = lambda: settings
    app_module.app.dependency_overrides[app_module.get_ai_provider] = lambda: provider
    return TestClient(app_module.app)


def _build_tiny_fake_index(tmp_path: Path, dim: int = 4):
    """A throwaway 2-vector FAISS index for testing app.py wiring only -
    not the production data/faiss_index.bin, no API calls involved."""
    import faiss
    import numpy as np

    index = faiss.IndexFlatL2(dim)
    index.add(np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype="float32"))

    metadata = np.array(
        [
            {"text": "dummy doc 1", "source": "dummy1.txt", "title": "Dummy 1", "type": "profile"},
            {"text": "dummy doc 2", "source": "dummy2.txt", "title": "Dummy 2", "type": "profile"},
        ],
        dtype=object,
    )

    index_path = tmp_path / "fake_index.bin"
    metadata_path = tmp_path / "fake_metadata.npy"
    faiss.write_index(index, str(index_path))
    np.save(str(metadata_path), metadata)
    return str(index_path), str(metadata_path)


def _make_client_with_index(tmp_path: Path, provider) -> TestClient:
    index_path, metadata_path = _build_tiny_fake_index(tmp_path)
    settings = Settings(
        ai_provider="fake",
        database_path=str(tmp_path / "assistant.db"),
        faiss_index_path=index_path,
        faiss_metadata_path=metadata_path,
        allowed_origins=["https://vassian.ru"],
        rag_top_k=2,
        max_history_messages=10,
    )
    app_module.app.dependency_overrides[app_module.get_settings_dep] = lambda: settings
    app_module.app.dependency_overrides[app_module.get_ai_provider] = lambda: provider
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app_module.app.dependency_overrides.clear()


def test_health_returns_ok_without_calling_provider(tmp_path, fake_provider):
    client = _make_client(tmp_path, fake_provider)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "faiss_index_present": False}
    assert fake_provider.generate_calls == []
    assert fake_provider.embed_calls == []


def test_chat_creates_new_session_id_when_absent(tmp_path, fake_provider):
    client = _make_client(tmp_path, fake_provider)

    response = client.post("/chat", json={"message": "Привет, что вы делаете?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_provider.reply
    assert body["session_id"]
    assert set(body.keys()) == {"answer", "session_id"}  # no raw RAG context leaked


def test_chat_reuses_provided_session_id(tmp_path, fake_provider):
    client = _make_client(tmp_path, fake_provider)

    response = client.post("/chat", json={"message": "Привет", "session_id": "my-session"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "my-session"


def test_chat_history_passed_to_generate_without_duplication(tmp_path, fake_provider):
    client = _make_client(tmp_path, fake_provider)
    session_id = "history-session"

    client.post("/chat", json={"message": "Первое сообщение", "session_id": session_id})
    client.post("/chat", json={"message": "Второе сообщение", "session_id": session_id})

    second_call_messages = fake_provider.generate_calls[1]
    contents = [m["content"] for m in second_call_messages]

    assert contents.count("Первое сообщение") == 1
    assert contents.count(fake_provider.reply) == 1
    assert contents.count("Второе сообщение") == 1
    assert contents[-1] == "Второе сообщение"


def test_chat_without_faiss_index_does_not_crash(tmp_path, fake_provider):
    client = _make_client(tmp_path, fake_provider)

    response = client.post("/chat", json={"message": "Сколько стоит бот?"})

    assert response.status_code == 200
    assert response.json()["answer"] == fake_provider.reply
    # no index on disk -> provider.embed_texts must not even be called
    assert fake_provider.embed_calls == []


def test_chat_missing_knowledge_is_handled_by_provider_not_by_a_crash(tmp_path, make_fake_provider):
    no_info_provider = make_fake_provider(
        reply="Точной информации по этому вопросу нет, уточните у Владимира напрямую."
    )
    client = _make_client(tmp_path, no_info_provider)

    response = client.post("/chat", json={"message": "Какая гарантия на проект?"})

    assert response.status_code == 200
    assert "нет" in response.json()["answer"].lower()


# --- build_retrieval_query (Stage 6B: conversational retrieval gap fix) ---


def test_build_retrieval_query_no_history_returns_only_current_message():
    query = build_retrieval_query("Расскажи кратко о проекте «Высота 63 м».", [])

    assert query == "Расскажи кратко о проекте «Высота 63 м»."


def test_build_retrieval_query_uses_previous_user_not_assistant():
    history = [
        {"role": "user", "content": "Расскажи кратко о проекте «Высота 63 м»."},
        {"role": "assistant", "content": "Это чат-бот для глэмпинга с квизом и AI-режимом."},
    ]

    query = build_retrieval_query("А какие технологии там использовались?", history)

    assert "Расскажи кратко о проекте «Высота 63 м»." in query
    assert "Это чат-бот для глэмпинга с квизом и AI-режимом." not in query


def test_build_retrieval_query_uses_only_last_previous_user_message():
    history = [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ"},
        {"role": "user", "content": "Второй вопрос"},
        {"role": "assistant", "content": "Второй ответ"},
    ]

    query = build_retrieval_query("Третий вопрос", history)

    assert "Второй вопрос" in query
    assert "Первый вопрос" not in query


def test_build_retrieval_query_does_not_duplicate_current_message():
    history = [
        {"role": "user", "content": "Предыдущий вопрос"},
        {"role": "assistant", "content": "Предыдущий ответ"},
    ]
    current = "Текущий вопрос"

    query = build_retrieval_query(current, history)

    assert query.count(current) == 1


def test_chat_sends_contextual_retrieval_query_not_raw_message(tmp_path, fake_provider):
    client = _make_client_with_index(tmp_path, fake_provider)
    session_id = "ctx-session"

    client.post(
        "/chat",
        json={"message": "Расскажи кратко о проекте «Высота 63 м».", "session_id": session_id},
    )
    client.post(
        "/chat",
        json={"message": "А какие технологии там использовались?", "session_id": session_id},
    )

    assert len(fake_provider.embed_calls) == 2

    first_texts, first_type = fake_provider.embed_calls[0]
    assert first_texts == ["Расскажи кратко о проекте «Высота 63 м»."]
    assert first_type == "query"

    second_texts, second_type = fake_provider.embed_calls[1]
    assert second_type == "query"
    assert len(second_texts) == 1
    retrieval_query = second_texts[0]
    # contextual query, not the raw current message alone
    assert retrieval_query != "А какие технологии там использовались?"
    assert "Расскажи кратко о проекте «Высота 63 м»." in retrieval_query
    assert "А какие технологии там использовались?" in retrieval_query
