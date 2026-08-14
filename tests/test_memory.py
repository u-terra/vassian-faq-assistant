import sqlite3
from pathlib import Path

from backend import memory


def _db_path(tmp_path: Path) -> str:
    return str(tmp_path / "assistant.db")


def test_init_db_creates_messages_table(tmp_path):
    db_path = _db_path(tmp_path)
    memory.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchall()
    assert tables


def test_add_and_get_history_preserves_chronological_order(tmp_path):
    db_path = _db_path(tmp_path)
    memory.init_db(db_path)
    session_id = "s1"

    memory.add_message(db_path, session_id, "user", "hello")
    memory.add_message(db_path, session_id, "assistant", "hi")
    memory.add_message(db_path, session_id, "user", "how are you")

    history = memory.get_history(db_path, session_id, limit=10)

    assert [m["content"] for m in history] == ["hello", "hi", "how are you"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]


def test_get_history_respects_limit_and_keeps_most_recent(tmp_path):
    db_path = _db_path(tmp_path)
    memory.init_db(db_path)
    session_id = "s1"
    for i in range(5):
        memory.add_message(db_path, session_id, "user", f"msg {i}")

    history = memory.get_history(db_path, session_id, limit=2)

    assert [m["content"] for m in history] == ["msg 3", "msg 4"]


def test_history_is_isolated_per_session(tmp_path):
    db_path = _db_path(tmp_path)
    memory.init_db(db_path)
    memory.add_message(db_path, "s1", "user", "from s1")
    memory.add_message(db_path, "s2", "user", "from s2")

    assert [m["content"] for m in memory.get_history(db_path, "s1", 10)] == ["from s1"]
    assert [m["content"] for m in memory.get_history(db_path, "s2", 10)] == ["from s2"]


def test_clear_session_removes_only_that_session(tmp_path):
    db_path = _db_path(tmp_path)
    memory.init_db(db_path)
    memory.add_message(db_path, "s1", "user", "hello")
    memory.add_message(db_path, "s2", "user", "keep me")

    memory.clear_session(db_path, "s1")

    assert memory.get_history(db_path, "s1", 10) == []
    assert [m["content"] for m in memory.get_history(db_path, "s2", 10)] == ["keep me"]
