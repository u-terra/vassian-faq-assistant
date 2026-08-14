import json
from pathlib import Path

import pytest

from backend import build_index, rag_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def test_load_faq_documents_matches_real_faqs_json():
    docs = build_index.load_faq_documents(str(DATA_DIR))

    with open(DATA_DIR / "faqs.json", encoding="utf-8") as f:
        raw = json.load(f)

    assert len(docs) == len(raw)
    for doc in docs:
        assert doc["type"] == "faq"
        assert doc["source"] == "faqs.json"
        assert doc["text"]
        assert doc["title"]


def test_load_txt_documents_reads_profile_files():
    docs = build_index.load_txt_documents(str(DATA_DIR), doc_type="profile")

    names = {doc["source"] for doc in docs}
    assert "about_me.txt" in names
    assert "services.txt" in names
    for doc in docs:
        assert doc["type"] == "profile"
        assert doc["text"]


def test_load_txt_documents_reads_case_files():
    docs = build_index.load_txt_documents(str(DATA_DIR / "cases"), doc_type="case")

    assert len(docs) >= 6
    for doc in docs:
        assert doc["type"] == "case"
        assert doc["text"]
        assert doc["source"].endswith(".txt")


def test_prepare_documents_combines_all_sources():
    docs = build_index.prepare_documents(str(DATA_DIR))

    types = {doc["type"] for doc in docs}
    assert types == {"faq", "profile", "case"}
    assert len(docs) == (
        len(build_index.load_faq_documents(str(DATA_DIR)))
        + len(build_index.load_txt_documents(str(DATA_DIR), "profile"))
        + len(build_index.load_txt_documents(str(DATA_DIR / "cases"), "case"))
    )


def test_load_index_missing_files_raises_controlled_error(tmp_path):
    with pytest.raises(rag_index.RAGIndexNotFoundError):
        rag_index.load_index(str(tmp_path / "missing.bin"), str(tmp_path / "missing.npy"))


def test_index_exists_false_when_files_missing(tmp_path):
    assert rag_index.index_exists(str(tmp_path / "a.bin"), str(tmp_path / "b.npy")) is False


def test_index_exists_true_when_both_files_present(tmp_path):
    index_path = tmp_path / "a.bin"
    metadata_path = tmp_path / "b.npy"
    index_path.write_bytes(b"x")
    metadata_path.write_bytes(b"x")

    assert rag_index.index_exists(str(index_path), str(metadata_path)) is True
