"""Build the FAISS RAG index from data/.

Two separate concerns, on purpose:
  - prepare_documents() / load_* helpers: pure, offline, testable without
    any AI provider - they just read data/ and normalize it. Existing
    content in data/ (faqs.json, *.txt, cases/*.txt) is read only, never
    modified.
  - build_and_save_index() / main(): the only parts that call an
    AIProvider (embeddings) and write the FAISS index + metadata to disk.

Importing this module builds nothing and calls no API. It must be run
explicitly:

    python -m backend.build_index

Stage 3 does NOT run this.
"""
import json
import os
from typing import Any, Dict, List

from .config import get_settings
from .providers import build_provider
from .providers.base import AIProvider

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_faq_documents(data_dir: str) -> List[Dict[str, Any]]:
    """Normalize data/faqs.json into {text, source, title, type} documents."""
    path = os.path.join(data_dir, "faqs.json")
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    documents = []
    for item in faqs:
        question = item["question"]
        answer = item["answer"]
        documents.append(
            {
                "text": f"{question}\n{answer}",
                "source": "faqs.json",
                "title": question,
                "type": "faq",
            }
        )
    return documents


def load_txt_documents(dir_path: str, doc_type: str) -> List[Dict[str, Any]]:
    """Normalize every *.txt file in dir_path into a {text, source, title, type} document."""
    documents = []
    if not os.path.isdir(dir_path):
        return documents

    for name in sorted(os.listdir(dir_path)):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(dir_path, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        lines = content.splitlines()
        title = next((line.strip() for line in lines if line.strip()), name)

        documents.append(
            {
                "text": content,
                "source": name,
                "title": title,
                "type": doc_type,
            }
        )
    return documents


def prepare_documents(data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """Collect and normalize all knowledge-base documents. No API calls, no writes."""
    documents: List[Dict[str, Any]] = []
    documents.extend(load_faq_documents(data_dir))
    documents.extend(load_txt_documents(data_dir, doc_type="profile"))
    documents.extend(load_txt_documents(os.path.join(data_dir, "cases"), doc_type="case"))
    return documents


def build_and_save_index(
    documents: List[Dict[str, Any]],
    provider: AIProvider,
    index_path: str,
    metadata_path: str,
) -> None:
    """Embed documents and write the FAISS index + metadata to disk.

    This is the only function here that calls the AI provider and touches
    the filesystem beyond reading data/.
    """
    import faiss
    import numpy as np

    if not documents:
        raise RuntimeError("No documents to index - data/ appears to be empty.")

    texts = [doc["text"] for doc in documents]
    vectors = provider.embed_texts(texts, text_type="doc")
    embeddings = np.array(vectors, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)

    metadata = np.array(documents, dtype=object)
    np.save(metadata_path, metadata)


def main() -> None:
    settings = get_settings()
    documents = prepare_documents(DATA_DIR)
    provider = build_provider(settings)
    build_and_save_index(documents, provider, settings.faiss_index_path, settings.faiss_metadata_path)
    print(f"Indexed {len(documents)} documents -> {settings.faiss_index_path}")


if __name__ == "__main__":
    main()
