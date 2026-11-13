"""ChromaDB helper for storing and retrieving reference evidence."""

from pathlib import Path
from typing import Any

import chromadb

ROOT_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
COLLECTION_NAME = "reference_knowledge"


def get_collection():
    """Open a persistent collection; Chroma uses its default local embedding function."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def search_reference(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    matches: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "text": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )
    return matches
