"""Ingest TruthfulQA and a configurable SQuAD sample into ChromaDB.

Examples:
    python -m backend.scripts.ingest_benchmarks --max-squad-rows 500
    python -m backend.scripts.ingest_benchmarks --max-squad-rows 0  # full SQuAD train split
"""

import argparse
import hashlib
from typing import Any

from datasets import load_dataset

from backend.app.knowledge_base import get_collection

BATCH_SIZE = 128


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split text into overlapping character chunks while retaining context."""
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def stable_id(dataset_name: str, record_id: str, chunk_index: int) -> str:
    raw = f"{dataset_name}:{record_id}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def flush_batch(collection, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        ids.clear()
        documents.clear()
        metadatas.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Index TruthfulQA and SQuAD reference data.")
    parser.add_argument(
        "--max-squad-rows",
        type=int,
        default=500,
        help="Number of SQuAD training rows to index; use 0 for the complete split.",
    )
    args = parser.parse_args()
    if args.max_squad_rows < 0:
        parser.error("--max-squad-rows must be zero or a positive integer")

    collection = get_collection()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    print("Loading TruthfulQA generation split...")
    truthful = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    for row_index, row in enumerate(truthful):
        record_id = str(row_index)
        text = (
            f"Question: {row.get('question', '')}\n"
            f"Reference answer: {row.get('best_answer', '')}\n"
            f"Category: {row.get('category', '')}\n"
            f"Source: {row.get('source', '')}"
        )
        for chunk_index, chunk in enumerate(chunk_text(text)):
            ids.append(stable_id("truthfulqa", record_id, chunk_index))
            documents.append(chunk)
            metadatas.append(
                {
                    "dataset": "TruthfulQA",
                    "record_id": record_id,
                    "category": str(row.get("category", ""))[:200],
                    "source": str(row.get("source", ""))[:500],
                    "dataset_license": "Apache-2.0 (see dataset card)",
                }
            )
        if len(ids) >= BATCH_SIZE:
            flush_batch(collection, ids, documents, metadatas)

    print("Loading SQuAD training split...")
    squad = load_dataset("rajpurkar/squad", split="train")
    if args.max_squad_rows:
        squad = squad.select(range(min(args.max_squad_rows, len(squad))))

    for row_index, row in enumerate(squad):
        answers = row.get("answers", {}).get("text", [])
        answer = answers[0] if answers else ""
        context = row.get("context", "")
        question = row.get("question", "")
        context_chunks = chunk_text(context)
        for chunk_index, chunk in enumerate(context_chunks):
            document = (
                f"Question: {question}\n"
                f"Reference answer: {answer}\n"
                f"Passage: {chunk}"
            )
            ids.append(stable_id("squad", str(row.get("id", row_index)), chunk_index))
            documents.append(document)
            metadatas.append(
                {
                    "dataset": "SQuAD",
                    "record_id": str(row.get("id", row_index)),
                    "title": str(row.get("title", ""))[:200],
                    "dataset_license": "CC-BY-SA-4.0 (see dataset card)",
                }
            )
        if len(ids) >= BATCH_SIZE:
            flush_batch(collection, ids, documents, metadatas)

    flush_batch(collection, ids, documents, metadatas)
    print(f"Done. ChromaDB collection now contains {collection.count()} indexed chunks.")
    print("Note: retrieved content is candidate evidence, not a guaranteed correct answer.")


if __name__ == "__main__":
    main()
