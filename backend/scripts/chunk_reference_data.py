
"""
Create searchable reference chunks from TruthfulQA and SQuAD.

TruthfulQA:
- Uses the question, best answer and correct answers.
- Preserves category and source URL metadata.
- Does not index incorrect answers as reference evidence.

SQuAD:
- Splits original reference contexts into overlapping chunks.
- Preserves article title and question metadata.
- Tracks answer positions relative to each chunk.

Output:
data/processed/reference_chunks.json
"""

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRUTHFULQA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "truthfulqa_clean.json"
)

SQUAD_FILE = (
    PROJECT_ROOT / "data" / "processed" / "squad_clean.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "reference_chunks.json"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Divide text into overlapping character-based chunks.

    start and end are character offsets in the original text.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    if not isinstance(text, str) or not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))
        content = text[start:end]

        if content.strip():
            chunks.append({
                "start": start,
                "end": end,
                "text": content,
            })

        if end == len(text):
            break

        start = end - overlap

    return chunks


def load_records(file_path: Path) -> list[dict[str, Any]]:
    """Load and validate a processed benchmark JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("records")

    if not isinstance(records, list):
        raise ValueError(
            f"Invalid records list in {file_path.name}"
        )

    return records


def create_truthfulqa_chunks(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create reference chunks from TruthfulQA records."""

    result = []

    for record in records:

        question = record.get("question", "").strip()
        best_answer = record.get("reference_answer", "").strip()
        correct_answers = record.get("correct_answers", [])

        if not question or not best_answer:
            continue

        if not isinstance(correct_answers, list):
            correct_answers = []

        valid_answers = [
            answer.strip()
            for answer in correct_answers
            if isinstance(answer, str) and answer.strip()
        ]

        reference_text = (
            f"Question: {question}\n"
            f"Best Answer: {best_answer}"
        )

        if valid_answers:
            reference_text += (
                "\nOther Correct Answers: "
                + "; ".join(valid_answers)
            )

        text_chunks = chunk_text(reference_text)

        for index, chunk in enumerate(text_chunks):

            result.append({
                "chunk_id": (
                    f"truthfulqa-{record['record_id']}-{index}"
                ),
                "dataset": "TruthfulQA",
                "record_id": record["record_id"],
                "chunk_index": index,
                "text": chunk["text"],
                "metadata": {
                    "category": record.get("category", ""),
                    "type": record.get("type", ""),
                    "question": question,
                    "reference_answer": best_answer,
                    "source_url": record.get("source_url", ""),
                    "character_start": chunk["start"],
                    "character_end": chunk["end"],
                },
            })

    return result


def create_squad_chunks(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create overlapping reference chunks from SQuAD contexts."""

    result = []

    for record in records:

        context = record.get("context", "")

        if not isinstance(context, str) or not context.strip():
            continue

        text_chunks = chunk_text(context)

        answers = record.get("reference_answers", [])
        answer_positions = record.get("answer_start_positions", [])

        if not isinstance(answers, list):
            answers = []

        if not isinstance(answer_positions, list):
            answer_positions = []

        # Pair answers with their original context positions.
        answer_pairs = [
            (answer, position)
            for answer, position in zip(answers, answer_positions)
            if (
                isinstance(answer, str)
                and isinstance(position, int)
                and position >= 0
            )
        ]

        for index, chunk in enumerate(text_chunks):

            chunk_start = chunk["start"]
            chunk_end = chunk["end"]

            local_answer_spans = []

            for answer, answer_start in answer_pairs:

                answer_end = answer_start + len(answer)

                # Include only answers fully contained in this chunk.
                if (
                    answer_start >= chunk_start
                    and answer_end <= chunk_end
                ):
                    local_answer_spans.append({
                        "text": answer,
                        "start": answer_start - chunk_start,
                        "end": answer_end - chunk_start,
                    })

            result.append({
                "chunk_id": (
                    f"squad-{record['record_id']}-{index}"
                ),
                "dataset": "SQuAD",
                "record_id": record["record_id"],
                "chunk_index": index,
                "text": chunk["text"],
                "metadata": {
                    "title": record.get("title", ""),
                    "question": record.get("question", ""),
                    "reference_answers": answers,
                    "answer_spans_in_chunk": local_answer_spans,
                    "source": record.get("source", "SQuAD"),
                    "character_start": chunk_start,
                    "character_end": chunk_end,
                },
            })

    return result


def main() -> int:
    """Create and save chunks from both benchmark datasets."""

    print("Starting reference data chunking...")
    print("Chunk size:", CHUNK_SIZE, "characters")
    print("Chunk overlap:", CHUNK_OVERLAP, "characters")
    print()

    try:
        truthfulqa_records = load_records(TRUTHFULQA_FILE)
        squad_records = load_records(SQUAD_FILE)

        print("TruthfulQA records loaded:", len(truthfulqa_records))
        print("SQuAD records loaded:", len(squad_records))
        print()

        truthfulqa_chunks = create_truthfulqa_chunks(
            truthfulqa_records
        )

        squad_chunks = create_squad_chunks(
            squad_records
        )

        all_chunks = truthfulqa_chunks + squad_chunks

        # Check chunk IDs are unique.
        chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk IDs detected.")

        output_data = {
            "chunking_strategy": "Character-based overlapping chunking",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "total_chunks": len(all_chunks),
            "truthfulqa_chunks": len(truthfulqa_chunks),
            "squad_chunks": len(squad_chunks),
            "total_source_records": (
                len(truthfulqa_records) + len(squad_records)
            ),
            "chunks": all_chunks,
        }

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                output_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print("Chunking completed successfully!")
        print()
        print("TruthfulQA chunks:", len(truthfulqa_chunks))
        print("SQuAD chunks:", len(squad_chunks))
        print("Total chunks:", len(all_chunks))
        print("Unique chunk IDs:", len(set(chunk_ids)))
        print()
        print("Saved reference chunks at:")
        print(OUTPUT_FILE)

        return 0

    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        print("Chunking failed:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
