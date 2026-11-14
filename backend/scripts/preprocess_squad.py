
"""
Preprocess the downloaded SQuAD benchmark dataset.

This script:
1. Reads the raw SQuAD JSON file.
2. Cleans question and title text.
3. Preserves the original reference context.
4. Extracts answer texts and answer positions.
5. Standardizes the data structure.
6. Saves the processed dataset as JSON.

Important:
The original context is preserved exactly because SQuAD answer_start
positions refer to character positions in that original context.
"""

import json
from pathlib import Path
from typing import Any


DATASET_ID = "rajpurkar/squad"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "squad_raw.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "squad_clean.json"
)


def clean_text(value: Any) -> str:
    """Normalize whitespace in question and title text."""

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    return " ".join(value.split())


def preprocess_record(
    raw_record: dict[str, Any],
) -> dict[str, Any]:
    """Convert one original SQuAD record into standard format."""

    row = raw_record.get("row", {})
    answers = row.get("answers", {})

    if not isinstance(answers, dict):
        answers = {}

    context = row.get("context")

    if not isinstance(context, str):
        context = ""

    answer_texts = answers.get("text", [])
    answer_starts = answers.get("answer_start", [])

    if not isinstance(answer_texts, list):
        answer_texts = []

    if not isinstance(answer_starts, list):
        answer_starts = []

    # Preserve answer text and positions without altering their offsets.
    answer_texts = [
        answer for answer in answer_texts
        if isinstance(answer, str) and answer.strip()
    ]

    answer_starts = [
        position for position in answer_starts
        if isinstance(position, int) and position >= 0
    ]

    return {
        "record_id": clean_text(row.get("id")),
        "source_row_index": raw_record.get("row_idx"),
        "dataset": DATASET_ID,
        "title": clean_text(row.get("title")),
        "question": clean_text(row.get("question")),

        # Do not normalize context: answer_start depends on exact positions.
        "context": context,

        "reference_answers": answer_texts,
        "answer_start_positions": answer_starts,

        "source": "SQuAD training benchmark",
    }


def main() -> int:
    """Load, preprocess, validate and save SQuAD."""

    print("Starting SQuAD preprocessing...")
    print()

    if not INPUT_FILE.exists():
        print("ERROR: Raw SQuAD dataset file was not found.")
        print("Expected location:", INPUT_FILE)
        return 1

    try:
        with INPUT_FILE.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)

        raw_records = raw_data.get("records")

        if not isinstance(raw_records, list):
            raise ValueError(
                "The raw dataset does not contain a valid records list."
            )

        print("Total raw records:", len(raw_records))

        cleaned_records = []
        skipped_records = 0

        for raw_record in raw_records:

            if not isinstance(raw_record, dict):
                skipped_records += 1
                continue

            cleaned_record = preprocess_record(raw_record)

            # Each usable record needs a question, context and answer.
            if (
                not cleaned_record["record_id"]
                or not cleaned_record["question"]
                or not cleaned_record["context"]
                or not cleaned_record["reference_answers"]
            ):
                skipped_records += 1
                continue

            cleaned_records.append(cleaned_record)

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed_data = {
            "dataset": DATASET_ID,
            "configuration": raw_data.get("config"),
            "split": raw_data.get("split"),
            "total_rows_available": raw_data.get(
                "total_rows_available"
            ),
            "total_raw_records": len(raw_records),
            "processed_records": len(cleaned_records),
            "skipped_records": skipped_records,
            "sample_size": len(cleaned_records),
            "records": cleaned_records,
        }

        with OUTPUT_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                processed_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print("SQuAD preprocessing completed successfully!")
        print("Raw records:", len(raw_records))
        print("Processed records:", len(cleaned_records))
        print("Skipped records:", skipped_records)
        print()
        print("Cleaned dataset saved at:")
        print(OUTPUT_FILE)

        return 0

    except (OSError, json.JSONDecodeError, ValueError) as error:
        print("Preprocessing failed:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
