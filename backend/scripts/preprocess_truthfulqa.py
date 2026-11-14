
"""
Preprocess the downloaded TruthfulQA benchmark dataset.

This script:
1. Reads the raw TruthfulQA JSON file.
2. Cleans unnecessary whitespace.
3. Standardizes field names.
4. Preserves correct and incorrect answers.
5. Preserves source URLs and original row indexes.
6. Saves the cleaned dataset as JSON.

No pandas or Hugging Face datasets dependency is required.
"""

import json
from pathlib import Path
from typing import Any


DATASET_ID = "domenicrosati/TruthfulQA"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "truthfulqa_raw.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "truthfulqa_clean.json"
)


def clean_text(value: Any) -> str:
    """Normalize whitespace without changing the meaning of text."""

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = value.replace("\u00a0", " ")

    return " ".join(value.split())


def split_answers(value: str) -> list[str]:
    """Convert semicolon-separated benchmark answers into a list."""

    if not value:
        return []

    return [
        answer
        for answer in (clean_text(item) for item in value.split(";"))
        if answer
    ]


def preprocess_record(raw_record: dict[str, Any]) -> dict[str, Any]:
    """Convert one original dataset record into the standard format."""

    row = raw_record.get("row", {})

    question = clean_text(row.get("Question"))
    best_answer = clean_text(row.get("Best Answer"))

    correct_answers_raw = clean_text(
        row.get("Correct Answers")
    )

    incorrect_answers_raw = clean_text(
        row.get("Incorrect Answers")
    )

    return {
        "record_id": f"truthfulqa-{raw_record.get('row_idx', 'unknown')}",
        "source_row_index": raw_record.get("row_idx"),
        "dataset": DATASET_ID,
        "type": clean_text(row.get("Type")),
        "category": clean_text(row.get("Category")),
        "question": question,
        "reference_answer": best_answer,
        "correct_answers": split_answers(correct_answers_raw),
        "correct_answers_raw": correct_answers_raw,
        "incorrect_answers": split_answers(incorrect_answers_raw),
        "incorrect_answers_raw": incorrect_answers_raw,
        "source_url": clean_text(row.get("Source")),
    }


def main() -> int:
    """Load, preprocess, validate and save TruthfulQA."""

    print("Starting TruthfulQA preprocessing...")
    print()

    if not INPUT_FILE.exists():
        print("ERROR: Raw TruthfulQA dataset file was not found.")
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

            # A usable benchmark record must contain both fields.
            if (
                not cleaned_record["question"]
                or not cleaned_record["reference_answer"]
            ):
                skipped_records += 1
                continue

            cleaned_records.append(cleaned_record)

        # Create the output directory if it does not exist.
        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed_data = {
            "dataset": DATASET_ID,
            "configuration": raw_data.get("config"),
            "split": raw_data.get("split"),
            "total_raw_records": len(raw_records),
            "processed_records": len(cleaned_records),
            "skipped_records": skipped_records,
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
        print("Preprocessing completed successfully!")
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
