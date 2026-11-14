
"""
Download a pilot sample of the SQuAD training dataset from Hugging Face.

Uses the Hugging Face Dataset Viewer API and requests.
Does not require pandas or the Hugging Face datasets library.
"""

import json
from pathlib import Path

import requests


DATASET_ID = "rajpurkar/squad"
CONFIG = "plain_text"
SPLIT = "train"

API_URL = "https://datasets-server.huggingface.co/rows"

BATCH_SIZE = 100
MAX_RECORDS = 500
TIMEOUT = 60


def main() -> int:
    """Download, validate and save a SQuAD pilot dataset."""

    project_root = Path(__file__).resolve().parents[2]

    output_file = (
        project_root / "data" / "raw" / "squad_raw.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    downloaded_rows = []
    total_rows = None
    offset = 0

    session = requests.Session()
    session.headers.update({
        "User-Agent": "AI-Response-Validation-Internship/0.1"
    })

    print("Starting SQuAD download...")
    print("Dataset:", DATASET_ID)
    print("Configuration:", CONFIG)
    print("Split:", SPLIT)
    print("Maximum records:", MAX_RECORDS)
    print()

    try:
        while len(downloaded_rows) < MAX_RECORDS:

            remaining = MAX_RECORDS - len(downloaded_rows)
            batch_length = min(BATCH_SIZE, remaining)

            params = {
                "dataset": DATASET_ID,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": batch_length,
            }

            response = session.get(
                API_URL,
                params=params,
                timeout=TIMEOUT,
            )
            response.raise_for_status()

            payload = response.json()

            if total_rows is None:
                total_rows = payload.get("num_rows_total")

                if not isinstance(total_rows, int):
                    raise RuntimeError(
                        "Could not determine the total SQuAD row count."
                    )

                print("Total training records available:", total_rows)
                print()

            batch = payload.get("rows", [])

            if not batch:
                raise RuntimeError(
                    f"No records returned at offset {offset}."
                )

            downloaded_rows.extend(batch)
            offset += len(batch)

            print(
                f"Downloaded {len(downloaded_rows)} "
                f"of {MAX_RECORDS} requested records",
                flush=True,
            )

        if len(downloaded_rows) != MAX_RECORDS:
            raise RuntimeError(
                "Downloaded record count does not match "
                "the requested pilot size."
            )

        dataset_output = {
            "dataset": DATASET_ID,
            "config": CONFIG,
            "split": SPLIT,
            "total_rows_available": total_rows,
            "downloaded_rows": len(downloaded_rows),
            "sample_size": MAX_RECORDS,
            "records": downloaded_rows,
        }

        output_file.write_text(
            json.dumps(
                dataset_output,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("SQuAD download completed successfully!")
        print("Total records downloaded:", len(downloaded_rows))
        print("Saved file:", output_file)

    except requests.RequestException as error:
        print("Network or Hugging Face API error:", error)
        return 1

    except (ValueError, RuntimeError) as error:
        print("Dataset processing error:", error)
        return 1

    finally:
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
