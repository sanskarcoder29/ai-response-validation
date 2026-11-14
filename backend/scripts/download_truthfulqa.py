
"""
Download the TruthfulQA benchmark dataset from Hugging Face.

This script uses the Hugging Face Dataset Viewer API and requests.
It does not require pandas or the Hugging Face datasets library.
"""

import json
from pathlib import Path

import requests


DATASET_ID = "domenicrosati/TruthfulQA"
CONFIG = "default"
SPLIT = "train"

API_URL = "https://datasets-server.huggingface.co/rows"

BATCH_SIZE = 100
TIMEOUT = 60


def main() -> int:
    # Find the project root directory.
    project_root = Path(__file__).resolve().parents[2]

    # Save downloaded data inside data/raw.
    output_file = (
        project_root / "data" / "raw" / "truthfulqa_raw.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    downloaded_rows = []
    total_rows = None
    offset = 0

    session = requests.Session()

    session.headers.update({
        "User-Agent": "AI-Response-Validation-Internship/0.1"
    })

    print("Starting TruthfulQA download...")
    print("Dataset:", DATASET_ID)
    print("Configuration:", CONFIG)
    print("Split:", SPLIT)
    print()

    try:
        while total_rows is None or offset < total_rows:

            params = {
                "dataset": DATASET_ID,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": BATCH_SIZE,
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
                        "Could not determine the dataset row count."
                    )

                print("Total records available:", total_rows)
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
                f"of {total_rows} records",
                flush=True,
            )

        if len(downloaded_rows) != total_rows:
            raise RuntimeError(
                "Downloaded record count does not match "
                "the dataset's reported row count."
            )

        dataset_output = {
            "dataset": DATASET_ID,
            "config": CONFIG,
            "split": SPLIT,
            "total_rows": total_rows,
            "downloaded_rows": len(downloaded_rows),
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
        print("TruthfulQA download completed successfully!")
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
