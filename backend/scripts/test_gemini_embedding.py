
"""
Test Gemini Embedding 2 with one sample sentence.

Uses the Gemini REST API and the API key stored in .env.
Does not print or expose the API key.
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

MODEL = "gemini-embedding-2"

API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL}:embedContent"
)

SAMPLE_TEXT = (
    "The capital of France is Paris."
)


def main() -> int:
    """Generate one test embedding and validate the response."""

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        print("ERROR: GEMINI_API_KEY was not found in .env.")
        return 1

    payload = {
        "content": {
            "parts": [
                {"text": SAMPLE_TEXT}
            ]
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    print("Starting Gemini embedding test...")
    print("Model:", MODEL)
    print("Sample text:", SAMPLE_TEXT)
    print()

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        print("HTTP Status:", response.status_code)

        response.raise_for_status()

        result = response.json()

        embedding = result.get("embedding", {})
        values = embedding.get("values", [])

        if not isinstance(values, list) or not values:
            print("ERROR: No embedding vector returned.")
            return 1

        print("Embedding generated successfully!")
        print("Embedding dimensions:", len(values))
        print(
            "First five vector values:",
            [round(float(value), 6) for value in values[:5]],
        )
        print()
        print("Gemini API connection test passed!")

        return 0

    except requests.HTTPError:
        print("Gemini API returned an HTTP error.")

        try:
            error_data = response.json()
            print(
                "API error:",
                error_data.get("error", {}).get(
                    "message", "Unknown API error."
                ),
            )
        except ValueError:
            print("Could not read the API error response.")

        return 1

    except requests.RequestException as error:
        print("Network error:", error)
        return 1

    except (ValueError, TypeError, AttributeError) as error:
        print("Unexpected response format:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
