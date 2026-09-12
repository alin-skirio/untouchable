"""Step 1: confirm the xAI API key can reach Grok."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    key = os.getenv("XAI_API_KEY", "").strip()
    if not key:
        print("No XAI_API_KEY found. Paste your key into .env and run this again.")
        sys.exit(1)

    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
    try:
        response = client.responses.create(
            model="grok-4.6",
            input="Reply with exactly: ok",
        )
    except Exception as exc:
        print(f"xAI request failed: {exc}")
        sys.exit(1)

    text = (response.output_text or "").strip()
    print(f"Grok said: {text}")
    print("API key works. Next step is the computer-control primitives.")


if __name__ == "__main__":
    main()
