"""Step 3: Grok plans computer.py tool calls from a text request."""

from __future__ import annotations

import argparse

from grok_agent import run_instruction

SAFE_DEFAULT = "What app is in front right now? Use get_context only. Do not press keys or type."
DEMO_PROMPT = (
    "Create a new Apple Note whose body is exactly: "
    "Hello from Grok Mac control. "
    "Use make_new_note. Do not use Spotlight. Do not press command+n."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Grok computer tools")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ask Grok to create a new Apple Note",
    )
    parser.add_argument("prompt", nargs="*", help="Text request for Grok")
    args = parser.parse_args()

    if args.demo:
        prompt = DEMO_PROMPT
    else:
        prompt = " ".join(args.prompt).strip() or SAFE_DEFAULT

    print(f"Request: {prompt}\n")
    try:
        reply = run_instruction(prompt)
    except Exception as exc:
        print(f"Grok tool loop failed: {exc}")
        raise SystemExit(1) from exc
    print(f"\nGrok said: {reply}")


if __name__ == "__main__":
    main()
