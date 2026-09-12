"""Step 2 check: print Mac context. Use --demo to create a visible new note."""

from __future__ import annotations

import argparse

import computer


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify computer.py primitives")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create a new Apple Note with a test line",
    )
    args = parser.parse_args()

    print("Current context:")
    print(computer.dump_context())

    if not args.demo:
        print("\nSafe check only. Run with --demo to create a new Apple Note.")
        return

    created = computer.make_new_note("Hand Control step 2 — generic typing works.")
    print("\nmake_new_note:")
    print(created)
    print("\nContext after demo:")
    print(computer.dump_context())


if __name__ == "__main__":
    main()
