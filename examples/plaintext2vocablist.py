"""Minimum viable plaintext -> Latin vocab list.

Usage:
    uv run python examples/plaintext2vocablist.py path/to/text.txt
    echo "agricolae in villa laborant" | uv run python examples/plaintext2vocablist.py

Requires: latincy-vocab installed, plus a LatinCy model (default la_core_web_lg).
"""

import sys

from vocabbuilder import VocabPipeline


def main() -> None:
    # Read plaintext from a file argument, or stdin if none given.
    text = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()

    vocab = VocabPipeline().process(text)

    # Reading order; swap for .by_frequency() or .by_alpha() as needed.
    for entry in vocab.by_first_occurrence():
        print(entry.formatted())


if __name__ == "__main__":
    main()
