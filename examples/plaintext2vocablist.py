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
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    vocab = VocabPipeline().process(text)

    # Reading order; swap for .by_frequency() or .by_alpha() as needed.
    # Skip coverage gaps (no gloss), matching the default of to_markdown()/to_json().
    for entry in vocab.by_first_occurrence():
        if entry.has_gloss or not vocab.glosses_expected:
            print(entry.formatted())


if __name__ == "__main__":
    main()
