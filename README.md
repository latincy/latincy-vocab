<img src="https://raw.githubusercontent.com/latincy/latincy-vocab/main/assets/latincy-vocab-logo.jpg" alt="LatinCy Vocab" width="400">

[![PyPI version](https://img.shields.io/pypi/v/latincy-vocab.svg)](https://pypi.org/project/latincy-vocab/)
[![Python versions](https://img.shields.io/pypi/pyversions/latincy-vocab.svg)](https://pypi.org/project/latincy-vocab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Latin vocabulary list builder powered by [LatinCy](https://github.com/latincy) models/tools/datasets.**

`latincy-vocab` takes Latin text, runs it through a LatinCy spaCy model, and returns structured vocabulary lists. Citation forms (principal parts, gender, etc.), POS markers, and dictionary glosses are sourced from [`latincy-lexicon`](https://github.com/latincy/latincy-lexicon) (Whitaker's Words); latincy-vocab is the formatting and aggregation layer over it. Output can be sorted by frequency, reading order, or alphabetically, and exported as JSON or Markdown.

> **Beta release (v0.2.0).** The API is functional but may change.

## Installation

```bash
pip install latincy-vocab
```

You also need a LatinCy spaCy model. `latincy-vocab` defaults to `la_core_web_lg` (best accuracy for citation forms and lemmatization):

```bash
pip install "https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.9.6-py3-none-any.whl"
```

A lighter `la_core_web_sm` is also available (swap `lg`→`sm` in the URL); set `PipelineConfig(spacy_model="la_core_web_sm")` to use it.

Glosses and citation forms come from [`latincy-lexicon`](https://github.com/latincy/latincy-lexicon) (Whitaker's Words), which is installed automatically as a dependency — no extra data files, environment variables, or sibling directories required.

## Usage

```python
from vocabbuilder import VocabPipeline

pipeline = VocabPipeline()

text = "agricolae in villa laborant et aquam portant"
vocab = pipeline.process(text)

# Reading order (first occurrence)
for entry in vocab.by_first_occurrence():
    print(entry.formatted())
```

Example output:

```
agricola, agricolae, m., farmer, cultivator, gardener, agriculturist
in, prep., in, on, at (space)
villa, villae, f., farm/country home/estate
laboro, laborare, laboravi, laboratum, v., work, labor
et, conj., and, and even
aqua, aquae, f., water
porto, portare, portavi, portatum, v., carry, bring
```

### Sorting

```python
vocab.by_frequency()       # most common first
vocab.by_alpha()           # alphabetical by lemma
vocab.by_first_occurrence() # reading order
```

### Filtering

```python
vocab.filter_pos({"NOUN", "VERB"})   # nouns and verbs only
vocab.filter_min_frequency(2)         # words appearing ≥ 2 times
```

**Static word lists.** Drop lemmas already covered by a reference list (e.g. the
DCC Latin Core Vocabulary), or keep only lemmas on a list. Matching folds u/v/j
and case, so lists in any orthography work:

```python
vocab.filter_lemmas(exclude=dcc_core)   # keep only what a passage adds beyond the core
vocab.filter_lemmas(keep=my_wordlist)   # keep only lemmas on my list
```

**Keyness measures.** Keep the words distinctive to a passage and drop corpus-wide
function words, using a keyness weight per lemma. `filter_keyness` accepts whatever
representation you already have — a plain `{lemma: weight}` dict, a pandas Series or
DataFrame, an `(lemma, weight)` iterable, or a scipy-sparse / numpy document-term
matrix straight from scikit-learn's `TfidfVectorizer` (no pandas/scipy/sklearn
dependency is added — they're duck-typed). A lemma absent from the weights counts as 0:

```python
# The DH-standard path: a sparse DTM from TfidfVectorizer.
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
dtm = tfidf.fit_transform(corpus)            # scipy sparse, (n_docs × n_terms)
names = tfidf.get_feature_names_out()
# Keep the 20 most distinctive words of document i:
vocab.filter_keyness(dtm, feature_names=names, document=i, top_n=20)

# A pandas DTM row (index = terms) needs no feature_names:
vocab.filter_keyness(dtm_df.loc["ep_6.16"], min_score=0.1)

# …or a plain mapping you built yourself:
vocab.filter_keyness({"pumex": 0.7, "ut": 0.01}, min_score=0.1)
```

### Export

```python
# Markdown glossary
print(vocab.to_markdown())

# JSON (all fields)
print(vocab.to_json())
```

### Entry fields

Each `VocabEntry` exposes:

| Field | Description |
|---|---|
| `headword` | Citation form (principal parts / nom+gen+gender) or display lemma |
| `pos_marker` | Abbreviated POS tag (`v.`, `adj.`, `adv.`, etc.) — empty for nouns (gender in citation) |
| `short_gloss` | Trimmed gloss (up to 3 senses) |
| `full_gloss` | All senses joined |
| `frequency` | Count across the input text |
| `forms_seen` | Set of inflected forms observed |

## Configuration

```python
from vocabbuilder import VocabPipeline, PipelineConfig

config = PipelineConfig(
    spacy_model="la_core_web_lg",  # default; lighter: "la_core_web_sm"
    min_frequency=2,               # drop words seen only once
)
pipeline = VocabPipeline(config)
```

## Related packages

- [`latincy-lexicon`](https://github.com/latincy/latincy-lexicon) — Whitaker's Words lexical data for LatinCy
- [`latincy-preprocess`](https://github.com/latincy/latincy-preprocess) — Latin text preprocessing utilities

## License

MIT
