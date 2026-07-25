"""Tests for core data models."""

from vocabbuilder.core.models import VocabEntry, VocabList


class TestVocabEntry:
    def test_key(self):
        entry = VocabEntry(lemma="bellum", display_lemma="bellum", pos="NOUN")
        assert entry.key == ("bellum", "NOUN")

    def test_merge(self):
        e1 = VocabEntry(
            lemma="pars",
            display_lemma="pars",
            pos="NOUN",
            forms_seen={"partes"},
            frequency=2,
            morphology=[{"Case": "Acc"}],
            passage_indices=[0],
        )
        e2 = VocabEntry(
            lemma="pars",
            display_lemma="pars",
            pos="NOUN",
            forms_seen={"partis"},
            frequency=1,
            morphology=[{"Case": "Gen"}],
            passage_indices=[1],
        )
        e1.merge(e2)
        assert e1.frequency == 3
        assert e1.forms_seen == {"partes", "partis"}
        assert len(e1.morphology) == 2
        assert e1.passage_indices == [0, 1]

    def test_merge_fills_empty_glosses(self):
        e1 = VocabEntry(lemma="sum", display_lemma="sum", pos="VERB", glosses=[])
        e2 = VocabEntry(lemma="sum", display_lemma="sum", pos="VERB", glosses=["to be"])
        e1.merge(e2)
        assert e1.glosses == ["to be"]


class TestVocabList:
    def _make_entries(self):
        return [
            VocabEntry(lemma="bellum", display_lemma="bellum", pos="NOUN", frequency=3),
            VocabEntry(lemma="ager", display_lemma="ager", pos="NOUN", frequency=1),
            VocabEntry(lemma="capio", display_lemma="capio", pos="VERB", frequency=5),
        ]

    def test_len_and_iter(self):
        vl = VocabList(entries=self._make_entries())
        assert len(vl) == 3
        assert [e.lemma for e in vl] == ["bellum", "ager", "capio"]

    def test_getitem(self):
        vl = VocabList(entries=self._make_entries())
        assert vl[1].lemma == "ager"

    def test_by_frequency(self):
        vl = VocabList(entries=self._make_entries()).by_frequency()
        assert [e.frequency for e in vl] == [5, 3, 1]

    def test_by_alpha(self):
        vl = VocabList(entries=self._make_entries()).by_alpha()
        assert [e.display_lemma for e in vl] == ["ager", "bellum", "capio"]

    def test_filter_pos(self):
        vl = VocabList(entries=self._make_entries()).filter_pos({"NOUN"})
        assert len(vl) == 2
        assert all(e.pos == "NOUN" for e in vl)

    def test_filter_min_frequency(self):
        vl = VocabList(entries=self._make_entries()).filter_min_frequency(3)
        assert len(vl) == 2
        assert all(e.frequency >= 3 for e in vl)

    def test_filter_lemmas_exclude(self):
        vl = VocabList(entries=self._make_entries()).filter_lemmas(exclude={"bellum", "capio"})
        assert [e.lemma for e in vl] == ["ager"]

    def test_filter_lemmas_keep(self):
        vl = VocabList(entries=self._make_entries()).filter_lemmas(keep={"ager", "capio"})
        assert [e.lemma for e in vl] == ["ager", "capio"]

    def test_filter_lemmas_combined(self):
        vl = VocabList(entries=self._make_entries()).filter_lemmas(
            keep={"ager", "capio"}, exclude={"capio"}
        )
        assert [e.lemma for e in vl] == ["ager"]

    def test_filter_lemmas_normalizes_orthography(self):
        # A v/j-form wordlist must match the u-form spaCy lemma "iacio".
        vl = VocabList(
            entries=[VocabEntry(lemma="iacio", display_lemma="iacio", pos="VERB")]
        ).filter_lemmas(exclude={"jacio"})
        assert len(vl) == 0

    def test_filter_lemmas_requires_argument(self):
        import pytest

        with pytest.raises(ValueError):
            VocabList(entries=self._make_entries()).filter_lemmas()

    def test_filter_keyness_min_score(self):
        scores = {"bellum": 0.8, "ager": 0.05, "capio": 0.3}
        vl = VocabList(entries=self._make_entries()).filter_keyness(scores, min_score=0.1)
        assert sorted(e.lemma for e in vl) == ["bellum", "capio"]

    def test_filter_keyness_drops_unscored_by_default(self):
        # No threshold: entries absent from the score table (weight 0) are dropped.
        scores = {"bellum": 0.8}
        vl = VocabList(entries=self._make_entries()).filter_keyness(scores)
        assert [e.lemma for e in vl] == ["bellum"]

    def test_filter_keyness_top_n_preserves_order(self):
        scores = {"bellum": 0.8, "ager": 0.2, "capio": 0.5}
        vl = VocabList(entries=self._make_entries()).filter_keyness(scores, top_n=2)
        # capio (0.5) and bellum (0.8) are the top 2; original order is bellum, capio.
        assert [e.lemma for e in vl] == ["bellum", "capio"]

    # --- filter_keyness: alternative weight representations ---

    def test_filter_keyness_accepts_pairs(self):
        pairs = [("bellum", 0.8), ("ager", 0.05), ("capio", 0.3)]
        vl = VocabList(entries=self._make_entries()).filter_keyness(pairs, min_score=0.1)
        assert sorted(e.lemma for e in vl) == ["bellum", "capio"]

    def test_filter_keyness_numpy_matrix_selects_document_row(self):
        import numpy as np

        # Two-document DTM; row order matches feature_names order.
        dtm = np.array([[0.8, 0.05, 0.3], [0.1, 0.9, 0.0]])
        names = ["bellum", "ager", "capio"]
        vl = VocabList(entries=self._make_entries()).filter_keyness(
            dtm, feature_names=names, document=1, min_score=0.5
        )
        assert [e.lemma for e in vl] == ["ager"]

    def test_filter_keyness_numpy_1d_vector(self):
        import numpy as np

        vec = np.array([0.8, 0.05, 0.3])
        names = ["bellum", "ager", "capio"]
        vl = VocabList(entries=self._make_entries()).filter_keyness(
            vec, feature_names=names, min_score=0.1
        )
        assert sorted(e.lemma for e in vl) == ["bellum", "capio"]

    def test_filter_keyness_matrix_requires_feature_names(self):
        import numpy as np
        import pytest

        dtm = np.array([[0.8, 0.05, 0.3]])
        with pytest.raises(ValueError, match="feature_names"):
            VocabList(entries=self._make_entries()).filter_keyness(dtm, document=0)

    def test_filter_keyness_sparse_like_densifies(self):
        # A minimal scipy-sparse stand-in (dep-free): 2-D, row-indexable, toarray().
        import numpy as np

        class _FakeSparse:
            ndim = 2

            def __init__(self, arr):
                self._a = np.asarray(arr)

            def __getitem__(self, i):
                row = self._a[i]

                class _Row:
                    def toarray(self_inner):
                        return row.reshape(1, -1)

                return _Row()

        dtm = _FakeSparse([[0.8, 0.05, 0.3]])
        names = ["bellum", "ager", "capio"]
        vl = VocabList(entries=self._make_entries()).filter_keyness(
            dtm, feature_names=names, document=0, min_score=0.1
        )
        assert sorted(e.lemma for e in vl) == ["bellum", "capio"]

    def test_filter_keyness_pandas_series(self):
        pd = __import__("pytest").importorskip("pandas")
        series = pd.Series({"bellum": 0.8, "ager": 0.05, "capio": 0.3})
        vl = VocabList(entries=self._make_entries()).filter_keyness(series, min_score=0.1)
        assert sorted(e.lemma for e in vl) == ["bellum", "capio"]

    def test_filter_keyness_pandas_dataframe_row(self):
        pd = __import__("pytest").importorskip("pandas")
        dtm = pd.DataFrame(
            [[0.8, 0.05, 0.3], [0.1, 0.9, 0.0]],
            columns=["bellum", "ager", "capio"],
            index=["d0", "d1"],
        )
        # Label-based row selection via .loc when document is not an int.
        vl = VocabList(entries=self._make_entries()).filter_keyness(
            dtm, document="d1", min_score=0.5
        )
        assert [e.lemma for e in vl] == ["ager"]

    def test_filter_keyness_scipy_sparse(self):
        pytest = __import__("pytest")
        sparse = pytest.importorskip("scipy.sparse")
        import numpy as np

        dtm = sparse.csr_matrix(np.array([[0.8, 0.05, 0.3]]))
        names = ["bellum", "ager", "capio"]
        vl = VocabList(entries=self._make_entries()).filter_keyness(
            dtm, feature_names=names, document=0, min_score=0.1
        )
        assert sorted(e.lemma for e in vl) == ["bellum", "capio"]

    def test_filter_keyness_rejects_unrecognized_type(self):
        import pytest

        with pytest.raises(TypeError, match="could not interpret"):
            VocabList(entries=self._make_entries()).filter_keyness(42)

    # --- filter_keyness: review fixes ---

    def test_filter_keyness_sums_uv_variant_collisions(self):
        # 'vita' and 'uita' are distinct vectorizer columns that fold to one lemma;
        # their weights must SUM (0.3 + 0.5 = 0.8), not clobber (last-wins 0.5).
        entries = [VocabEntry(lemma="uita", display_lemma="vita", pos="NOUN")]
        vl = VocabList(entries=entries).filter_keyness(
            {"vita": 0.3, "uita": 0.5}, min_score=0.6
        )
        # 0.8 >= 0.6 keeps it; last-wins 0.5 would have dropped it.
        assert [e.lemma for e in vl] == ["uita"]

    def test_filter_keyness_matrix_length_mismatch_raises(self):
        import numpy as np
        import pytest

        row = np.array([[0.8, 0.1]])  # 2 weights
        with pytest.raises(ValueError, match="must align"):
            VocabList(entries=self._make_entries()).filter_keyness(
                row, feature_names=["bellum", "ager", "capio"], document=0  # 3 names
            )

    def test_filter_keyness_default_keeps_signed_measure(self):
        # A negative (log-likelihood-style) weight is still "scored" → kept by the
        # default, which drops only lemmas absent from the measure.
        scores = {"bellum": -0.5}
        vl = VocabList(entries=self._make_entries()).filter_keyness(scores)
        assert [e.lemma for e in vl] == ["bellum"]

    def test_filter_keyness_nested_list_rejected_as_matrix(self):
        import pytest

        # A raw 2-column matrix as a plain list must not be misread as pairs.
        with pytest.raises(TypeError, match="matrix row"):
            VocabList(entries=self._make_entries()).filter_keyness([[0.8, 0.1], [0.3, 0.5]])

    def test_filter_keyness_dataframe_integer_label(self):
        pd = __import__("pytest").importorskip("pandas")
        # Integer index LABELS (not positions): document=20 must select the row
        # labeled 20, not .iloc[20] (which would IndexError).
        dtm = pd.DataFrame(
            [[0.1, 0.9, 0.0], [0.8, 0.05, 0.3]],
            columns=["bellum", "ager", "capio"],
            index=[10, 20],
        )
        vl = VocabList(entries=self._make_entries()).filter_keyness(
            dtm, document=20, min_score=0.5
        )
        assert [e.lemma for e in vl] == ["bellum"]
