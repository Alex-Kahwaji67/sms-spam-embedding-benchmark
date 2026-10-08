"""Synthetic regression checks; these are not SMS benchmark performance results."""

from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from src import benchmark, create_clean_notebook


ROOT = Path(__file__).resolve().parents[1]


def synthetic_messages(count):
    # Alphabetic sentinels survive both the TF-IDF and Word2Vec tokenizers.
    return pd.Series([
        f"{'offer prize' if i % 2 else 'meeting lunch'} shared "
        f"sentinel{chr(97 + i // 26)}{chr(97 + i % 26)} "
        f"{'occasional' if i % 3 == 0 else ''}"
        for i in range(count)
    ], index=np.arange(count) * 7 + 100)


class FoldIsolationTests(unittest.TestCase):
    def setUp(self):
        self.texts = synthetic_messages(20)
        self.labels = pd.Series([i % 2 for i in range(20)], index=self.texts.index)
        self.folds = list(StratifiedKFold(5, shuffle=False).split(self.texts, self.labels))
        self.k_values = [1, 3]

    def assert_fit_inputs(self, observed):
        expected = Counter()
        for train, _ in self.folds:
            expected[tuple(self.texts.iloc[train])] += len(self.k_values)
        # The sole full-training fit is the selected model's final refit.
        expected[tuple(self.texts)] += 1
        self.assertEqual(Counter(observed), expected)

    def test_tfidf_grid_fits_only_fold_training_text_and_idf(self):
        observed = []
        original = TfidfVectorizer.fit_transform

        def record_fit(vectorizer, X, y=None):
            docs = list(X)
            features = original(vectorizer, docs, y)
            observed.append(tuple(docs))
            analyzer = vectorizer.build_analyzer()
            document_terms = [set(analyzer(doc)) for doc in docs]
            vocabulary = set().union(*document_terms)
            self.assertEqual(set(vectorizer.vocabulary_), vocabulary)
            # Check document frequency too: held-out repetitions of known words
            # must not influence IDF even if they introduce no new vocabulary.
            for term, column in vectorizer.vocabulary_.items():
                frequency = sum(term in terms for terms in document_terms)
                expected_idf = np.log((1 + len(docs)) / (1 + frequency)) + 1
                self.assertAlmostEqual(vectorizer.idf_[column], expected_idf)
            return features

        with patch.object(TfidfVectorizer, "fit_transform", record_fit):
            _, scores, model = benchmark.tune_knn(
                self.texts, self.labels,
                representation=TfidfVectorizer(ngram_range=(1, 2), max_features=5000),
                k_values=self.k_values, cv=self.folds,
            )
        self.assert_fit_inputs(observed)
        self.assertEqual(scores["k"].tolist(), self.k_values)
        self.assertEqual(model.named_steps["knn"].n_samples_fit_, len(self.texts))
        self.assertNotIn("quarantinetest", model.named_steps["representation"].vocabulary_)

    def test_word2vec_grid_fits_only_fold_training_text(self):
        observed = []
        original = benchmark.MeanWord2Vec.fit

        def record_fit(transformer, X, y=None):
            docs = list(X)
            result = original(transformer, docs, y)
            observed.append(tuple(docs))
            expected_words = {word for doc in docs for word in benchmark.tokenize(doc)}
            self.assertEqual(set(transformer.model_.wv.key_to_index), expected_words)
            self.assertEqual(transformer.model_.corpus_count, len(docs))
            for word in expected_words:
                count = sum(benchmark.tokenize(doc).count(word) for doc in docs)
                self.assertEqual(transformer.model_.wv.get_vecattr(word, "count"), count)
            return result

        with patch.object(benchmark.MeanWord2Vec, "fit", record_fit):
            _, _, model = benchmark.tune_knn(
                self.texts, self.labels,
                representation=benchmark.MeanWord2Vec(vector_size=8),
                k_values=self.k_values, cv=self.folds,
            )
        self.assert_fit_inputs(observed)
        self.assertEqual(model.named_steps["knn"].n_samples_fit_, len(self.texts))
        self.assertNotIn("quarantinetest", model.named_steps["representation"].model_.wv)

    def test_transform_does_not_change_fitted_state(self):
        train, validation = self.folds[0]
        held_out = list(self.texts.iloc[validation]) + ["quarantinetest " * 100, "shared " * 100]

        tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit(self.texts.iloc[train])
        vocab_before, idf_before = dict(tfidf.vocabulary_), tfidf.idf_.copy()
        training_before = tfidf.transform(self.texts.iloc[train]).toarray()
        tfidf.transform(held_out)
        self.assertEqual(tfidf.vocabulary_, vocab_before)
        np.testing.assert_array_equal(tfidf.idf_, idf_before)
        np.testing.assert_array_equal(tfidf.transform(self.texts.iloc[train]).toarray(), training_before)

        w2v = benchmark.MeanWord2Vec(vector_size=8).fit(self.texts.iloc[train])
        vocab_before = dict(w2v.model_.wv.key_to_index)
        weights_before = w2v.model_.wv.vectors.copy()
        training_before = w2v.transform(self.texts.iloc[train])
        w2v.transform(held_out)
        self.assertEqual(w2v.model_.wv.key_to_index, vocab_before)
        np.testing.assert_array_equal(w2v.model_.wv.vectors, weights_before)
        np.testing.assert_array_equal(w2v.transform(self.texts.iloc[train]), training_before)
        np.testing.assert_array_equal(w2v.transform(["quarantinetest", "123 !"]), np.zeros((2, 8)))
        self.assertEqual(w2v.transform([]).shape, (0, 8))

    def test_frozen_features_keep_original_knn_selection_behavior(self):
        # Synthetic fixed vectors exercise BERT's passthrough path without
        # downloading a model or claiming to validate BERT inference quality.
        features = np.random.default_rng(42).normal(size=(len(self.texts), 8))
        best, scores, fitted = benchmark.tune_knn(
            features, self.labels, representation="passthrough",
            k_values=self.k_values, cv=self.folds,
        )
        expected = [cross_val_score(
            KNeighborsClassifier(n_neighbors=k), features, self.labels,
            cv=self.folds, scoring="f1", error_score="raise",
        ) for k in self.k_values]
        np.testing.assert_allclose(scores["cv_f1"], [fold_scores.mean() for fold_scores in expected])
        np.testing.assert_allclose(scores["cv_f1_std"], [fold_scores.std() for fold_scores in expected])
        self.assertEqual(best["k"], self.k_values[int(np.argmax(scores["cv_f1"]))])
        self.assertEqual(fitted.predict(features[:3]).shape, (3,))


class NotebookChecks(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((ROOT / "notebooks/sms_spam_embedding_benchmark.ipynb").read_text())

    def test_generator_matches_notebook_and_code_compiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.ipynb"
            with patch.object(create_clean_notebook, "OUTPUT_NOTEBOOK", path):
                create_clean_notebook.main()
            self.assertEqual(json.loads(path.read_text()), self.notebook)
        for i, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{i}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_non_bert_notebook_cells_run_on_synthetic_csv(self):
        # Run the real cells, real Gensim and the complete six-k/five-fold grid.
        # Keep the fixture outside data/; never replace the user's original CSV.
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "synthetic.csv"
            pd.DataFrame({
                "v1": ["spam" if i % 2 else "ham" for i in range(80)],
                "v2": synthetic_messages(80).to_numpy(),
            }).to_csv(path, index=False, encoding="cp1252")
            namespace = {}
            for index in [2, 4, 6, 8, 9]:
                exec(compile("".join(self.notebook["cells"][index]["source"]),
                             f"cell-{index}", "exec"), namespace)
                if index == 2:
                    namespace["DATA_PATH"] = path
            self.assertEqual((len(namespace["X_train"]), len(namespace["X_test"])), (64, 16))
            for name in ["tfidf", "w2v"]:
                result = namespace[f"{name}_result"]
                self.assertEqual(len(result["predictions"]), 16)
                self.assertEqual(result["confusion_matrix"].sum(), 16)
                self.assertEqual(namespace[f"{name}_cv"]["k"].tolist(), benchmark.K_VALUES)
            # Final refits contain all training words and none of the test-only
            # sentinel words, including after final prediction on the test split.
            train_words = {w for doc in namespace["X_train"] for w in benchmark.tokenize(doc)}
            test_words = {w for doc in namespace["X_test"] for w in benchmark.tokenize(doc)}
            test_only = test_words - train_words
            self.assertTrue(test_only)
            tfidf = namespace["knn_tfidf"].named_steps["representation"]
            w2v = namespace["knn_w2v"].named_steps["representation"]
            self.assertTrue(train_words.issubset(tfidf.vocabulary_))
            self.assertEqual(set(w2v.model_.wv.key_to_index), train_words)
            self.assertTrue(test_only.isdisjoint(tfidf.vocabulary_))
            self.assertTrue(test_only.isdisjoint(w2v.model_.wv.key_to_index))


if __name__ == "__main__":
    unittest.main()
