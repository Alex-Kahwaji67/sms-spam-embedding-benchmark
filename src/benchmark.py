"""Fold-local text representations and KNN selection for the notebook."""

import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


K_VALUES = [1, 3, 5, 7, 11, 21]


def tokenize(text):
    return re.sub(r"[^a-zA-Z\s]", " ", text.lower()).split()


class MeanWord2Vec(TransformerMixin, BaseEstimator):
    """Learn word vectors on fit inputs only; transform never updates them."""

    def __init__(self, vector_size=100, window=5, min_count=1, workers=1, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.seed = seed

    def fit(self, X, y=None):
        self.model_ = Word2Vec(
            sentences=[tokenize(text) for text in X],
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "model_")
        rows = []
        for text in X:
            vectors = [self.model_.wv[word] for word in tokenize(text)
                       if word in self.model_.wv]
            rows.append(np.mean(vectors, axis=0) if vectors
                        else np.zeros(self.vector_size, dtype=np.float32))
        return np.asarray(rows, dtype=np.float32).reshape(-1, self.vector_size)


def tune_knn(X_train, y_train, *, representation, k_values=K_VALUES, cv=5):
    """Select k and refit using only the training split.

    Pass raw text with an unfitted learned representation. GridSearchCV clones
    the entire pipeline per candidate/fold, so vocabulary, IDF and Word2Vec
    weights see only that fold's training rows. After selection, refit=True
    learns a fresh representation and KNN on all training rows.

    representation="passthrough" is reserved for frozen, independently computed
    BERT features; it must not be used for globally fitted TF-IDF or Word2Vec.
    This function never receives held-out test inputs or labels.
    """
    pipeline = Pipeline([
        ("representation", representation),
        ("knn", KNeighborsClassifier(n_jobs=-1)),
    ])
    search = GridSearchCV(
        pipeline,
        {"knn__n_neighbors": list(k_values)},
        scoring="f1",  # spam is the positive class (1), as in the coursework
        cv=cv,
        refit=True,
        error_score="raise",
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    scores = pd.DataFrame({
        "k": [params["knn__n_neighbors"] for params in search.cv_results_["params"]],
        "cv_f1": search.cv_results_["mean_test_score"],
        "cv_f1_std": search.cv_results_["std_test_score"],
    })
    best = scores.iloc[search.best_index_].to_dict()
    best["k"] = int(best["k"])
    return best, scores, search.best_estimator_
