import json
from pathlib import Path
from textwrap import dedent


OUTPUT_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/sms_spam_embedding_benchmark.ipynb"


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(True),
    }


def main():
    cells = [
        markdown(
            """
            # SMS Spam Embedding Benchmark

            Benchmark TF-IDF, averaged Word2Vec-style embeddings, and BERT sentence embeddings with KNN classifiers for SMS spam detection.
            """
        ),
        markdown(
            """
            ## 1) Intro

            The goal is to compare how different text representations affect spam classification. The same KNN classifier family is used across representations so the benchmark focuses on the feature/embedding choice.
            """
        ),
        code(
            """
            from pathlib import Path
            import sys
            import time

            import numpy as np
            import pandas as pd
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
            from sklearn.model_selection import StratifiedKFold, train_test_split

            # Support kernels started from the repository root or notebooks/.
            REPO_ROOT = Path.cwd()
            if not (REPO_ROOT / "src" / "benchmark.py").is_file():
                REPO_ROOT = REPO_ROOT.parent
            if not (REPO_ROOT / "src" / "benchmark.py").is_file():
                raise FileNotFoundError("Start the notebook from the repository root or notebooks/.")
            sys.path.insert(0, str(REPO_ROOT))
            from src.benchmark import K_VALUES, MeanWord2Vec, tune_knn

            DATA_PATH = REPO_ROOT / "data" / "spam.csv"
            RANDOM_STATE = 42
            """
        ),
        markdown(
            """
            ## 2) Data + Preprocessing

            The coursework used `spam.csv`, an SMS spam/ham dataset with labels in `v1` and message text in `v2`. Raw data is not included in this staged repo.
            """
        ),
        code(
            """
            if not DATA_PATH.exists():
                raise FileNotFoundError(
                    f"Missing {DATA_PATH}. See data/README_DATA.md for placement notes."
                )

            df = pd.read_csv(DATA_PATH, encoding="cp1252")
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
            df["target"] = df["label"].map({"ham": 0, "spam": 1})

            X_train, X_test, y_train, y_test = train_test_split(
                df["text"],
                df["target"],
                test_size=0.20,
                random_state=RANDOM_STATE,
                stratify=df["target"],
            )
            # Matches the previous classifier cv=5 behavior, with identical
            # positional folds reused for every representation and k value.
            CV_SPLITS = list(StratifiedKFold(n_splits=5, shuffle=False).split(X_train, y_train))

            print("Rows:", len(df))
            print("Train/test:", len(X_train), len(X_test))
            print("Spam rate:", round(df["target"].mean(), 4))
            """
        ),
        markdown(
            """
            ## 3) Embeddings

            Three representations are compared:

            - TF-IDF with unigrams and bigrams.
            - Word2Vec-style sentence vectors created by averaging word embeddings trained on each fold's training text.
            - Frozen BERT sentence embeddings from `bert-base-uncased` using mean pooling.

            TF-IDF and Word2Vec are fitted inside the KNN cross-validation pipeline.
            Validation text is transformed with that fold's fitted representation only.
            After selecting k by mean spam F1, the full pipeline is refitted on the
            80% training split. The 20% test split is used only for final evaluation.
            BERT is never fine-tuned or adapted on this corpus, so its independent
            per-message embeddings can be computed once before KNN cross-validation.
            """
        ),
        code(
            """
            def evaluate_model(name, model, X_test_features, y_test):
                preds = model.predict(X_test_features)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, preds, average=None, labels=[0, 1]
                )
                return {
                    "representation": name,
                    "accuracy": accuracy_score(y_test, preds),
                    "ham_precision": precision[0],
                    "ham_recall": recall[0],
                    "ham_f1": f1[0],
                    "spam_precision": precision[1],
                    "spam_recall": recall[1],
                    "spam_f1": f1[1],
                    "confusion_matrix": confusion_matrix(y_test, preds),
                    "predictions": preds,
                }
            """
        ),
        markdown("## 4) Models"),
        code(
            """
            # TF-IDF + KNN
            start = time.time()
            best_tfidf, tfidf_cv, knn_tfidf = tune_knn(
                X_train, y_train,
                representation=TfidfVectorizer(ngram_range=(1, 2), max_features=5000),
                k_values=K_VALUES, cv=CV_SPLITS,
            )
            tfidf_result = evaluate_model("TF-IDF", knn_tfidf, X_test, y_test)
            tfidf_result["best_k"] = int(best_tfidf["k"])
            tfidf_result["dimensionality"] = len(knn_tfidf.named_steps["representation"].vocabulary_)
            tfidf_result["runtime_seconds"] = time.time() - start

            tfidf_cv
            """
        ),
        code(
            """
            # Word2Vec averaged embeddings + KNN
            start = time.time()
            best_w2v, w2v_cv, knn_w2v = tune_knn(
                X_train, y_train,
                representation=MeanWord2Vec(vector_size=100, window=5, min_count=1,
                                           workers=1, seed=RANDOM_STATE),
                k_values=K_VALUES, cv=CV_SPLITS,
            )
            w2v_result = evaluate_model("Word2Vec average", knn_w2v, X_test, y_test)
            w2v_result["best_k"] = int(best_w2v["k"])
            w2v_result["dimensionality"] = knn_w2v.named_steps["representation"].vector_size
            w2v_result["runtime_seconds"] = time.time() - start

            w2v_cv
            """
        ),
        code(
            """
            # BERT sentence embeddings + KNN
            # This section requires torch and transformers. It is slower than TF-IDF and Word2Vec.
            import torch
            from transformers import BertModel, BertTokenizer

            def encode_texts_batch(texts, tokenizer, model, batch_size=32, max_length=128):
                embeddings = []
                device = next(model.parameters()).device
                texts_list = list(texts)
                for start in range(0, len(texts_list), batch_size):
                    batch = texts_list[start:start + batch_size]
                    encoded = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    with torch.no_grad():
                        outputs = model(**encoded)
                        mask = encoded["attention_mask"].unsqueeze(-1)
                        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                    embeddings.append(pooled.cpu().numpy())
                return np.vstack(embeddings)


            start = time.time()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            bert = BertModel.from_pretrained("bert-base-uncased").to(device)
            bert.eval()
            bert.requires_grad_(False)

            X_train_bert = encode_texts_batch(X_train, tokenizer, bert)

            best_bert, bert_cv, knn_bert = tune_knn(
                X_train_bert, y_train, representation="passthrough",
                k_values=K_VALUES, cv=CV_SPLITS,
            )
            X_test_bert = encode_texts_batch(X_test, tokenizer, bert)
            bert_result = evaluate_model("BERT", knn_bert, X_test_bert, y_test)
            bert_result["best_k"] = int(best_bert["k"])
            bert_result["dimensionality"] = X_train_bert.shape[1]
            bert_result["runtime_seconds"] = time.time() - start

            bert_cv
            """
        ),
        markdown("## 5) Evaluation"),
        code(
            """
            summary = pd.DataFrame([
                {k: v for k, v in result.items() if k not in ["confusion_matrix", "predictions"]}
                for result in [tfidf_result, w2v_result, bert_result]
            ])
            summary.sort_values("spam_f1", ascending=False)
            """
        ),
        code(
            """
            for result in [tfidf_result, w2v_result, bert_result]:
                print(result["representation"])
                print(result["confusion_matrix"])
                print()
            """
        ),
        markdown(
            """
            ## 6) Error Analysis

            The public notebook does not print raw SMS examples by default because the dataset includes phone numbers, URLs, and personal message text. The following are historical coursework observations, not findings from a rerun of the corrected pipeline:

            - TF-IDF misses contextual spam patterns and struggles when spam lacks obvious trigger words.
            - Averaged Word2Vec loses word order and weakens negation.
            - BERT fixes many TF-IDF/Word2Vec errors but still confuses some legitimate messages with spam-like wording and some subtle spam with conversational language.
            """
        ),
        code(
            """
            error_counts = []
            for result in [tfidf_result, w2v_result, bert_result]:
                preds = result["predictions"]
                false_positives = int(((y_test == 0) & (preds == 1)).sum())
                false_negatives = int(((y_test == 1) & (preds == 0)).sum())
                error_counts.append({
                    "representation": result["representation"],
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "total_errors": false_positives + false_negatives,
                })

            pd.DataFrame(error_counts)
            """
        ),
        markdown(
            """
            ## 7) Discussion / Limits

            The historical coursework reported BERT as strongest by spam F1, TF-IDF
            as fastest, and Word2Vec as a speed/quality tradeoff. Those rankings and
            timings have not been revalidated after correcting representation
            fitting within cross-validation. The original test split was separate;
            the CV issue alone does not establish that its reported scores were
            incorrect. Refit and rerun with the original CSV before publishing
            updated metrics. Word2Vec now uses one worker for reproducibility;
            set PYTHONHASHSEED=0 before starting Python/Jupyter for repeatable word
            initialization across processes. Corrected runtimes include fold-local
            representation fitting and are not directly comparable with old timings.

            Limitations include a small benchmark dataset, possible dated SMS
            language, compute cost for BERT, and privacy concerns around raw SMS text.
            """
        ),
    ]

    for index, cell in enumerate(cells):
        cell["id"] = f"sms-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
