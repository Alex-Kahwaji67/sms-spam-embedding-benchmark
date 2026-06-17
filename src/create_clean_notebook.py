import json
from pathlib import Path
from textwrap import dedent


OUTPUT_NOTEBOOK = Path("../notebooks/sms_spam_embedding_benchmark.ipynb")


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
            import re
            import time
            import warnings

            import numpy as np
            import pandas as pd
            from gensim.models import Word2Vec
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
            from sklearn.model_selection import cross_val_score, train_test_split
            from sklearn.neighbors import KNeighborsClassifier

            warnings.filterwarnings("ignore")

            DATA_PATH = Path("../data/spam.csv")
            RANDOM_STATE = 42
            K_VALUES = [1, 3, 5, 7, 11, 21]
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
                    f"Missing {DATA_PATH}. See ../data/README_DATA.md for download and placement notes."
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
            - Word2Vec-style sentence vectors created by averaging word embeddings trained on the SMS corpus.
            - BERT sentence embeddings from `bert-base-uncased` using mean pooling.
            """
        ),
        code(
            """
            def tune_knn(X_train_features, y_train, k_values=K_VALUES):
                scores = []
                for k in k_values:
                    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
                    cv_scores = cross_val_score(model, X_train_features, y_train, cv=5, scoring="f1")
                    scores.append({"k": k, "cv_f1": cv_scores.mean(), "cv_f1_std": cv_scores.std()})
                best = max(scores, key=lambda row: row["cv_f1"])
                return best, pd.DataFrame(scores)


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
            tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)

            best_tfidf, tfidf_cv = tune_knn(X_train_tfidf, y_train)
            knn_tfidf = KNeighborsClassifier(n_neighbors=int(best_tfidf["k"]), n_jobs=-1)
            knn_tfidf.fit(X_train_tfidf, y_train)
            tfidf_result = evaluate_model("TF-IDF", knn_tfidf, X_test_tfidf, y_test)
            tfidf_result["best_k"] = int(best_tfidf["k"])
            tfidf_result["dimensionality"] = X_train_tfidf.shape[1]
            tfidf_result["runtime_seconds"] = time.time() - start

            tfidf_cv
            """
        ),
        code(
            """
            # Word2Vec averaged embeddings + KNN
            def tokenize(text):
                text = text.lower()
                text = re.sub(r"[^a-zA-Z\\s]", " ", text)
                return text.split()


            def sentence_vector(tokens, model, vector_size):
                vectors = [model.wv[word] for word in tokens if word in model.wv]
                if not vectors:
                    return np.zeros(vector_size)
                return np.mean(vectors, axis=0)


            start = time.time()
            train_tokens = X_train.apply(tokenize)
            test_tokens = X_test.apply(tokenize)

            vector_size = 100
            w2v = Word2Vec(
                sentences=train_tokens,
                vector_size=vector_size,
                window=5,
                min_count=1,
                workers=4,
                seed=RANDOM_STATE,
            )
            X_train_w2v = np.vstack([sentence_vector(tokens, w2v, vector_size) for tokens in train_tokens])
            X_test_w2v = np.vstack([sentence_vector(tokens, w2v, vector_size) for tokens in test_tokens])

            best_w2v, w2v_cv = tune_knn(X_train_w2v, y_train)
            knn_w2v = KNeighborsClassifier(n_neighbors=int(best_w2v["k"]), n_jobs=-1)
            knn_w2v.fit(X_train_w2v, y_train)
            w2v_result = evaluate_model("Word2Vec average", knn_w2v, X_test_w2v, y_test)
            w2v_result["best_k"] = int(best_w2v["k"])
            w2v_result["dimensionality"] = X_train_w2v.shape[1]
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

            X_train_bert = encode_texts_batch(X_train, tokenizer, bert)
            X_test_bert = encode_texts_batch(X_test, tokenizer, bert)

            best_bert, bert_cv = tune_knn(X_train_bert, y_train)
            knn_bert = KNeighborsClassifier(n_neighbors=int(best_bert["k"]), n_jobs=-1)
            knn_bert.fit(X_train_bert, y_train)
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

            The public notebook does not print raw SMS examples by default because the dataset includes phone numbers, URLs, and personal message text. The original coursework found:

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

            BERT produced the strongest spam-class F1 in the coursework, while TF-IDF was fastest and Word2Vec offered a strong speed/quality tradeoff. Limitations include a small benchmark dataset, possible dated SMS language, compute cost for BERT, and privacy concerns around raw SMS text.
            """
        ),
    ]

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
