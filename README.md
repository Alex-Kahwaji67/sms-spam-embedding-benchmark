# SMS Spam Embedding Benchmark

This project benchmarks TF-IDF, averaged Word2Vec-style embeddings, and BERT sentence embeddings for SMS spam detection. Each representation is paired with a KNN classifier so the comparison focuses on how text representation changes performance, runtime, and error patterns.

The project comes from Text and Social Media Analytics coursework and has been cleaned for portfolio staging.

## Problem

SMS spam detection is a text classification problem where obvious spam keywords are useful but not enough. The project compares sparse keyword features, static averaged embeddings, and contextual BERT embeddings to see which representation best separates ham from spam.

## Data

The coursework used `spam.csv`, with:

- `v1`: label (`ham` or `spam`)
- `v2`: SMS message text

Raw data is not included in this repo. To run the notebook, place the dataset here:

```text
data/spam.csv
```

The original notebook reads the file using CP1252 encoding. See `data/README_DATA.md` for data notes and privacy cautions.

## Methods

- TF-IDF with unigrams and bigrams, capped at 5,000 features.
- Word2Vec-style sentence embeddings by training Word2Vec on the SMS training corpus and averaging word vectors into 100-dimensional message vectors.
- BERT sentence embeddings using `bert-base-uncased` with mean pooling, producing 768-dimensional message vectors.
- KNN classifiers for all representations.
- 5-fold cross-validation over k values: 1, 3, 5, 7, 11, and 21.

## Evaluation

The project uses:

- Accuracy
- Class-wise precision
- Class-wise recall
- Class-wise F1
- Confusion matrices
- False-positive and false-negative error analysis

The notebook's final comparison identifies BERT + KNN as the strongest performer by accuracy and spam-class F1. TF-IDF is fastest, while Word2Vec provides a strong speed/quality balance.

## How To Run

1. Use Python 3.10+.
2. Install key libraries:

```bash
pip install pandas numpy scikit-learn gensim torch transformers
```

3. Place the dataset at:

```text
data/spam.csv
```

4. Open and run:

```text
notebooks/sms_spam_embedding_benchmark.ipynb
```

The BERT section can run slowly on CPU. A GPU is helpful but not required.

## Limitations / Future Work

- The dataset is a relatively small benchmark and may not reflect current spam tactics.
- BERT improves contextual understanding but adds compute cost.
- Averaged Word2Vec embeddings lose word order and weaken negation.
- Future work could compare logistic regression, linear SVM, or transformer fine-tuning against the KNN benchmark.
- Raw SMS examples should not be committed because they may contain phone numbers, URLs, adult/spam text, and personal conversational messages.

