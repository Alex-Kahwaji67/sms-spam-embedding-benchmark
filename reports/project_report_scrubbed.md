# SMS Spam Embedding Benchmark - Scrubbed Report

> Historical coursework record. The figures and qualitative findings below have
> not been rerun or revalidated after the October 7 cross-validation correction.
> The public notebook contains no saved outputs and the original CSV was not
> available during the fix. All historical numbers, including conflicting
> Word2Vec entries, are preserved rather than replaced with unexecuted estimates.

## Objective

Benchmark text representations for SMS spam detection while holding the classifier family constant. The project compares TF-IDF, averaged Word2Vec-style sentence embeddings, and BERT sentence embeddings with KNN classifiers.

## Dataset

- File used in coursework: `spam.csv`
- Label column: `v1`, mapped to ham/spam
- Text column: `v2`
- Historically reported train/test split: 4,457 training messages and 1,115 test messages
- Encoding used: CP1252

Raw SMS examples are omitted from this scrubbed report because the dataset includes phone numbers, URLs, adult/spam content, and conversational personal messages.

## Methods

The corrected implementation preserves the stratified 80/20 split with seed 42,
the k grid, and spam-class F1 selection. It fits TF-IDF and Word2Vec separately
inside each training fold, then refits the selected pipeline on the full training
split. Frozen BERT embeddings are computed independently per message. The prior
implementation fitted TF-IDF and Word2Vec before CV; its test split was still
held out, so that CV issue alone does not show the historical test results were
incorrect. See `cv_validation.md` for validation of the correction.

### TF-IDF + KNN

- TF-IDF features with unigrams and bigrams
- Maximum features: 5,000
- KNN classifier
- 5-fold cross-validation over k values: 1, 3, 5, 7, 11, 21

### Word2Vec-Style Averaged Embeddings + KNN

- Lowercasing and simple tokenization
- Word2Vec trained on each CV training partition, then refitted on the full training split (corrected implementation)
- Averaged word vectors to create 100-dimensional message embeddings
- KNN classifier
- 5-fold cross-validation over the same k values

### BERT Sentence Embeddings + KNN

- `bert-base-uncased`
- Mean-pooled sentence embeddings
- 768-dimensional embeddings
- KNN classifier
- 5-fold cross-validation over the same k values

## Evaluation

The project used:

- Accuracy
- Class-wise precision
- Class-wise recall
- Class-wise F1
- Confusion matrices
- Error analysis by false positives and false negatives

## Historical Reported Results — Corrected Rerun Pending

| Representation | Best k | Accuracy | Spam F1 | Runtime |
|---|---:|---:|---:|---:|
| TF-IDF | 1 | 0.9489 | 0.7799 | 0.45s |
| Word2Vec average | 5 in notebook / 7 in report table | 0.9596 in notebook / 0.9561 in report narrative | 0.8534 in notebook / 0.8414 in report table | 2.27s in notebook / 1.25s in report table |
| BERT | 1 | 0.9785 | 0.9216 | 25.2s in notebook / 23.0s in report table |

The historical comparison identified BERT + KNN as strongest by accuracy and spam F1, TF-IDF as fastest, and Word2Vec as a speed/quality tradeoff. These are historical claims, not results of the corrected code. Corrected runtimes include fold-local representation fitting; Word2Vec now uses one worker for reproducibility, so old timings are not directly comparable.

## Error Patterns

These are historical observations; they have not been rechecked using the corrected model selections.

- TF-IDF captures classic spam tokens such as free, win, claim, prize, URLs, and phone-number-like patterns, but misses spam without obvious trigger words.
- TF-IDF can falsely flag legitimate messages that contain promotional-looking language, phone numbers, or urgent wording.
- Averaged Word2Vec improves semantic generalization but loses word order and weakens negation.
- BERT handles context better and fixes many TF-IDF/Word2Vec errors, but still struggles with subtle spam that mimics normal conversation and legitimate messages with spam-like surface patterns.

## Privacy / Publishing Notes

Do not publish raw test examples, prediction CSVs, or error-analysis tables containing raw SMS text. The original error examples include phone numbers, URLs, adult/spam content, and personal conversational text.

