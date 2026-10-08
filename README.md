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
- The original stratified 80/20 train/test split (`random_state=42`).
- The same five stratified, unshuffled training folds for all representations, over k values: 1, 3, 5, 7, 11, and 21; selection uses mean spam-class F1.
- TF-IDF vocabulary/IDF and Word2Vec weights are learned separately inside each fold's training partition, using a scikit-learn pipeline. Validation text is only transformed. The selected pipeline is then refitted on the complete training split before test evaluation.
- BERT stays frozen (`eval`, no gradients or fine-tuning); independently encoded training messages can therefore be reused across KNN folds. Test embeddings are computed after k selection.

## Evaluation

The project uses:

- Accuracy
- Class-wise precision
- Class-wise recall
- Class-wise F1
- Confusion matrices
- False-positive and false-negative error analysis

The historical coursework reported BERT + KNN as strongest by accuracy and spam-class F1, TF-IDF as fastest, and Word2Vec as a speed/quality tradeoff. **These results have not been revalidated after the cross-validation correction.** The public notebook has no saved execution outputs, and the original CSV was unavailable during the fix. Historical numbers and their inconsistencies are preserved in `reports/project_report_scrubbed.md`; no corrected performance metrics are claimed.

Previously, TF-IDF and Word2Vec were fitted on the full training split before cross-validation, allowing validation-fold text to influence model selection. The held-out test split remained separate, so this finding alone does not establish that historical test scores are incorrect. Corrected k selections and performance require an actual rerun with the original CSV. See `reports/cv_validation.md` for the executed checks and remaining limitations.

## How To Run

1. Use Python 3.10+.
2. Install key libraries:

```bash
pip install pandas numpy scikit-learn "gensim>=4.4" torch transformers jupyter
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

Word2Vec uses `seed=42` and one worker to avoid multithreaded training variation. Set `PYTHONHASHSEED=0` **before** starting Python/Jupyter for repeatable word initialization across processes (on PowerShell: `$env:PYTHONHASHSEED="0"`). Exact results may still depend on library versions and hardware. Runtimes now include fold-local representation fitting and are not directly comparable with the historical timings.

### Lightweight validation without the dataset or BERT download

From the repository root, using Python with `numpy`, `pandas`, `scikit-learn` and `gensim>=4.4` installed:

```bash
python -m unittest discover -s tests -v
```

The checks record actual TF-IDF/Word2Vec fit inputs for every fold and final refit; verify vocabulary, IDF and word counts against only the fit rows; and verify that transforming held-out text cannot change fitted state. They also run the actual TF-IDF and Word2Vec notebook cells through the full k grid on a temporary synthetic CSV, then confirm test-only tokens are absent from the final fitted vocabularies. Synthetic data is used only to check code behavior, not to report benchmark performance. A separate fixed-vector check covers BERT's KNN path without claiming to execute BERT inference.

`src/create_clean_notebook.py` generates the committed notebook. When editing its cells, update the generator and run `python src/create_clean_notebook.py`; the tests check that both stay in sync.

## Limitations / Future Work

- The dataset is a relatively small benchmark and may not reflect current spam tactics.
- BERT adds compute cost; its historical advantage needs a corrected rerun.
- Averaged Word2Vec embeddings lose word order and weaken negation.
- Future work could compare logistic regression, linear SVM, or transformer fine-tuning against the KNN benchmark.
- Raw SMS examples should not be committed because they may contain phone numbers, URLs, adult/spam text, and personal conversational messages.

