# Cross-validation correction: validation record

Validated on October 7, 2026, against the repository starting at
`99c8a007ab212bfd2fcc5280ff20cc659ea9954e`.

## Change

The old notebook fitted TF-IDF and Word2Vec on the entire training split before
five-fold KNN selection. Validation-fold messages could therefore affect the
vocabulary, IDF, and Word2Vec weights. The corrected `tune_knn` fits a complete
representation-plus-KNN pipeline inside `GridSearchCV`. Each candidate/fold gets
a fresh clone; the selected pipeline is then fitted on the full training split.
This follows [scikit-learn's guidance on preventing preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

The stratified 80/20 split, seed 42, unshuffled five-fold stratification, k grid,
spam F1 objective, feature settings, tokenization, and mean-pooling comparison
are retained. One fold list is shared by all three representations. Word2Vec
uses one worker instead of four to reduce training nondeterminism; a fixed
`PYTHONHASHSEED` is also needed before process startup. Timing comparisons with
historical runs must account for this and repeated representation fitting.

BERT remains an external frozen `bert-base-uncased` encoder: evaluation mode,
disabled gradients, no corpus fitting, no fine-tuning, and no learned postprocessing.
Its training embeddings are computed independently per message before KNN CV.
Test embeddings are computed after k selection. No test data or labels enter
`tune_knn` for any representation.

## Executed checks

Command from the repository root:

```bash
PYTHONHASHSEED=0 python -m unittest discover -s tests -v
```

**Result: all six tests passed.** The tests use actual scikit-learn and Gensim;
recording wrappers observe fits while still executing the real implementations.

| Check | Evidence |
|---|---|
| TF-IDF fold isolation | Actual fit inputs match exactly each training fold for each candidate, plus one full-training refit. Vocabulary and smoothed IDF match only those fit rows. |
| Word2Vec fold isolation | Actual fit inputs match the same expected folds/refit. Learned vocabulary, token counts and corpus count match only fit rows. |
| Read-only held-out transformation | Validation text and repeated held-out/known words leave TF-IDF vocabulary/IDF and Word2Vec vocabulary/vectors unchanged; training features are unchanged. Unknown-only Word2Vec messages return zero vectors. |
| Frozen-feature KNN selection | Synthetic fixed features produce the same CV F1 means/stds and selected k as the former KNN-only selection calculation. This does not execute BERT inference. |
| Notebook/source consistency | Regeneration exactly matches the committed notebook; every code cell compiles and all saved execution outputs remain empty. |
| Notebook execution smoke check | Actual data/split/evaluation/TF-IDF/Word2Vec cells run on an 80-row temporary synthetic CSV, using all six k values and five folds. Both final pipelines predict the 16 test rows; their vocabularies exclude every test-only sentinel token. |

Additional checks passed: `nbformat.validate`, Python source compilation,
`git diff --check`, and clean startup from `notebooks/` reaching the expected
missing-CSV error. The synthetic fixture lives in a temporary directory and
does not replace or populate `data/spam.csv`.

Environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0,
scikit-learn 1.8.0, Gensim 4.4.0, nbformat 5.11.1.

## Not executed / results policy

The original `data/spam.csv` was absent from the checkout and accessible local
files; searches for an uploaded copy returned no matches. No substitute corpus
was used to make benchmark claims. Torch, Transformers and pretrained BERT
weights were unavailable in this execution environment; BERT inference was
statically reviewed, not executed.

No original-data benchmark was rerun, and no corrected accuracy, F1, selected k,
or runtime is reported. Synthetic scores are implementation checks only. The
historical report's numbers and conflicting Word2Vec entries are preserved,
with a clear notice that the corrected pipeline has not revalidated them.
The CV defect alone does not establish that historical test scores were wrong:
the original test split was separate.

## Complete the original-data rerun

1. Place the original CP1252 CSV, with its original row order, at `data/spam.csv`.
   Do not substitute a similarly named dataset if claiming comparability.
2. Install the dependencies from the README and set `PYTHONHASHSEED=0` before
   starting Python/Jupyter. Record dependency versions, device and CSV hash.
3. Restart the kernel and run all cells in order, keeping the same split and
   hyperparameter grid. Use the held-out split only for the final evaluation;
   do not adjust k or representation settings in response to test scores.
4. Retain executed aggregate outputs separately from the clean source notebook.
   Update reported metrics only from those outputs, distinguishing the corrected
   run from historical figures. Do not publish raw messages or prediction files
   containing message text.
