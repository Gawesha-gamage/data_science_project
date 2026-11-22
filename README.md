# Project 15: Duplicate Question Finder

**Topic:** Nearest neighbours in embedding space (Small Self-Learning Projects, section C)

## Problem

> Does nearest-neighbour search in a semantic embedding space find duplicate questions more accurately than a TF-IDF keyword baseline?

A duplicate finder takes a new question and returns questions that were already asked. Two questions are duplicates when they mean the same thing, even if the words differ.

## Data

- **Intended:** Quora Question Pairs (Kaggle). Place the file at `data/quora/train.csv`. It is public but not bundled here.
- **Used for the reported results:** a seeded **synthetic** dataset generated inside the notebook. Questions are built from action/object templates with synonyms. Two questions are duplicates when they share an action and an object. Splits are made by intent, so no test question appears in training. The generated file is saved to `data/synthetic_pairs.csv`.

The notebook uses the Quora file automatically when it is present. Results on Quora have not been produced yet.

## Method

1. **Baseline:** TF-IDF (word uni- and bi-grams) with cosine similarity.
2. **Embedding:** a 100-dimensional LSA projection of TF-IDF (truncated SVD) with cosine similarity. If `sentence-transformers` and `all-MiniLM-L6-v2` are installed, that pretrained encoder is used instead.
3. **Decision rule:** duplicate if cosine similarity ≥ a threshold. The threshold maximises F1 on the training split only.
4. **Retrieval:** for each test question, rank all other test questions and report recall@5 and MRR.

Evaluation: 5-fold CV on train for threshold stability, validation for comparing methods, one test run with a bootstrap 95% CI for F1, and error analysis of the wrong test predictions.

## Results (synthetic data, test split, 576 pairs)

| Method | F1 (95% CI) | Accuracy | ROC-AUC | PR-AUC | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| TF-IDF baseline | 0.777 (0.741–0.810) | 0.741 | 0.777 | 0.738 | 0.662 | 0.443 |
| Embedding NN (LSA-100) | 0.762 (0.727–0.795) | 0.721 | 0.767 | 0.722 | 0.672 | 0.439 |

Retrieval uses 192 query questions. The CV F1 on train is 0.788 ± 0.008 for the embedding and 0.788 ± 0.006 for TF-IDF.

**Finding:** the LSA embedding did not beat the TF-IDF baseline. The difference is within the confidence intervals. The pretrained sentence encoder named in the question has not yet been tested.

**Error analysis** (149 wrong TF-IDF test predictions):
- 100 false alarms between questions with low word overlap. Shared boilerplate phrases such as "How do I" push these over the low threshold.
- 29 missed duplicates with completely different wording (for example, "set up" vs "deploy").
- 20 false alarms with high word overlap and different meanings.

Full list: `results/errors_test.csv`.

## Limitations

- Synthetic data is easier than real Quora questions and is not comparable to published Quora results.
- LSA is not a neural sentence encoder. A pretrained model could change the outcome.
- With the Quora file, the random split can place the same question in train and test. Grouping by question ID would be stricter.
- A single global threshold is used. Calibration (Project 6) could help.
- Quora labels are noisy, so some errors come from the labels.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook duplicate_question_finder.ipynb   # or: jupyter lab
```

Run all cells from top to bottom. Runtime is under a minute on synthetic data. Set `CONFIG["quora_path"]` or place the Quora CSV at `data/quora/train.csv` to use real data. Optionally, `pip install sentence-transformers` to use the pretrained encoder.

## Files

```
duplicate_question_finder.ipynb   notebook with outputs (run in order)
requirements.txt                  pinned package versions used for the run
data/synthetic_pairs.csv          generated pairs (seed 42)
results/test_metrics.csv          final test metrics
results/validation_metrics.csv    validation metrics
results/cv_folds.csv              5-fold CV on train
results/retrieval_test.csv        recall@5 and MRR
results/errors_test.csv           wrong predictions with error types
results/pr_curve_and_scores.png   PR curve and score distribution
results/config_used.json          configuration and thresholds
```

## Privacy

No real student or staff data is used. Quora questions are public. No data is sent to any external LLM service.
