# Week 5 — Choosing Models Fairly and Measuring Performance

Sources: [model selection solution](../solutions/week_05/week_05_model_selection_solution.ipynb) · [performance measures solution](../solutions/week_05/week_05_performance_measures_solution.ipynb)

## The big idea

Two jobs must use **different data**:

- **Selecting** (choosing a model, a setting, a threshold)
- **Estimating** (saying how good the chosen thing really is)

The score you used to pick the winner is **too optimistic**, because you picked whatever happened to do best on those rows, luck included. That's why we have test sets and nested cross-validation.

---

## Part 1 — Model selection

### Cross-validation (CV)

One single train/validation split is a sample of size one: you might just get a lucky or unlucky split. **K-fold CV** cuts the training data into K pieces, trains on K−1 and validates on the remaining one, rotates K times and averages. Every row gets used for validation exactly once.

| Variant | When to use it |
|---|---|
| **Stratified K-fold** | Classification: keeps class proportions in every fold |
| **Repeated K-fold** | Run K-fold several times with different cuts to see how much scores vary |
| **Leave-one-out** | K = number of rows. Very expensive |
| **GroupKFold** | Rows come in groups (same person, same competition). A whole group stays on one side, so you test on **unseen groups** |
| **TimeSeriesSplit** | Time matters. Always train on the past and validate on the future |

**The split is a modelling assumption.** Pick it based on how the model will be used in real life ("new athletes?", "new competitions?", "next year?"), **not** based on which one gives the best score.

### Everything goes inside the folds

Imputation, encoding, scaling and feature selection must all be **refitted inside each training fold**. A `Pipeline` does this automatically. If anything is fitted outside the folds, it has already seen the validation rows, which is leakage.

### Tuning hyperparameters: grid search

**Grid search** tries every combination of the settings you list, scores each with CV, and keeps the best. At the end it **refits the winner on all training data**, so the search hands you a finished model ready to use.

### Nested cross-validation (the gold standard)

Problem: the best grid-search score is optimistic (it's the best of many tries). Solution:

- **Outer loop:** splits the data into folds for *estimating*.
- **Inner loop:** inside each outer training part, runs the whole grid search to *choose* settings.
- The chosen model is then scored on the outer fold it never saw.

The inner loop **chooses**; the outer loop **grades the whole choosing procedure**. Nested CV doesn't give you a model. It gives you an honest score for "running this search". To get the final model you run the search once more on all training data.

---

## Part 2 — Performance measures

**Choose the metric before seeing results**, based on which mistakes cost more.

### Regression metrics (lower is better, except R²)

| Metric | Plain meaning |
|---|---|
| **MAE** | Average size of the miss, in real units (euros). Easiest to explain |
| **RMSE** | Like MAE but **big misses count much more** (errors are squared first) |
| **MedAE** | The typical miss; ignores the rare huge ones |
| **MAPE** | Average miss as a percentage; breaks when true values are near zero |
| **R²** | Share of variation explained. 1 = perfect, 0 = no better than predicting the average |

MAE and RMSE can rank models differently: one model might have small usual errors but a few disasters, which RMSE punishes.

### Classification metrics

- **Accuracy**: share of all predictions that are correct.
- **Precision**: when I say YES, how often am I right?
- **Recall**: of the real YES cases, how many did I catch?
- **F1**: a balance of precision and recall.
- **F-beta**: like F1 but you choose the emphasis. β > 1 cares more about recall, β < 1 cares more about precision.

**F1 trap:** F1 ignores true negatives. Here 60% of athletes win, so "predict WIN for everyone" already gets F1 ≈ 0.75. Always compare against that floor. That's why the course reports **F1, ROC-AUC and balanced accuracy** together.

### Thresholds

A classifier really outputs a **probability**. The **threshold** (default 0.5) turns it into YES/NO.
- Lower threshold → more YES → higher recall, more false alarms.
- Higher threshold → fewer YES → higher precision, more misses.

The model doesn't change at all, only where you cut. The threshold is a choice too, so pick it on validation data.

### ROC and precision-recall curves

- **ROC curve:** catch rate vs false-alarm rate across all thresholds. **AUC** = area under it. 0.5 means random guessing and 1 means perfect **ranking**. It measures how well the model *orders* cases, not whether its probabilities are accurate.
- **Precision-recall curve:** focuses on the positive class. More informative when positives are rare.

### More than two classes (optional)

Compute the metric per class, then average:
- **Macro:** every class counts equally, so rare classes matter.
- **Weighted:** big classes count more, which can hide failure on rare classes.
- **Micro:** pool all the counts first. For normal single-label problems this equals accuracy.

## Check yourself

1. Why isn't the best grid-search score an honest final estimate?
2. When would you use GroupKFold or TimeSeriesSplit instead of normal K-fold?
3. What does nested CV estimate, and how do you get the final model?
4. Why can MAE and RMSE prefer different models?
5. Why can "always predict win" get a decent F1?
6. What happens to precision and recall when you lower the threshold?
