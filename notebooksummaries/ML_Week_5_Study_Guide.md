# Machine Learning — Week 5: Model Selection and Performance Measures

Sources: [Model selection solution](../solutions/week_05/week_05_model_selection_solution.ipynb) · [Performance measures solution](../solutions/week_05/week_05_performance_measures_solution.ipynb).

## Key learning points

1. **Selection and assessment need separate data.** The score used to choose a model, setting, or threshold is optimistic in expectation as an estimate of the winner's future performance.
2. **Choose the splitter from the prediction problem.** New random cases, unseen groups, and future observations require different validation designs.
3. **Fit the complete pipeline inside every training fold**, including imputation, encoding, scaling, and feature selection.
4. **Choose metrics before seeing scores**, based on the consequences of errors. Different metrics can prefer different models.
5. **A threshold is another choice to validate.** Changing it changes labels and metrics without retraining the model.

## Notebook 1 — Model selection

### Train, validation, and test

- **Training:** fit preprocessing and model parameters.
- **Validation:** choose model family and hyperparameters.
- **Test:** independently assess the final choice; do not use it to tune.

The notebook reserves 20% for testing, then takes 25% of the remaining 80% for validation: **60% train / 20% validation / 20% test**. Classification splits are stratified.

After selecting a configuration, **refit the whole pipeline on train + validation**, then assess it on the test set. The notebook displays test scores repeatedly for teaching; using those scores to make further choices would compromise their independence.

### Cross-validation: choose the right design

| Design | How it works | Main lesson |
|---|---|---|
| Holdout | One training/validation split | Cheap, but depends heavily on one partition |
| K-Fold | Train on K−1 folds and validate on the remaining fold; rotate | Uses every row for validation once |
| Stratified K-Fold | K-Fold preserving class proportions | Useful for classification, especially imbalanced classes |
| Repeated K-Fold | Redraw folds and repeat CV | Shows sensitivity to partitions; costs more fits |
| Leave-One-Out | One validation row per fit | Expensive; pool held-out predictions for F1 rather than average one-row F1 scores |
| GroupKFold | Keep each group entirely on one side of a split | Estimates performance on unseen groups |
| TimeSeriesSplit | Train on earlier observations, validate on later ones | Matches prediction into the future; preserve chronological order |

The notebook groups by competition level and orders time splits by `Edition`. Athlete IDs have already been deduplicated, so they do not define repeated groups here.

**Do not choose a splitter because it gives the highest score.** Unseen competition levels are a harder, different question from new athletes in familiar levels. A random test split cannot fairly judge an estimate for a different deployment setting.

### Compare and tune fairly

- **Parameters** are learned during fitting; **hyperparameters** are chosen settings such as tree depth or logistic regression `C`.
- Compare candidates using the **same folds and metric**. A large training–validation gap suggests overfitting; poor scores on both suggest underfitting.
- **GridSearchCV** tries every supplied configuration. A `Pipeline` ensures preprocessing is refitted within each fold; step parameters use names such as `classifier__C`.
- `best_score_` is the winning mean CV score used for selection. With refitting enabled, `best_estimator_` is fitted on all data passed to the search.
- Caching identical preprocessing **within each fold** avoids redundant fits. Never reuse preprocessing fitted across validation boundaries. Standard and cached searches should agree when their computation is equivalent.
- Optional alternatives: randomized search samples configurations; successive halving removes weak candidates early.

### Nested cross-validation

**Outer training fold → inner CV selects settings → refit winner on outer training rows → score once on outer held-out rows.** Repeat across outer folds.

The inner loop **selects**; the outer loop **assesses the entire selection procedure**. After assessment, run the search on all development data to obtain the final fitted model.

Nested CV reduces selection optimism, but its score need not be lower than ordinary CV on every run. Report variation as well as the mean: small score differences within the fold spread do not establish a clear winner. Changing validation design alone does not change a fixed model refitted on the same rows.

## Notebook 2 — Performance measures

### Regression: know the errors and their units

Let `e = y − ŷ`, `n` be the number of observations, and `p` the number of predictors.

| Metric | Formula | Interpretation / limitation |
|---|---|---|
| R² | `1 − Σe² / Σ(y − mean(y))²` | Higher is better; 1 is perfect, 0 matches predicting the evaluation-set mean, negative is worse; assumes a nonconstant target |
| Adjusted R² | `1 − (1 − R²)(n − 1)/(n − p − 1)` | Penalizes predictor count in linear regression; requires `n > p + 1`; does not replace held-out evaluation |
| MAE | `mean(abs(e))` | Average miss in target units; easy to interpret |
| MSE | `mean(e²)` | Strongly penalizes large misses; squared target units |
| RMSE | `sqrt(mean(e²))` | Also emphasizes large misses, in target units |
| MedAE | `median(abs(e))` | Typical absolute miss; robust but can hide severe tail errors |
| MAPE | `mean(abs(e / y))` | Relative error; unstable near zero and unsuitable when targets can be zero |

All error metrics above are **lower is better**. scikit-learn returns MAPE as a fraction: `0.10` means 10%.

For car prices, MAE and RMSE are in euros. RMSE penalizes a few huge pricing mistakes more strongly, so it can rank models differently from MAE. A high R² does not guarantee small euro errors or good performance for every price range.

### Binary classification: define the positive class

For the athletes, `Outcome = 1` means a win. **TP** is a correctly predicted win, **FP** a predicted win that was a loss, **FN** a missed win, and **TN** a correctly predicted loss.

| Metric | Formula | Question |
|---|---|---|
| Accuracy | `(TP + TN) / N` | What fraction of all predictions are correct? |
| Precision (`P`) | `TP / (TP + FP)` | Of predicted positives, how many are correct? |
| Recall (`R`, sensitivity/TPR) | `TP / (TP + FN)` | Of actual positives, how many are found? |
| F1 | `2PR / (P + R)` | How well are precision and recall balanced? |
| F-beta | `(1 + β²)PR / (β²P + R)` | How well are they balanced with a chosen emphasis? |

`β > 1` emphasizes recall; `β < 1` emphasizes precision. These formulas need a convention when denominators are zero.

**F1 ignores TN.** When positives are common, always predicting positive can already achieve high F1. If the positive share is `q`, this baseline has precision `q`, recall `1`, and F1 `2q/(1+q)`. Always report a relevant baseline.

### Probabilities, thresholds, and curves

Predict positive when the positive-class score reaches a threshold. Lowering the threshold flags more cases and cannot reduce recall on fixed data; precision can fluctuate. Raising it generally favors fewer false alarms at the expense of missed positives.

| Tool | What it shows | What to remember |
|---|---|---|
| ROC curve | TPR (recall) against `FPR = FP/(FP + TN)` across thresholds | Summarizes ranking across both classes; can look favorable despite poor precision for rare positives |
| ROC AUC | Area under ROC | Higher is better; random ranking is about 0.5; use scores/probabilities rather than hard labels |
| Precision–recall curve | Precision against recall across thresholds | Focuses on positives; random-ranking baseline is their prevalence |
| Youden's J | `TPR − FPR` | ROC-based threshold criterion; need not maximize F1 or minimize business costs |
| Maximum F1 | Threshold with highest validation F1 | A different objective can choose a different threshold |

Report the chosen threshold **together with its metrics**. Choose it on validation data and evaluate it on independent data. AUC measures ranking, not probability calibration or performance at a particular operating threshold.

### Optional: multiclass averages

Start with the **confusion matrix** (actual classes in rows, predictions in columns), then inspect per-class metrics.

- **Macro:** average class scores equally; highlights performance on rare classes.
- **Weighted:** average class scores by true class frequency; can hide failures on rare classes.
- **Micro:** pool TP/FP/FN before scoring. For single-label classification covering all classes, micro precision = micro recall = micro F1 = accuracy.

Use `classification_report` for per-class and aggregate scores. Choose averaging based on class importance, rather than whichever average looks best.

## Check yourself

1. Why is the best grid-search CV score insufficient as an independent final estimate?
2. When should you use group-aware or time-aware splitting?
3. What does nested CV assess, and how do you obtain the final model afterward?
4. Why might MAE and RMSE choose different models?
5. Why can always predicting “win” give a deceptively high F1?
6. Why can Youden's J and maximum F1 select different thresholds?
7. When is macro F1 more informative than weighted F1?
