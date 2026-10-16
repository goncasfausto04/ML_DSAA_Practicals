# Machine Learning — Week 2: The ML Pipeline

Source: [ML pipeline solution](../solutions/week_02/week_02_ml_pipeline_solution.ipynb).

## Key learning points

1. Follow the workflow: **business need → data → exploration → preparation → modelling → tuning → assessment → deployment**.
2. Split before fitting. Learn transformations and model parameters from training rows only.
3. Choose model settings using validation performance, then assess the final choice on an untouched test set.
4. Match the metric and prediction format to the decision being made.

## Training, validation, and complexity

| Term | Meaning |
|---|---|
| Training set | Fits preprocessing and model parameters |
| Validation set | Compares models and hyperparameters |
| Test set | Assesses the final choice independently |
| Parameter | Learned during fitting, such as a tree's split threshold |
| Hyperparameter | Chosen setting, such as `max_depth` |
| Stratification | Preserves class proportions across splits |
| `random_state` | Makes random operations reproducible |

The notebook uses a **70/30 training/validation split** with stratification. Once used for tuning, validation is no longer an independent final assessment.

A **decision tree** learns branching if/then rules. Deep trees can **overfit**: excellent training performance but worse validation performance. Trees that are too shallow can **underfit**, missing useful patterns on both sets. The notebook's perfect training score did not imply perfect generalization; limiting depth improved validation F1.

## Classification metrics

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | TN | FP |
| Actual positive | FN | TP |

- **Accuracy:** `(TP + TN) / N` — fraction correct.
- **Precision:** `TP / (TP + FP)` — how many flagged cases are truly positive?
- **Recall:** `TP / (TP + FN)` — how many actual positives are found?
- **F1:** `2PR / (P + R)` — harmonic mean of precision and recall; does not directly represent business costs.

For inspections, FP wastes a visit and FN misses a drug plantation. Choose priorities from those consequences.

## From fitting to use

- `.fit(X_train, y_train)` learns; `.predict(X_val)` returns class labels.
- `.predict_proba(X)` returns estimated class probabilities. Rank the positive-class probabilities when inspection capacity is limited.
- Probability estimates need calibration checks before being interpreted as reliable probabilities.
- A classifier's `.score()` commonly returns accuracy; request F1 explicitly when needed.
- Save preprocessing together with the model so new data receives the same treatment.

## Check yourself

1. How do parameters differ from hyperparameters?
2. What does a large training–validation gap suggest?
3. Why use probabilities to prioritize inspections?
4. Why must a test set remain untouched during tuning?
