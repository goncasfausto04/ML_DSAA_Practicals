# Machine Learning — Week 2: The Complete ML Pipeline

## Glossary

| Key term | Meaning |
|---|---|
| Training / fitting | Learning model parameters from training examples |
| Training set | Data used to fit the model |
| Validation set | Held-out data used to compare models or settings |
| Test set | Untouched data reserved for final assessment |
| Holdout | Data set aside from fitting to check generalization |
| Stratified split | A split that preserves class proportions |
| `random_state` | A seed used to make random operations reproducible |
| Parameter | A value learned by the model during fitting |
| Hyperparameter | A setting chosen by the practitioner, such as tree depth |
| Decision tree | A model made of learned threshold-based if/then rules |
| Overfitting | Learning training-specific noise and generalizing poorly |
| Underfitting | Being too simple to capture meaningful patterns |
| Tuning | Comparing hyperparameter settings using validation evidence |
| Accuracy | The fraction of predictions that are correct |
| Precision | Of predicted positives, the fraction that are truly positive |
| Recall | Of actual positives, the fraction that are found |
| F1 | The harmonic mean of precision and recall |
| Deployment | Applying the chosen model to new real-world cases |

# Week 2 — The Complete ML Pipeline

## The takeaways

1. **ML is a workflow, not `.fit()` alone:** business need → data → exploration → preparation → model → optimization → assessment → deployment.
2. **Split before fitting learned transformations.** Training data teaches; held-out data checks generalization.
3. **A decision tree learns if/then splits.** An unconstrained tree can overfit.
4. **Parameters are learned; hyperparameters are chosen.** Tune settings such as `max_depth` on validation data.
5. **Choose metrics based on the decision.** Accuracy, F1, and the confusion matrix answer different questions.
6. **Predictions must be usable.** A ranked probability list is more useful than a hard class label when inspectors have limited capacity.

## Core concepts

| Concept | Meaning |
|---|---|
| Training | Learning model parameters from examples |
| Fitting | Calling an algorithm to learn its parameters from training data |
| Holdout | Set aside data not used to fit the model for evaluation |
| Training set | Data used to learn the model |
| Validation set | Data used to compare candidate models/settings |
| Test set | Untouched data for final assessment after model selection |
| Stratified split | Preserve class proportions across subsets |
| `random_state` | Seed for reproducible randomness |
| Decision tree | A series of learned threshold-based if/then rules |
| Parameter | A value learned during fitting, e.g. a tree split threshold |
| Hyperparameter | A setting selected by the practitioner, e.g. maximum depth |
| Overfitting | Learning training-specific details/noise; poor generalization |
| Underfitting | Model too simple to capture meaningful patterns |
| Bias–variance tradeoff | Balance between oversimplifying and chasing sample-specific variation |
| Tuning | Comparing hyperparameter settings using validation evidence |
| Deployment | Applying the chosen model to real new cases |

### Understand the split

Week 2 uses **70% training / 30% validation**, with `stratify=y`. Training rows fit the tree; validation rows estimate performance and compare depths. **Important nuance:** once you use validation scores to select a depth, that validation set is *not* an untouched final test set. A separate test set or nested cross-validation is needed for a less biased final estimate.

**Never do this:** compute medians, fit encoders, or select features on the whole dataset before splitting. Even without reading validation labels, this leaks information about validation inputs.

### Decision trees and overfitting

A decision tree asks questions like `BD1 > threshold?`, follows branches, and predicts at a leaf. Deep trees can memorize cases; shallow trees may miss real patterns.

The notebook's default tree reached approximately **1.000 training accuracy vs 0.833 validation accuracy**, and **1.000 training F1 vs 0.694 validation F1**. That gap is evidence of overfitting. Restricting `max_depth` can improve generalization even if training accuracy falls.

In the notebook, depth **4** was preferred to depth **5** because validation F1 was about **0.784 vs 0.756**. **Lesson:** choose on held-out performance, not training score or complexity alone.

### Classification metrics you must know

Confusion matrix for binary classification (`1` = positive):

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| **Actual 0** | TN (correct negative) | FP (false alarm) |
| **Actual 1** | FN (missed positive) | TP (correct positive) |

- **Accuracy** = `(TP + TN) / (TP + TN + FP + FN)` — fraction correct.
- **Precision** = `TP / (TP + FP)` — of flagged plantations, how many really are drug plantations?
- **Recall** = `TP / (TP + FN)` — of actual drug plantations, how many did we catch?
- **F1** = `2 × precision × recall / (precision + recall)` — balances precision and recall (harmonic mean).

A **false negative** misses a drug plantation; a **false positive** wastes inspection capacity. The real decision maker must decide how these costs compare. F1 balances precision and recall, but **does not directly encode the business costs** of each error.

### Hard predictions versus probabilities

- `.predict(X)` returns class labels, such as `0` or `1`.
- `.predict_proba(X)` returns estimated probabilities for classes. Use the class-1 column to **rank** the 40 unlabelled plantations by priority.
- **Caution:** a model's probability output is not automatically well calibrated. A predicted `0.8` should not be treated as an objectively guaranteed 80% chance without calibration checks.

### Essential scikit-learn pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix

X = df[["BD1", "BD2", "BD3", "BD4"]]
y = df["DrugPlant"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_val)
print(f1_score(y_val, pred))
print(confusion_matrix(y_val, pred))
```

The seed `42` is an illustrative code value, not a claimed notebook setting. `.score()` for this classifier returns accuracy; do **not** assume `.score()` always returns F1 or the same metric for every estimator. The notebook also uses `joblib` to save and reload the fitted model. In real use, save preprocessing **together** with the model.

### Check yourself

1. Why is a 100% training score not necessarily good news?
2. What does `stratify=y` protect against?
3. Why can validation not be called an untouched final test after tuning?
4. When is recall more important than precision? When is precision more important?
5. Why use `predict_proba` to make an inspection list?

**Week 2 in one sentence:** Protect held-out data, control model complexity, evaluate the mistakes that matter, and deliver predictions in a useful format.

---

