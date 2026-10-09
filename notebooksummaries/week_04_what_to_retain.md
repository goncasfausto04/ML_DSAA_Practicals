
# Week 4: What to Retain

This is a study guide for:

- [Week 4 classification](../solutions/week_04/week_04_feature_work_classification_solution.ipynb)
- [Week 4 regression](../solutions/week_04/week_04_feature_work_regression_solution.ipynb)
- [Week 4 `course_helpers.py`](../solutions/week_04/course_helpers.py)

The notebooks contain a lot of implementation detail. The central idea is
smaller:

> Build a useful representation of the data, select features without leaking
> information from the held-out data, and keep a change only when repeated
> held-out evaluation supports it.

---

## 1. Classification versus regression

| Task | Week 4 example | Model used in the notebook | Main metric |
|---|---|---|---|
| Classification | Predict whether an athlete belongs to a class | Logistic regression | F1 score |
| Regression | Predict a used-car price | Ridge regression | Mean absolute error (MAE), in euros |

The feature-work ideas are shared, but the target changes the metric and the
statistics used by feature filters:

- **Classification** predicts a discrete label. F1 balances precision and
  recall, which is more informative than accuracy when class errors are not
  equally useful.
- **Regression** predicts a number. MAE is the average absolute prediction
  error, so an MAE of 2,996 means predictions are wrong by about 2,996 euros
  on average. Lower is better.

Do not compare an F1 value with an MAE value. They measure different tasks.

---

## 2. The correct evaluation regime

Both notebooks use the same basic experiment:

1. Draw 20 repeated 80/20 train/held-out splits.
2. Fit every learned preprocessing step on the training 80% only.
3. Transform the held-out 20% using the already-fitted training objects.
4. Fit the model and selector on the training data.
5. Score on the held-out data.
6. Compare methods split by split, not just by unrelated averages.

The most important rule is:

> If a step learns anything from data, it must be fitted inside the training
> portion of each split.

This includes:

- imputers and KNN neighbours;
- category encoders;
- scalers;
- PCA or TruncatedSVD;
- filters that use the target;
- RFE and other wrappers;
- Lasso-based selection;
- any hyperparameter or feature-count search.

### Why leakage matters

If preprocessing is fitted before the split, information from the held-out
rows influences the representation used to evaluate the model. The score then
looks better than the score expected on genuinely unseen data.

The held-out set is not a place to learn. It is a place to ask, “How well did
the decisions learned from training generalise?”

### Read the uncertainty

The notebooks report a paired difference and its standard error (SEM):

- For **MAE**, a negative difference means the new method reduced error.
- For **F1**, a positive difference means the new method improved the score.
- A difference within roughly two SEMs of zero is treated as noise rather than
  a reliable improvement.

An apparent improvement is not enough. Ask:

1. How large is the effect?
2. Is it consistent across splits?
3. Is it larger than the split-to-split uncertainty?
4. Was the choice made using the same data used to report the score?

---

## 3. Preprocessing and encoding

The model ultimately receives a numeric matrix. The notebooks therefore turn
raw columns into a model-ready representation in this order:

1. Apply the cleaning decisions recorded by Week 3.
2. Fill missing values.
3. Encode categorical values.
4. Apply numeric transforms such as `log1p`.
5. Scale numeric-valued columns when the encoder/model combination needs it.
6. Optionally reduce or select features.
7. Fit the model.

### Encoding choices

- **One-hot encoding:** creates indicator columns for categories. It does not
  invent an order between categories and is usually the safest default for
  nominal categories.
- **Ordinal encoding:** replaces categories with numbers, but the numbers can
  create an artificial order.
- **Count encoding:** replaces a category with its training-set frequency.
- **Target encoding:** replaces a category with a target-related statistic. It
  can be powerful, but it is target-aware and must be fitted using training
  rows only.

An encoded number is still a number to a penalised model. Scaling may therefore
matter after ordinal, count, or target encoding. One-hot indicators are left as
0/1 in the notebook.

The classification matrix changes between 48 and 49 columns because a very rare
category may be absent from a particular training split. This is a useful
reminder that the number of encoded columns is itself learned from the
training data.

---

## 4. Feature engineering

Feature engineering changes the representation before selection. A selector
cannot keep a feature that has not been created yet.

### Transforming a feature

`log1p(x)` compresses a right-skewed feature while remaining defined at zero.
It can reduce the influence of very large values. Other choices include square
root, power transforms, quantile transforms, and discretisation.

The transformation must be chosen by its effect on held-out performance, not
because it looks mathematically attractive.

Important distinction:

- Transforming a **feature** changes an input representation.
- Transforming the **target** changes the meaning of the error metric.

The regression notebook does not log-transform `price`, even though the target
is skewed, because MAE on `log(price)` is not measured in euros and cannot be
directly compared with the euro MAE used by the course.

### Creating a combination feature

The regression notebook creates:

```text
km_per_year = mileage / (2026 - year)
```

This represents usage intensity rather than total mileage or age alone. It is
an example of domain reasoning producing a feature with a different meaning
from either parent feature.

The unlogged feature helped, so it was retained. Applying `log1p` to it made
the result worse by about 44 euros and made it almost duplicate
`log1p(mileage)` (correlation about 0.9943). The lesson is:

> A transformation is selected per feature. A transform that helps one column
> can hurt a derived column made from it.

---

## 5. Dimensionality reduction is not feature selection

These are different operations:

- **Dimensionality reduction** creates new substitute columns, such as
  principal components.
- **Feature selection** keeps some of the original named columns and drops the
  rest.

### PCA and TruncatedSVD

- **PCA** finds orthogonal directions with the greatest variance after
  centring.
- **TruncatedSVD** is similar in spirit but does not centre the matrix, which
  makes it useful for sparse data such as text.
- A component is a mixture of source columns. It is compact but less directly
  interpretable.

The key warning is:

> Variance is not the same thing as predictive signal.

In the classification notebook, 14 PCA components retained 80% of variance
but reduced F1 from 0.8523 to 0.8191. In regression, about 8.2 PCA components
retained 80% of variance but increased MAE from 2,996 euros to 5,337 euros.
Both reducers lost on all 20 splits in the regression comparison.

Use reduction when the data is genuinely wide or redundant, such as text,
images, or sensor arrays. For a small set of named predictors, supervised
selection may preserve both performance and interpretability.

---

## 6. The three families of feature selection

### A. Filter methods

Filters rank features without fitting the final predictive model.

They are fast and simple, but a univariate filter usually sees one feature at a
time. It may miss a pair of features whose joint effect is useful.

Important filter ideas:

- **VarianceThreshold:** removes constant or near-constant columns. It measures
  spread, not usefulness. In these notebooks it removed nothing.
- **Relevance:** measures how strongly a feature relates to the target.
- **Redundancy:** measures whether two features repeat the same information.

Use a statistic appropriate to the types:

- numeric measurement versus numeric target: Pearson correlation;
- binary feature versus numeric target: point-biserial correlation;
- categorical feature versus numeric target: ANOVA and correlation ratio;
- categorical feature versus binary class: chi-square/Cramer's V;
- numeric feature versus binary class: point-biserial correlation.

A p-value is not an effect size. Also check whether category groups are large
enough for a statistic to be trustworthy.

### B. Wrapper methods

Wrappers repeatedly fit a predictive model to decide which features to keep.
They can detect joint effects that a univariate filter misses, but they cost
more computation.

- **RFE:** repeatedly fits a model, removes the least important features, and
  continues until a requested number remains.
- **RFECV:** uses internal cross-validation to help choose the number of
  features.
- **Sequential feature selection:** adds or removes features based on model
  performance and also uses validation internally.

The notebook uses RFE for the main comparison. The width chosen by searching
the same validation splits is a model-selection result, not an untouched test
estimate. Nested validation is needed for an unbiased estimate after choosing
the width.

### C. Embedded methods

Embedded methods select during model fitting.

- **L1/Lasso selection:** the penalty can shrink coefficients exactly to zero.
  The penalty strength (`C` for logistic regression or `alpha` for Lasso)
  controls how aggressively features are removed.
- **Tree importance:** a tree assigns impurity-based importances. A threshold
  such as the median is a decision rule imposed on those importances; the
  importance itself does not say where to cut.

A coefficient or importance is only a score. The threshold turns it into a
selection decision.

---

## 7. Combining selectors

The notebooks compare two ways to combine one filter, one wrapper, and one
embedded method:

- **Sequence/funnel:** the first selector removes columns permanently; the
  next selector sees only what survived. This is cheaper, but an early mistake
  can erase a useful feature.
- **Vote:** every selector sees the full training matrix, and a feature is kept
  when at least two methods keep it. This is more expensive, but one method
  cannot veto a feature alone.

The vote is the safer exported strategy for these notebooks because the
algorithm has not yet been fixed and the three families judge features in
different ways.

Results:

- Classification: the vote kept about 30 columns and scored F1 0.8525 versus
  0.8523 with all features. The difference was inside the noise. The sequence
  lost measurably, scoring 0.8492.
- Regression: the vote kept about 21.7 columns and reached 2,992 euros MAE,
  about 3.71 euros better than all features. It was the only selection result
  whose gain cleared two SEMs. The sequence lost about 42 euros.

The point is not that voting always wins. The point is to choose a combination
rule deliberately and evaluate it on held-out data.

---

## 8. What the Week 4 results actually say

### Classification

- Reduction preserved variance but lost predictive performance.
- Every fixed-budget filter was worse than using all features.
- L1 selection kept about 26 features without a measurable loss.
- The vote kept about 30 features without a measurable loss.
- Selection mainly reduced model width; it did not improve prediction.

### Regression

- The derived `km_per_year` feature was useful unlogged.
- Logging that derived feature backfired because it became redundant with
  logged mileage.
- PCA and TruncatedSVD compressed variance but badly damaged price accuracy.
- Fixed-budget filters were worse than the full matrix.
- Strong Lasso selection was roughly tied with the full matrix.
- The vote was the only selection strategy with a statistically clear,
  although commercially small, improvement.

Do not memorise the exact numbers as universal facts. They are results for these
datasets, preprocessing choices, models, and splits. Retain the reasoning that
produced them.

---

## 9. What `course_helpers.py` does

The Week 4 helper file is deliberately small. It is **not** the machine
learning pipeline. The notebook keeps preprocessing, selection, fitting, and
evaluation visible because those boundaries are what the practical is teaching.

`course_helpers.py` provides:

- `PLOT_BLUE` and `PLOT_ORANGE`: shared plot colours;
- `CleaningStep`: an immutable record containing a column, action, reason,
  affected-row count, and optional machine-readable `carries`;
- `CleaningLog`: an ordered collection of cleaning decisions;
- `CleaningLog.record(...)`: records a decision and rejects a blank reason;
- `CleaningLog.records()`: returns display-friendly dictionaries;
- `CleaningLog.to_json(path)`: saves the decisions, including `carries`;
- `CleaningLog.load(path)`: reconstructs a log from JSON;
- `CleaningLog.plan(column)`: retrieves the rule carried forward for a column;
- `len(log)`: counts recorded cleaning steps.

The important design idea is:

> The log stores a reproducible rule, not a fitted model or fitted statistic.

For example, it can store “fill these columns with the training median” or
“apply `log1p` to these columns”. Each later split still learns its own median,
categories, neighbours, and scaling values. The next notebook should carry the
strategy or formula, not a list of columns selected from one historical split.

The Week 4 copy is cumulative: it contains the helper functionality introduced
by earlier weeks, but it intentionally does not import pandas or scikit-learn.
That keeps the helper reusable and makes the actual ML decisions visible in the
notebooks.

---

## 10. Exam-style checklist

Before trusting a feature-work experiment, ask:

- Is the task classification or regression?
- What does the metric mean, and is better higher or lower?
- Was the split fixed before comparing methods?
- Was every learned step fitted on training rows only?
- Was the target-aware step kept inside the training split?
- Was the feature engineered before selection?
- Is this reduction or selection?
- Does the filter measure relevance, redundancy, variance, or something else?
- Is the statistic appropriate for the variable types?
- Did the method detect joint effects or only one-feature effects?
- If a feature count was searched, was the score reported as a selection score
  rather than an unbiased final test estimate?
- Did the effect beat the uncertainty and matter in practical units?
- Is the exported rule reproducible on future splits?

## One-sentence memory aid

> **Engineer first, fit inside the fold, select with the right question, and
> trust only repeated held-out evidence.**
