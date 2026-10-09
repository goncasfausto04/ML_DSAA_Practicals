# Week 3: What to Retain

This guide condenses:

- [Week 3 classification](../solutions/week_03/week_03_deepen_exploration_solution.ipynb)
- [Week 3 regression](../solutions/week_03/week_03_deepen_exploration_regression_solution.ipynb)
- [Week 3 `course_helpers.py`](../solutions/week_03/course_helpers.py)

Week 3 is the data-quality week. It establishes the data that later weeks
will model.

> **Explore first, define valid values, compare alternatives, choose a recipe,
> and record the decision.**

The most important idea is that cleaning is not a collection of automatic
fixes. Cleaning choices change the learning problem and must be justified by
domain knowledge and held-out evidence.

---

## 1. What Week 3 is doing

The notebooks start with raw files and produce reusable cleaned files:

| Task | Raw data | Clean output | Target |
|---|---:|---:|---|
| Classification | 4,280 competition rows | 4,000 rows in `champions.csv` | `Outcome` (won or not) |
| Regression | 4,600 car listings | 4,000 rows in `cars4you.csv` | `price` in euros |

The output file is not “the data as it really is”. It is the result of a
documented sequence of decisions. Every later model inherits those decisions.

The reusable process is:

```text
inspect -> classify columns -> define domain rules -> compare treatments
-> select a recipe -> apply reproducibly -> log the recipe
```

---

## 2. Explore before changing anything

First establish what every column means and what role it should have:

- **target:** what the model predicts;
- **identifier:** identifies a row or entity but should not be a predictor;
- **numeric measurement:** a quantity with meaningful arithmetic;
- **count:** numeric, but constrained to whole non-negative values;
- **nominal category:** labels without a natural order;
- **ordinal category:** labels with an order;
- **boolean:** two valid states.

Do not infer a column's role only from its pandas dtype. The notebooks contain
numbers that are labels and text that represents booleans.

Examples:

- `RecordID`, `Athlete Id`, and `CarID` identify rows/entities. They are not
  model features.
- `Edition` is a category, even though it is stored numerically.
- `Previous attempts` is a count.
- `Education`, `Income`, and most `Age group` levels are ordered categories.
- Boolean columns may arrive as strings such as `"TRUE"` and `"FALSE"`.

Inspect:

- shape and data types;
- unique values and category counts;
- missing-value counts;
- duplicates and repeated entities;
- quantiles, skew, and distributions;
- the target distribution;
- suspicious or impossible values.

An ordinary duplicate-row check is not enough. The classification data has no
duplicate rows, but some athletes appear in several rows. That can allow the
same person to appear in both training and held-out data, which is a form of
information leakage.

---

## 3. Domain validity comes before statistics

Define valid values from what a column measures:

- percentages must stay within their scale;
- prices, distances, tax, and counts cannot be negative;
- strictly positive measurements cannot be zero;
- years must be plausible whole numbers;
- categories must use accepted spellings;
- booleans must use valid states.

An impossible value is not automatically the same thing as missing data. First
identify it, then decide whether it should be:

- repaired;
- replaced with `NaN`;
- clipped to a boundary;
- flagged with an indicator;
- or removed with its row.

For classification, the value `FASE` is a misspelling of `FALSE`, and `0` in
`Age group` is a placeholder rather than a genuine age band. The negative
athlete score is an administrative code, not a meaningful score.

For regression, invalid category spellings include malformed `Brand`,
`transmission`, and `fuelType` values. Numeric domain checks cover year,
mileage, tax, mpg, engine size, paint quality, previous owners, and price.

The target is checked against its own domain too. A target value that is
impossible is not allowed simply because it is the value the model predicts.

---

## 4. Outliers: a diagnostic is not a deletion command

The standard boxplot rule marks values outside:

```text
Q1 - 1.5 * IQR       to       Q3 + 1.5 * IQR
```

This is a screening rule, not proof that a value is wrong.

It fails when a variable has a large spike at one value. For example, several
training variables contain mostly zero minutes and a smaller group of genuine
non-zero training values. Then `Q1 = Q3 = 0`, so the IQR is zero and every
non-zero value is labelled an outlier.

A rule that removes most of the data, removes one class more often than another,
or removes real hybrids/expensive cars is changing the population rather than
cleaning it.

Better questions are:

- Is the value impossible or merely unusual?
- Is the tail meaningful?
- Does a treatment preserve the rows that matter?
- Does it improve held-out performance?
- Does it alter class balance or the target distribution?

Often the right treatment is to **reshape** a valid long tail rather than
delete it. `log1p(x)` compresses large values and preserves every row.

---

## 5. The evaluation protocol

Cleaning alternatives are compared using repeated, paired held-out evaluation:

1. Draw the same 20 repeated 80/20 splits for every candidate.
2. Fit the candidate treatment on the training rows only.
3. Apply it to training rows and untouched held-out rows.
4. Fit the simple model used as a measuring instrument.
5. Score the same held-out rows for every candidate.
6. Compare the paired score differences and their standard error.

The model used in Week 3 is mainly a measuring instrument for cleaning choices:
logistic regression for classification and ordinary least squares for
regression. Week 3 is not primarily about choosing the final model.

### Why pairing matters

Every alternative is evaluated on the same splits. The shared difficulty of
each split cancels when scores are subtracted, making the treatment difference
easier to see.

Read the mean and SEM together:

- a difference much larger than its SEM is evidence of a real effect;
- a difference about the size of its SEM is noise, not a reason to prefer one
  treatment;
- a result from one lucky split is not enough.

### Leakage rule

Anything that learns from data must be fitted inside the training split:

- medians, modes, and KNN neighbours;
- category replacements or encoders;
- outlier fences;
- scalers;
- transformations with learned parameters;
- any target-related statistic.

The finished file deliberately keeps some values missing and stores a **recipe**
for filling them later. If Week 3 filled the entire dataset first, later weeks
would use values computed partly from their future held-out rows.

---

## 6. Classification decisions

The classification recipe is:

1. Repair the boolean misspelling (`FASE` to `FALSE`).
2. Blank impossible values rather than clipping or dropping rows.
3. Keep the earliest `RecordID` for each athlete.
4. Leave missing values open in the cleaned file.
5. Carry forward median/mode filling for later training splits.
6. Apply `log1p` to the eleven training-volume columns.

### Invalid values

Blanking and clipping the impossible athlete score performed identically because
the training median was zero. Keeping the impossible value and adding a flag
was not meaningfully different. Dropping the affected rows was worse because
those rows contained winners at a higher rate than the overall data.

The correct lesson is not “always blank”. It is:

> When the competing treatments are statistically indistinguishable, prefer
> the treatment that preserves rows and makes the least unsupported claim.

### Repeated athletes

Keeping every row risks the same athlete appearing in training and held-out
data. Keeping the earliest `RecordID` is chosen for reproducibility, not
because it proved more accurate. Arbitrary row-selection rules performed
similarly.

This is an important distinction:

- **deduplication is required for valid evaluation;**
- the particular tie-break rule is mainly a reproducibility decision here.

### Missing values

The classification data has missing values across many columns. The notebook
compares median/mode, zero fills, an explicit missing category, proportional
category fills, KNN, and iterative imputation.

Median/mode is retained. More elaborate KNN and iterative fills did not produce
a meaningful gain on this dataset. A simple method is preferable when extra
complexity has no evidence behind it.

### Outlier treatment

`log1p` on the training-duration columns improved F1 from about 0.8182 to
0.8506, a paired improvement of about 0.0325, on all 20 splits.

Dropping IQR outliers removed too many rows and changed the class balance.
The model needs to see legitimate high-training athletes; deleting them
removes signal.

### Classification metric

F1 balances precision and recall:

```text
F1 = 2 * precision * recall / (precision + recall)
```

Higher is better. F1 is useful when both missed positives and false positives
matter, especially when class balance makes accuracy misleading.

---

## 7. Regression decisions

The regression recipe is:

1. Validate category spellings and numeric domains.
2. Drop rows containing invalid values rather than silently repairing malformed
   car records.
3. Drop the high-cardinality `model` feature.
4. Leave missing values open in the cleaned file.
5. Carry forward KNN imputation (`k=5`) for numeric columns and an explicit
   missing level for categories.
6. Apply `log1p` to `mpg` and `mileage`.
7. Record no scaler for ordinary least squares.

### Invalid rows

Repairing malformed labels beat dropping the affected rows by about 19.34
euros on 19 of 20 splits. However, the final domain rule drops the invalid
rows because the malformed records cannot be trusted as valid car listings.
The measured cost of that external data-quality constraint is about 0.6%.

This demonstrates that predictive performance is not the only criterion:
domain validity can override a slightly better score, but the cost should be
measured and reported.

### High-cardinality categories

`model` has hundreds of levels. One-hot encoding it would create a very wide
matrix, with many model levels supported by only a few cars. The notebook drops
it because `Brand` carries much of the useful information at a more manageable
width.

Cardinality is a structural modelling cost that can be identified before fitting
a model.

### Missing values

KNN imputation wins for the numeric car fields: it is about 37 euros better than
median/own-level filling on 19 of 20 splits. Here, the other car variables
contain useful information about a missing numeric value.

The categorical columns use an explicit `(missing)` level. This preserves the
fact that the value was absent instead of pretending it was the most common
category.

The classification and regression datasets choose different imputers. That is
not inconsistent. Imputation is data-dependent and must be measured for the
file being modelled.

### Outlier treatment

The IQR rule removes valid cars, including cheap cars and hybrids. `log1p` on
`mpg` and `mileage` compresses their long tails without deleting observations.

### Scaling and ordinary least squares

For a full-rank ordinary least-squares model, rescaling a feature mainly
rescales its coefficient, so predictions should be theoretically unchanged.
The notebook confirms that the MAE differences between scalers are much smaller
than the uncertainty.

Scaling still matters for:

- KNN, because distances depend on scale;
- Ridge and Lasso, because penalties depend on coefficient size;
- gradient-based optimisation, because step sizes depend on feature scale.

The regression notebook is a useful control case: it shows when scaling should
not change the fitted predictions, while later models show when it does.

### Target skew

`price` is strongly right-skewed, and `log1p(price)` makes it more symmetric.
But transforming the target changes the question:

- error on raw price is measured in euros;
- error on log-price is closer to proportional error.

The notebook measures the target's shape but does not apply the target
transformation, because the course's comparison metric is euro MAE.

### Regression metric

Mean absolute error is:

```text
MAE = average(abs(actual - prediction))
```

It is expressed in euros and is easy to explain. RMSE penalises large errors
more heavily:

```text
RMSE = sqrt(average((actual - prediction)^2))
```

Lower is better for both.

---

## 8. `course_helpers.py`

The Week 3 helper module is deliberately not an ML pipeline. It contains
shared bookkeeping used by the notebooks:

- `PLOT_BLUE`, `PLOT_ORANGE`, and `CLASS_COLOURS` provide consistent plot
  colours.
- `CleaningStep` is an immutable record of one decision:
  column, action, reason, affected-row count, and optional `carries`.
- `CleaningLog` stores the ordered cleaning decisions.
- `record(...)` adds a decision and rejects a blank reason.
- `records()` returns display-friendly dictionaries.
- `to_json(path)` saves the complete recipe, including machine-readable
  `carries`.
- `load(path)` reconstructs a log from JSON.
- `plan(column)` retrieves the rule later notebooks need to apply.
- `len(log)` counts decisions.
- `repr(log)` gives a compact dataset/step summary.

The key design idea is:

> Store the rule, not the fitted value.

For example, store “fill these numeric columns with a training median” rather
than one median computed from the whole dataset. Later folds must calculate
their own values using only their training rows.

The helper file uses only the Python standard library. It does not import
pandas or scikit-learn because the notebooks keep the actual cleaning,
imputation, fitting, and evaluation steps visible for teaching.

---

## 9. What to remember for exams and practical work

Ask these questions in order:

1. What does each column mean, and what is its role?
2. Which values are impossible according to the column's domain?
3. Is this an invalid value, a missing value, an outlier, or a valid rare case?
4. Could repeated entities cross the train/test boundary?
5. Which treatment preserves the intended population?
6. Were all alternatives tested on the same held-out rows?
7. Was every learned statistic fitted on training rows only?
8. Does the metric match the task and have understandable units?
9. Is the effect larger than its uncertainty?
10. Can another notebook rebuild the exact recipe from the log?

Do not memorise the exact winning treatment as a universal rule:

- median/mode worked for classification;
- KNN/explicit missing level worked for regression;
- `log1p` helped different columns in the two datasets;
- dropping invalid regression rows was a domain-validity decision, not simply
  the highest-scoring option.

## One-sentence memory aid

> **Understand the columns, define impossible values from domain knowledge,
> preserve valid observations, compare treatments on identical held-out rows,
> and carry forward rules rather than leaked fitted values.**
