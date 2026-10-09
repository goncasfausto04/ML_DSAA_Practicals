# Machine Learning — Week 3: Exploration, Data Quality, and Cleaning

## Glossary

| Key term | Meaning |
|---|---|
| Identifier | A value that identifies an entity, usually not predictive information |
| Continuous numeric | A measurement where numerical differences have meaning |
| Count | A non-negative integer quantity |
| Nominal categorical | A category with no natural ranking |
| Ordinal categorical | A category with a meaningful order |
| Boolean | A variable with two valid states |
| Missing value | A value that was not recorded |
| Invalid value | A value that violates domain rules |
| Outlier | An unusually extreme value that may still be valid |
| Duplicate / repeated entity | The same real-world entity appearing more than once |
| IQR | Interquartile range, `Q3 - Q1` |
| Imputation | Filling or estimating missing values |
| Median imputation | Replacing missing numeric values with a training median |
| KNN imputation | Estimating missing values from similar observations |
| Data leakage | Allowing information from held-out data to influence preprocessing or fitting |
| Cleaning recipe | A reproducible description of cleaning actions, not fixed values learned from all data |
| MAE | Mean absolute regression error |
| RMSE | Root mean squared error, which penalizes large errors more strongly |
| SEM | Standard error of the mean |

# Week 3 — Exploration, Data Quality, and Cleaning

## The takeaways

1. **Understand every column before modifying it.** A numeric dtype does not tell you whether a variable is a measurement, identifier, count, or encoded category.
2. **Distinguish invalid values, missing values, outliers, and valid rare cases.** They require different decisions.
3. **Domain rules outrank blind statistical rules.** A boxplot flag does not prove a value is wrong.
4. **Repeated entities can leak information.** Identical rows are not the only duplicates to investigate.
5. **Cleaning is a model choice.** Compare alternatives on the same held-out splits; don't assume complicated methods win.
6. **Carry forward recipes, not fitted values.** An imputation strategy can be reused; a median learned from the full dataset cannot safely be reused in evaluation folds.

## Know the variable types

| Type | What it means | Example |
|---|---|---|
| Identifier | Identifies an entity, usually not predictive information | `Athlete Id`, `CarID` |
| Continuous numeric | Measurement with meaningful numerical differences | `mileage`, `price` |
| Count | Non-negative integer quantity | Previous attempts |
| Nominal categorical | Categories with no natural ranking | `Brand`, `fuelType` |
| Ordinal categorical | Categories with meaningful order | Education level, income band |
| Boolean | Two valid states | TRUE/FALSE |

A column like `Edition` can be stored as numbers yet function as a category. A string like `"FALSE"` can represent a boolean. **Meaning comes from the domain, not the storage type.**

## Four problems you must distinguish

- **Missing value:** No recorded value (`NaN`). Decide whether to fill it, represent missingness explicitly, or drop the affected observation.
- **Invalid value:** Violates domain rules (e.g. negative mileage or an impossible age code). It may be repaired, blanked, flagged, or removed after justification.
- **Outlier:** Unusually far from typical values, but may be perfectly valid (e.g. an expensive car or highly trained athlete).
- **Duplicate/repeated entity:** The same real entity appears more than once, even if the rows differ. Can make train/validation performance misleading.

### IQR and skew

- `IQR = Q3 - Q1`.
- Standard boxplot flags values below `Q1 - 1.5 × IQR` or above `Q3 + 1.5 × IQR`.
- This is a **screening heuristic**, not an instruction to delete observations.
- If `Q1 = Q3 = 0`, then `IQR = 0`: every nonzero value may be flagged, even if real.
- `log1p(x) = ln(1 + x)` compresses a non-negative, right-skewed feature and works at zero. It does not automatically improve a model.

### Imputation concepts

| Method | Idea | Main tradeoff |
|---|---|---|
| Median | Replace missing numeric values with training median | Simple, robust; ignores relationships |
| Mode | Replace with most common training category | Simple; can hide meaningful missingness |
| Explicit missing category | Treat missing categorical value as its own level | Preserves absence information |
| KNN imputation | Estimate missing numeric values using similar training observations | Uses relationships; distance/scaling and computation matter |
| Iterative imputation | Estimate missing values using other columns iteratively | More complex; not guaranteed better |

**Fit imputers on training rows only.** Apply the learned rule to validation rows without refitting. If selecting an imputation method using repeated validation scores, those scores are selection evidence, not an untouched final performance estimate.

## What the classification practical found

The raw competition data had **4,280 rows**, producing **4,000 cleaned rows**, with `Outcome` as target.

- Repair typo `FASE` → `FALSE`.
- Treat `Age group = 0` as a placeholder and negative athlete score as an administrative code rather than a real measurement.
- Keep the earliest `RecordID` per athlete to avoid repeated-athlete leakage; the tie-break is a reproducibility rule, **not** evidence that earliest is intrinsically most predictive.
- Keep missing values unresolved in the exported clean file; use training-fold **median/mode** filling later.
- Use `log1p` for 11 training-volume columns. The course reports F1 improvement from approximately **0.8182 to 0.8506** across its paired evaluation.
- Do not blindly drop IQR outliers: that discarded valid high-training athletes and changed class balance.

**What to learn:** A simpler cleaning method can be preferable when more complex ones do not measurably improve held-out F1. Preserving valid rows matters.

## What the regression practical found

The raw car data had **4,600 rows**, producing **4,000 cleaned rows**, with `price` in euros as target.

- Check category spellings and domain validity of `year`, `mileage`, `tax`, `mpg`, `engineSize`, ownership counts, and price.
- Drop records deemed untrustworthy by domain rules, even though repairing some malformed values gave slightly better prediction scores. **Domain validity can outweigh a small metric gain.**
- Drop the very high-cardinality `model` feature; keep `Brand` as a more manageable representation.
- Use training-fold **KNN imputation (`k=5`)** for numeric fields and an explicit missing category for categorical fields.
- Use `log1p` on `mpg` and `mileage` to compress long tails.
- The target `price` is skewed, but the notebook keeps it in euros because the comparison uses euro MAE.
- For ordinary full-rank least squares, feature rescaling changes coefficient units without theoretically changing predictions. Scaling matters much more for distance-based, penalized, and gradient-based methods.

**What to learn:** Classification and regression can legitimately choose different imputers. The choice is empirical and domain-dependent.

## Regression metrics

- **MAE** = average of `|actual - predicted|`. Lower is better. If MAE = €3,000, predictions miss by €3,000 on average in absolute value.
- **RMSE** = square root of average squared errors. Lower is better; penalizes large misses more strongly.
- **Target transformation:** predicting `log(price)` changes the training objective and the meaning of errors. Predictions can be converted back to euros, but scores computed in log space are not directly euro MAE.

## Why repeated paired evaluation matters

Week 3 uses **20 repeated 80/20 splits**. Every candidate treatment is compared on the **same splits**, with preprocessing fitted only on training rows. Compare the difference per split, then its mean and **SEM** (standard error of the mean). A tiny average difference relative to its uncertainty is weak evidence. Repeated splits reduce dependence on one lucky partition, but they do **not** automatically produce an unbiased final test after choosing the winner.

## What `course_helpers.py` is for

`CleaningStep` records a column, action, reason, affected-row count, and optional machine-readable `carries`. `CleaningLog` records and saves an ordered recipe to JSON and loads it later. **The helper does not clean or train data.** The notebook performs those operations. Remember: store *“impute with a training median”*, not *“use this median computed on all rows.”*

### Check yourself

1. Why might `drop_duplicates()` fail to detect leakage between train and validation?
2. Is a car with extremely high mileage necessarily an invalid observation?
3. Why is `IQR = 0` dangerous for automatic outlier deletion?
4. Why did the course choose different missing-value treatments for athletes and cars?
5. What does it mean to fit an imputer **inside** each training fold?
6. What is the difference between a cleaning *recipe* and a learned cleaning *value*?

**Week 3 in one sentence:** Define valid data from its meaning, preserve legitimate observations, compare cleaning choices fairly, and store reproducible rules without leakage.

---

