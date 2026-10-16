# Machine Learning — Week 3: Exploration, Data Quality, and Cleaning

Sources: [Classification solution](../solutions/week_03/week_03_deepen_exploration_solution.ipynb) · [Regression solution](../solutions/week_03/week_03_deepen_exploration_regression_solution.ipynb).

## Key learning points

- Understand each column's meaning: identifier, measurement, count, nominal category, ordered category, or boolean. Storage dtype alone is insufficient.
- Separate **missing**, **invalid**, **unusually extreme**, and **repeated** observations. They require different decisions.
- Use domain knowledge before automatic cleaning rules. A rare or extreme value can be valid.
- Compare cleaning strategies on the same held-out splits, fitting learned values only on training rows.
- Preserve a reproducible cleaning **recipe**, then refit its learned values in every training fold.

## Outliers and transformations

- `IQR = Q3 - Q1`; boxplots flag values outside `Q1 - 1.5 × IQR` and `Q3 + 1.5 × IQR`.
- These are investigation flags, not automatic deletion rules. If `IQR = 0`, even valid nonzero values may be flagged.
- `log1p(x) = ln(1 + x)` compresses non-negative, right-skewed features and handles zero. Validate whether it helps prediction.
- Repeated entities can leak across splits even when their rows are not identical; `drop_duplicates()` alone may miss this.

## Missing values

| Strategy | Key idea or limitation |
|---|---|
| Median / mode | Simple numeric / categorical filling using training statistics |
| Explicit missing category | Retains the fact that a category was not recorded |
| KNN imputation | Uses similar training cases; sensitive to distance and scaling |
| Iterative imputation | Predicts missing values from other columns; complexity does not guarantee improvement |

**Fit on training, transform held-out data.** Never carry a median learned from the full dataset into evaluation folds.

## Classification notebook: athletes

- Repair category typos and interpret placeholder or administrative codes using domain meaning.
- Keep one record per athlete to prevent repeated-athlete leakage; the earliest `RecordID` is the chosen reproducible tie-break.
- Retain valid high-training athletes rather than deleting every IQR outlier.
- The selected recipe uses training-fold median/mode imputation and `log1p` on training-volume features.
- **Lesson:** straightforward cleaning can beat unnecessary complexity while preserving valid observations.

## Regression notebook: car prices

- Check plausibility and category spelling; remove records judged untrustworthy by domain rules.
- Drop the high-cardinality `model` feature while keeping `Brand`.
- The selected recipe uses numeric KNN imputation (`k=5`), a missing category, and `log1p` on `mpg` and `mileage`.
- Keep the target in euros for euro MAE. Training or evaluating in log-price space changes what errors mean.
- **Lesson:** different datasets can justify different cleaning recipes; a small score gain need not outweigh invalid data.

## Comparing cleaning choices

Both notebooks use **20 repeated 80/20 splits**, comparing candidates on matching splits. **MAE** is average absolute error; **RMSE** penalizes large errors more strongly. Lower is better for both.

Compare the mean paired score difference with its **SEM** (standard error of the mean). Small changes relative to uncertainty give weak evidence. Reusing these scores to choose a strategy means they are selection evidence, rather than an untouched final test.

`CleaningLog` records the actions and reasons; the notebook performs the cleaning. Save “use a training median”, rather than a fixed full-dataset median.

## Check yourself

1. How does an invalid value differ from a valid outlier?
2. How can repeated entities cause leakage?
3. Why must imputation be refitted inside each training fold?
4. Why might athletes and cars need different cleaning strategies?
