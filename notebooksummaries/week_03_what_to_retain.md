# Week 03: What to Retain

This guide covers both solution notebooks:

- **Week 3:** athlete outcome classification (`Outcome`).
- **Week 3b:** used-car price regression (`price`).

The datasets differ, but the transferable lesson is the same:

```text
inspect -> define valid values -> compare alternatives -> choose -> apply -> log
```

## 1. Explore before cleaning

Before changing data, check:

- shape, data types, unique values, missing values, and duplicates;
- distributions, quantiles, and suspicious values;
- the target distribution;
- each variable's role: numeric, boolean, nominal categorical, ordinal, or identifier.

Identifiers such as `RecordID`, `CarID`, and `Athlete Id` identify rows or people. Do not use them as model features.

## 2. Use domain rules, not only statistics

Define what is physically or logically possible for each column. For example:

- percentages stay within their scale;
- prices, distances, and counts cannot be negative;
- strictly positive measurements cannot be zero;
- years must be plausible whole numbers;
- categories and booleans need valid spellings.

An impossible value is not automatically the same as a missing value. Decide whether to repair it, replace it with `NaN`, or remove the row, and record the reason. An unusual value may still be real: a boxplot's 1.5 x IQR rule is a screening tool, not an automatic deletion rule.

## 3. Compare decisions fairly

Cleaning and preprocessing choices are part of the model. Compare alternatives such as:

- drop, repair, or blank invalid values;
- mean, median, mode, KNN, or iterative imputation;
- keep or remove outliers;
- transform or do not transform a feature;
- different scalers or encodings.

Use the **same held-out rows** for competing alternatives. Repeated paired splits are better than one lucky split. Report the average and uncertainty; a tiny score difference may not be meaningful.

## 4. Avoid leakage

Split before learning preprocessing values. Fit on training data, then apply to both training and test data:

```python
train, test = train_test_split(data, random_state=42)
imputer.fit(train_features)
train_features = imputer.transform(train_features)
test_features = imputer.transform(test_features)
```

This rule applies to imputers, encoders, scalers, transforms, feature selection, and any statistic learned from data. Using the full dataset makes the test score too optimistic.

## 5. Missing values and transformations

Missingness is a modelling decision, not automatically a reason to delete a row. Ask whether missingness carries information and evaluate different fills on held-out data. A fill value must be computed from the training portion only.

`log1p(x)` can reduce a long right tail, but skewness alone does not justify it. Test each candidate column with and without the transform:

> Apply a transform only when it improves the relevant held-out metric with meaningful evidence.

## 6. Encoding, scaling, and metrics

- One-hot encoding is a safe default for nominal categories. Ordinal encoding is appropriate only when the order and spacing are meaningful.
- Measure category cardinality: a high-cardinality column can create a very large design matrix.
- Scaling often helps optimization-based models such as logistic regression when feature magnitudes differ.
- Ordinary least-squares predictions are theoretically scale-invariant for a full-rank design, although scaling can still help numerical conditioning or other model types.

For classification, F1 balances precision and recall:

$$F1 = 2 \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

For regression:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

MAE is easy to interpret in the target's units. RMSE penalizes large errors more strongly.

## 7. What each notebook selected

These are examples from these files, not universal rules:

| Decision          | Week 3: classification                   | Week 3b: regression                                   |
| ----------------- | ---------------------------------------- | ----------------------------------------------------- |
| Invalid values    | Blank invalid cells                      | Compare repair, drop, and blank; select from evidence |
| Repeated entities | Keep each athlete's earliest `RecordID`  | Not applicable                                        |
| Missing values    | Left open for later weeks                | Training median plus a missingness-level strategy     |
| Outliers          | Keep all observations                    | Keep all observations; flagged hybrids were valid     |
| Transform         | `log1p` for training-volume variables    | `log1p` for `mpg`, not every skewed column            |
| Scaling           | Standard scaling for logistic regression | No scaler for ordinary least squares                  |
| Main metric       | F1                                       | MAE, with RMSE as support                             |

## 8. Reproducibility checklist

Before trusting a model, ask:

- Did I understand every feature and the target?
- Did I remove identifiers from the features?
- Did I define impossible values from domain knowledge?
- Did I inspect distributions instead of blindly deleting boxplot points?
- Did I compare alternatives on identical held-out data?
- Was every preprocessing value fitted on training data only?
- Is the metric appropriate for the task?
- Did I record the action, reason, affected rows, and parameters?

**Main takeaway:** explore first, define the problem in domain terms, compare alternatives without leakage, and keep the recipe supported by evidence.
