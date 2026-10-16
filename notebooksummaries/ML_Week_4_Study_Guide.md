# Machine Learning — Week 4: Feature Engineering and Selection

Sources: [Classification solution](../solutions/week_04/week_04_feature_work_classification_solution.ipynb) · [Regression solution](../solutions/week_04/week_04_feature_work_regression_solution.ipynb).

## Key learning points

- **Engineering** creates or transforms features; **selection** keeps existing columns; **reduction** replaces them with fewer derived components.
- Fit encoders, scalers, dimensionality reduction, and selectors inside each training fold.
- More features or more complicated transformations do not automatically improve prediction.
- Judge changes by held-out performance, uncertainty, interpretability, and computational cost.

## Encoding and scaling

| Method | What to remember |
|---|---|
| One-hot encoding | One indicator per category; avoids inventing order but can create many columns |
| Ordinal encoding | Numeric ranks; appropriate when categories have a real order |
| Count/frequency encoding | Uses training category frequencies; compact but loses category identity |
| Target encoding | Uses target statistics; especially vulnerable to leakage and needs careful out-of-fold handling |
| Scaling | Puts numeric features on comparable scales; important for distances and coefficient penalties |

Handle unseen categories consistently. A basic decision tree generally does not need scaling.

## Engineering versus reduction

Create candidate features **before** selection. The regression notebook adds usage intensity (`km_per_year`); logging this feature worsened MAE and made it nearly redundant with logged mileage. **A transform that helps one feature may hurt another.**

**PCA** creates components that preserve input variance, not necessarily predictive signal. **TruncatedSVD** supports compression without centering and is useful for sparse inputs. In both notebooks, retaining 80% of variance with PCA worsened prediction. Explained variance is not predictive value.

## Three selection families

| Family | Mechanism | Examples | Main limitation |
|---|---|---|---|
| Filter | Scores variation or association | VarianceThreshold, correlation, chi-square, ANOVA | Often misses joint usefulness and interactions |
| Wrapper | Repeatedly fits models on subsets | RFE, RFECV, sequential selection | Computationally expensive |
| Embedded | Selects during model fitting | L1/Lasso, tree importance | Depends on the fitted model and selection rule |

- **Relevance:** relation to the target. **Redundancy:** repetition of another feature's information.
- **RFE** repeatedly removes weak features; **RFECV** uses internal CV to choose the count.
- **L1** can shrink coefficients to zero; **L2/Ridge** shrinks coefficients without generally removing them. For logistic regression, smaller `C` means stronger regularization.
- Match association tests to variable types. Statistical significance (a p-value) is not effect size.

## Combining selectors

- **Sequence/funnel:** apply selectors in order; early exclusions cannot be recovered.
- **Vote:** run selectors on the full training representation and retain columns supported by enough selectors.

The notebooks' vote combined filter, wrapper, and embedded methods. Classification became smaller with essentially tied F1; regression saw a small MAE improvement. The sequence performed worse. **Learn the trade-off, not “voting always wins”.**

## Evaluating changes

Both notebooks use **20 paired 80/20 splits**. Positive F1 differences and negative MAE differences indicate improvement. Their rule of roughly two SEMs is a practical uncertainty heuristic, not a universal statistical guarantee. A statistically detectable gain may still be too small to matter in practice.

Selecting features or methods using validation scores makes the winning score optimistic; use an independent test or nested validation for final assessment.

## Check yourself

1. How do engineering, selection, and reduction differ?
2. Why can PCA preserve most variance but worsen predictions?
3. What separates filters, wrappers, and embedded methods?
4. Why can fewer features be useful even when F1 barely changes?
