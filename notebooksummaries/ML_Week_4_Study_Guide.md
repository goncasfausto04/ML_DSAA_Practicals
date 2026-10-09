# Machine Learning — Week 4: Feature Engineering, Encoding, Reduction, and Selection

## Glossary

| Key term | Meaning |
|---|---|
| Feature engineering | Creating or transforming model inputs |
| One-hot encoding | Representing each category with a separate 0/1 column |
| Ordinal encoding | Mapping ordered categories to numeric values |
| Count/frequency encoding | Replacing a category with its training frequency |
| Target encoding | Replacing a category with a target-related training statistic |
| Feature scaling | Changing numeric feature magnitudes to a common scale |
| Feature selection | Choosing which existing features to retain |
| Dimensionality reduction | Replacing features with fewer derived components |
| PCA | Principal component analysis, a variance-preserving reduction method |
| Filter method | Selecting features using feature-level scores without repeatedly fitting a model |
| Wrapper method | Selecting features by repeatedly fitting and evaluating a model |
| Embedded method | Selecting features as part of model fitting |
| Leakage | Letting held-out or target information influence learned preprocessing or selection |
| Logistic regression | A linear classification model that estimates class probabilities |
| Ridge regression | Linear regression with an L2 penalty |
| Lasso | Linear regression with an L1 penalty that can set coefficients to zero |
| F1 | The harmonic mean of precision and recall |
| MAE | Mean absolute regression error |
| SEM | Standard error of the mean |

# Week 4 — Feature Engineering, Encoding, Reduction, and Selection

## The takeaways

1. **Models need numeric representations.** Choose encodings that respect categorical meaning.
2. **Feature engineering creates or transforms inputs; feature selection keeps or drops inputs; dimensionality reduction creates new components.** Do not confuse them.
3. **Variance is not predictive value.** PCA can preserve most variation and still damage accuracy/F1.
4. **Feature selectors answer different questions:** filters score features, wrappers repeatedly fit models, embedded methods select while fitting.
5. **Fit every learned operation inside the training fold.** Target-aware encoding and feature selection are especially prone to leakage.
6. **Smaller is not automatically better; more complex is not automatically better.** Judge changes using repeated paired held-out performance, uncertainty, and interpretability.

## Encoding: turning categories into numbers

| Encoding | How it works | Risk or advantage |
|---|---|---|
| One-hot | One 0/1 column per category | Avoids inventing order; can make wide matrices |
| Ordinal | Assign numbers to categories | Appropriate for real order; otherwise may invent order |
| Count/frequency | Replace category with how often it occurs in training | Compact, but loses category identity |
| Target | Replace category with target-related training statistic | Can be powerful; high leakage/overfitting risk without careful training-only or out-of-fold handling |

The course classification encoding produces roughly **48–49 columns**, depending on whether rare categories appear in a training split. That is expected: learned encoders depend on the training sample. Handle unseen categories consistently at prediction time.

## Feature scaling

**Scaling** changes numeric feature magnitudes (e.g. standardization to mean 0 and standard deviation 1). It matters when an algorithm compares distances or penalizes coefficient sizes. It is generally unnecessary for a basic decision tree. The course keeps one-hot 0/1 indicators as-is in the compared setup, but scaling policy depends on the model and representation.

## Feature engineering versus selection

- **Transformation:** `log1p(mileage)` changes the scale of an existing feature.
- **Combination/engineering:** `km_per_year = mileage / (2026 - year)` creates a new usage-intensity feature from two inputs (the notebook's 2026 reference year is course-specific).
- **Selection:** Keep only certain original or encoded columns.
- **Reduction:** Replace columns with derived components, often mixtures of several inputs.

**Order matters:** Create candidate features **before** asking a selector to choose among them.

In the regression notebook, unlogged `km_per_year` helped; logging it worsened MAE by about **€44** and made it highly redundant with logged mileage (correlation about **0.9943**). **Lesson:** a transform that helps one variable can hurt another.

## Dimensionality reduction

- **PCA:** Finds directions with high variance after centering the feature matrix; new components are combinations of original columns.
- **TruncatedSVD:** Similar component-based compression without centering; useful for sparse matrices.
- **Explained variance:** How much of the original input variation is represented by the components. **Not** how much predictive signal is preserved.

Course results: 80%-variance PCA reduced classification F1 from about **0.8523 to 0.8191**, and increased regression MAE from about **€2,996 to €5,337**. Do not use PCA simply because it retains “most of the information”: the retained variance may be irrelevant to the target.

## Three feature-selection families

| Family | Basic mechanism | Strength | Weakness | Course methods |
|---|---|---|---|---|
| Filter | Score features without fitting the final predictor | Fast, simple | Often misses interactions | VarianceThreshold, correlations, chi-square/ANOVA-based relevance |
| Wrapper | Fit models repeatedly on candidate subsets | Can capture joint usefulness | Computationally expensive; needs careful validation | RFE, RFECV, sequential selection |
| Embedded | Select as part of fitting | Integrates selection and model | Depends on model and penalty/importance rule | L1/Lasso, tree importance |

### Filters: relevance versus redundancy

- **Relevance:** Is this feature related to the target?
- **Redundancy:** Does this feature mostly repeat another feature?
- **Variance:** Does the feature vary at all? A varying feature is not necessarily useful.
- **Univariate limitation:** A feature with little standalone association might be useful in combination with another.

Know which variable types a statistic handles: Pearson for numeric–numeric linear association; point-biserial for binary–numeric; chi-square/Cramér's V for categorical association; ANOVA/correlation ratio for categorical groups versus numeric outcomes. **A p-value is not an effect size.**

### Wrappers and embedded methods

- **RFE (recursive feature elimination):** Fit a model, remove least-important feature(s), repeat until the desired count remains.
- **RFECV:** Uses internal cross-validation to choose a feature count.
- **Sequential selection:** Adds or removes features while repeatedly checking model performance.
- **L1/Lasso:** A penalty can shrink some coefficients exactly to zero; `alpha` or `C` controls penalty strength depending on estimator (for logistic regression, smaller `C` means stronger regularization).
- **Tree importance:** Provides a relative importance score, but the cutoff for selecting features is a separate decision.

**Caution:** Choosing a feature count or selector on validation results and then reporting the winning validation score as an unbiased final result is optimistic. Use nested validation or a separate final test when necessary.

## Combining selectors: funnel versus vote

- **Sequence/funnel:** Run selectors one after another; early exclusions are permanent. Faster, but a mistake early on cannot be recovered.
- **Vote:** Run multiple selectors on the full training representation; retain a feature if at least two vote to keep it. More expensive, but no single selector has unilateral veto power.

The course compared a filter, wrapper, and embedded method:

- **Classification:** vote retained about **30 columns**, F1 **0.8525 vs 0.8523** for all features — effectively tied within uncertainty. It mainly reduced width.
- **Regression:** vote retained about **21.7 columns**, MAE around **€2,992 vs €2,996** for all features — a small improvement that cleared the notebook's uncertainty rule.
- Sequence performed worse in both comparisons.

**Do not memorize “vote always wins.”** The important reasoning is that selectors have different blind spots, and any combination must be evaluated.

## How to interpret repeated results

Both Week 4 notebooks use **20 repeated 80/20 train/held-out splits**, evaluating alternatives on matching splits. Report the **paired mean difference** and **SEM**:

- F1: positive difference = improvement.
- MAE: negative difference = improvement.
- A difference around or below roughly **two SEMs** from zero is treated as insufficiently clear evidence in these notebooks. This is a practical rule of thumb, not a universal statistical guarantee.

Even if a difference is statistically distinguishable, ask if it matters operationally (e.g. €4 lower MAE may be commercially negligible).

## What you should know about the Week 4 models

- **Logistic regression:** Despite its name, used here for **classification**; estimates class probability from a linear combination of features and a logistic link.
- **Ridge regression:** Linear **regression** with an L2 penalty that discourages large coefficients; scaling can affect the penalty's behavior.
- **Lasso:** Linear model with an L1 penalty that can force some coefficients to zero, enabling embedded feature selection.

### Check yourself

1. Why is one-hot encoding often safer than arbitrary numeric labels for nominal categories?
2. Why can PCA retain 80% of variance but lose predictive performance?
3. What is the difference between engineering, selection, and reduction?
4. What makes target encoding especially vulnerable to leakage?
5. Why might RFE find useful combinations that a simple filter misses?
6. Why can a smaller feature set be preferable even if its F1 is essentially unchanged?
7. Why is a tiny MAE improvement not automatically meaningful?

**Week 4 in one sentence:** Engineer useful features, fit every learned transformation inside the split, distinguish selection from reduction, and keep only changes supported by held-out evidence.

---

