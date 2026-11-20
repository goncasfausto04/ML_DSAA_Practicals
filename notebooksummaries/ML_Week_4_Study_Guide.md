# Week 4 — Feature Engineering and Feature Selection

Sources: [classification solution](../solutions/week_04/week_04_feature_work_classification_solution.ipynb) · [regression solution](../solutions/week_04/week_04_feature_work_regression_solution.ipynb)

## The big idea

Before modelling, you shape the columns the model will see. There are three different moves:

- **Engineering:** create or transform features (for example, `km_per_year = mileage / age`).
- **Selection:** keep some of the existing columns and drop the rest.
- **Reduction:** replace many columns with a few new combined ones (PCA).

More features, or fancier features, don't automatically help. Every change has to prove itself on held-out data.

## Encoding: turning categories into numbers

Models need numbers, so categories must be converted:

| Method | How it works | When |
|---|---|---|
| **One-hot** | One 0/1 column per category | Categories with no order (colour, brand) |
| **Ordinal** | Categories become ranks 1, 2, 3… | Categories with a real order (small < medium < large) |
| **Frequency** | Replace a category by how often it appears | Compact, but loses which category it was |
| **Target encoding** | Replace a category by the average target for it | Powerful but **leaks easily**; handle with care |

## Scaling

Scaling puts numeric columns on a similar range. It matters for anything based on **distances** (KNN) or **penalties on coefficients** (regularised regression). Without it, a column measured in big units dominates just because of its units. Decision trees don't need scaling.

## PCA: keeps variance, not signal

PCA builds new columns that keep as much **spread (variance)** of the data as possible. But it **never looks at the target**, so the spread it keeps may not be what predicts the outcome. In both notebooks, keeping 80% of the variance made predictions **worse**. Lesson: "explains most of the variance" ≠ "useful for prediction".

## Three families of feature selection

| Family | How it works | Weakness |
|---|---|---|
| **Filter** | Score each feature alone (correlation, variance, statistical test) | Looks at one feature at a time, so it misses features that only help together, and can't see when two features repeat each other |
| **Wrapper** | Try subsets by actually training models (RFE removes the weakest feature repeatedly) | Slow, many fits |
| **Embedded** | Selection happens during training (Lasso/L1 sets weak coefficients to zero; tree importances) | Depends on that one model |

Two separate ideas:
- **Relevance:** is the feature related to the target?
- **Redundancy:** does it just repeat information another feature already has?

## Combining selectors: voting

Instead of trusting one method, run several and **keep the features that most of them agree on**. This depends less on the quirks of any single method. A "funnel" (one selector after another) is riskier: if an early step drops a feature by mistake, you can't get it back.

## Fit everything inside the training split

Selection looks at the target. If you select features on the full data and then test, the test rows already influenced which features were chosen, and that's **leakage**. So, like cleaning, what you save is the **strategy** ("vote of these three selectors"), and it gets re-run inside each training split.

## Check yourself

1. What's the difference between engineering, selection and reduction?
2. Why can PCA keep 80% of the variance and still hurt predictions?
3. Why might a filter method miss a useful feature?
4. Why save the selection *strategy* and not the list of chosen columns?
