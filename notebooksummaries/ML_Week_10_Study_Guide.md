# Week 10 — Bagging and Random Forests

Sources: [bagging notebook](../notebooks/week_10/week_10_ensembles_bagging.ipynb) · [regression version](../notebooks/week_10/week_10_ensembles_bagging_regression.ipynb) · plus notes from the lecture

## The big idea

One deep decision tree is **unstable**: change a few training rows and you get a very different tree. **Bagging** turns that weakness into a strength:

1. Make many **different training sets** by resampling the rows.
2. Train **one copy of the same algorithm on each**, in parallel.
3. **Combine** their answers: vote for classification, average for regression.

Each tree makes different mistakes, and averaging cancels a lot of them out. In the professor's words: **any algorithm, multiple parallel copies, bootstrap samples, decision by vote/average.**

## Why resample? (bootstrapping)

If every tree saw the exact same data, they'd all be the same tree and averaging would do nothing. So each member gets a **bootstrap sample**: draw n rows **with replacement** from the n training rows.
- Some rows get picked 2–3 times, and others **not at all**.
- On average about **36.8% of rows are left out** of each member (the math: (1 − 1/n)^n → 1/e ≈ 0.368). The notebook measured ≈ 36.5%.
- That's what makes the trees **different from each other**.

**Bagging can't fix a mistake every member makes.** If all trees get a row wrong, the vote is still wrong. Diversity helps only when the errors differ.

## Bagging helps unstable models, not stable ones

| Model | Stable? | Gain from bagging |
|---|---|---|
| Deep decision tree | **Unstable**: small data changes → very different tree | **Big** (F1 0.80 → 0.84) |
| Pruned tree | Less unstable | Small |
| Logistic regression | Stable | ~Nothing (and very slow) |
| KNN | Stable | ~Nothing |
| Naive Bayes | Stable | ~Nothing (slightly worse) |

The notebook measured how much the bagged members **disagreed**: the trees disagreed the most and gained the most. Stable models make nearly the same fit on every bootstrap sample, so there's nothing to average away.

**Rule: bagging is for high-variance, unstable models.**

## The default result

`BaggingClassifier()` = 10 unpruned trees. **Training F1 ≈ 0.99, validation F1 ≈ 0.84.** The big gap is normal: each deep tree memorises its own sample. What matters is that validation went up compared to a single tree.

(In scikit-learn the classifier actually **averages the members' probabilities** and picks the larger class, a "soft" vote.)

## One tool, four sampling methods

`BaggingClassifier` can do more than bagging, depending on how **rows** and **columns** are sampled:

| Method | Rows | Columns | Is it bagging? |
|---|---|---|---|
| **Bagging** | Random **with** replacement (`bootstrap=True`) | All | ✅ Yes |
| **Pasting** | Random subset **without** replacement (`bootstrap=False`, `max_samples<1`) | All | ❌ |
| **Random subspaces** | All rows | Random subset of columns per member | ❌ |
| **Random patches** | Random subset | Random subset | ❌ |

**`bootstrap=True` → bagging. `bootstrap=False` → not bagging.** Careful: with `bootstrap=False` *and* all rows, every member sees the same data, so you get near-identical trees and no improvement (F1 fell back to ~0.80).

## Out-of-bag (OOB) score: free validation

Since each member skipped ~37% of rows, we can test each row using **only the members that never saw it**. That's the **OOB score**: an estimate of performance **without holding out any data**, almost like built-in cross-validation.
- In the notebook, OOB error and validation scores were **pretty similar** in level, though they didn't always pick the same best setting.
- OOB only exists when `bootstrap=True` (otherwise no rows are left out).
- By default `oob_score_` is **accuracy**, not F1.

## Bagging parameters

Only `n_estimators` (and `estimator`) are about the *model*. The others are about **the dataset: which rows and columns go to which tree.**

| Parameter | Meaning | Default |
|---|---|---|
| `estimator` | The base algorithm each member uses | Decision tree |
| `n_estimators` | How many members | 10 |
| `max_samples` | How many rows each member draws | 1.0 |
| `max_features` | How many columns each member gets (**chosen once per member**) | 1.0 |
| `bootstrap` | Rows drawn with replacement? | True |
| `bootstrap_features` | Columns drawn with replacement? | False |

### n_estimators: how many trees

Tried every value from 1 to 150. The score **rises fast, then flattens** (≈ 0.79 with 1 tree, 0.84 with 10, 0.86 with 50, and after that almost flat). The best was **139** (0.864).

**Professor's rule: when scores are basically tied, pick the smallest number.** More models = more cost, and beyond the plateau they add nothing real. Sometimes more members is even slightly worse.

(Fun fact: 2 trees scored worst. A 1–1 split averages to 0.5 and the tie goes to the negative class.)

### max_samples: how many rows per tree

**Float vs integer matters!**
- `1.0` (float) = **100% of the rows** (the default for bagging).
- `0.7` = 70% of the rows.
- `1` (integer) = **literally one row**. Don't confuse them.
- In `RandomForestClassifier` the default is `None`, which also means "as many as the training set".

Best was **0.7**, not the full size. Smaller samples → more different trees, but each tree learns from less. **Bigger isn't always better.**

### max_features: how many columns per tree

Each member gets a random subset of columns and **uses that same subset for all its splits**. Best was again **0.7**. Fewer columns → more diverse trees but each one is weaker.

## Random Forest = bagging + random columns at every split

A **random forest** is a special case of bagging:
- members are **decision trees**,
- rows are **bootstrapped** (like bagging),
- **plus:** at **every split (node)**, the tree may only choose from a **fresh random subset of columns**. (That's the "features at each node" part.)

**The key difference:**
- **Bagging's `max_features`:** pick the columns **once per tree**.
- **Forest's `max_features`:** pick new candidate columns **at every split** (default `sqrt` of the number of columns for classification).

Why it helps: in plain bagging, if one feature is very strong, **every tree puts it at the top** and the trees look alike. Forcing random columns per split makes the trees **more different (less correlated)**, so averaging works better. This is also why a forest still works with `bootstrap=False`: the per-split randomness alone keeps the trees different.

**Extra Trees (Extremely Randomized Trees):** a random forest where the **thresholds are random too** (like `splitter='random'`), and by default no bootstrap. Basically "a forest of random trees". Even more diversity, and faster.

### Forest results

- The default forest (100 trees) got validation F1 ≈ **0.867**, better than bagging.
- The best `n_estimators` was **127**, but the curve **doesn't really improve above ~60 trees**.
- Training F1 = **1.0**: individual trees overfit by default (unlimited depth), and the averaging is what saves it.
- Other forest sweeps: limiting depth mostly hurt, `min_samples_leaf=2` was slightly best, and `sqrt` columns won by a small margin. **All the differences were small.**

## How to choose a model (professor's advice)

**Don't pick the model just because it's top of the leaderboard (the "Kaggle" mindset).** The best CV score here was at the full length (127+ trees), but the professor would choose a much smaller forest (around where the curve flattens), because it's **almost as good, cheaper, and more reliable**.

**For the project:** prefer the **reliable model that doesn't overfit**. Choose the absolute best CV model only if you can **justify** it.

## From one-parameter sweeps to a joint search

Since week 6 we've gone parameter by parameter, but that was **to learn what each one does**. In practice parameters **interact**, so you should search them **jointly**.

### The problem with grid search

Grid search tests **every combination**. Here the combined space had **285,000 combinations**, way too many to measure.

### Random search with 60 trials

Instead, **sample 60 random combinations**. Why 60 is enough:
- Each random trial has a 5% chance of landing in the **top 5%** of all combinations.
- The chance that **none** of 60 trials does is 0.95^60 ≈ 5%.
- So with 60 trials you have **~95% chance of getting at least one top-5% combination**, **no matter how big the space is**.

**From now on, every class uses random search with 60 trials, inside nested CV.**

### The final result

Nested CV on the random search gave **F1 ≈ 0.866**. The model kept (refitted on all training data) is a **Random Forest with `bootstrap=True`**: 135 trees, log-loss criterion, max depth 10, min 2 rows per leaf, `sqrt` columns per split. Test F1 ≈ 0.88. **The best model on the course board so far.**

## Edge case (in the notebook, not really lectured)

**A bagged ensemble of trees and a random forest can be exactly the same model.** If the forest is allowed to look at **all columns at every split** (`max_features=1.0`) and both use the **same bootstrap samples**, they build identical trees. The notebook checked every split and threshold: all equal. Change one thing (for example, the forest looks at only 1 column per split) and they differ. So **"random forest = bagging + per-split column sampling"**; remove that sampling and it's just bagging.

## Check yourself

1. Why do we give each tree a different bootstrap sample?
2. Why does bagging help a deep tree but not logistic regression or KNN?
3. About what share of rows does each bootstrap member leave out? What can we do with them?
4. What's the difference between bagging and pasting?
5. What's the difference between `max_features` in bagging and in a random forest?
6. If the best CV score is at 139 trees but 60 trees is almost the same, which do you pick, and why?
7. Why does random search with 60 trials work, however big the search space is?
8. Why does averaging more trees score higher on athletes they weren't trained on?
