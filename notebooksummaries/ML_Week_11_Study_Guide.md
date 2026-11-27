# Week 11 — Boosting and Combining Models

Sources: [boosting](../solutions/week_11/week_11_ensembles_boosting_solution.ipynb) · [combining models](../solutions/week_11/week_11_combining_models_solution.ipynb) · lecture notes

## Boosting vs bagging

| | Bagging / Forest | Boosting |
|---|---|---|
| Training | Parallel, independent copies | **Sequential**: each model fixes the previous one's mistakes |
| Fixes | Variance (unstable models) | Bias (weak models) |
| Best base model | Strong but unstable (deep tree) | **Weak** (stump) |
| Final answer | Plain vote / average | **Weighted** vote / sum |

---

## AdaBoost

**How it works:** every row has a weight (all equal at the start). At each stage:
1. Fit a weak model on the weighted rows.
2. Weighted error **E** = the sum of the weights of the wrong rows.
3. The model's "say": **alpha = learning_rate × log((1 − E) / E)**. Low error → big say.
4. **New weight of each wrong row = old weight × e^alpha**, then re-normalise so the weights sum to 1 (so the correct rows shrink).
5. The next model focuses on the rows that are now heavy.

The final prediction is a weighted vote, with each stage voting by its alpha.

### The 3 parameters

**1. `estimator`.** We gave AdaBoost every previous week's best algorithm (KNN is excluded because it can't take row weights).
- Same pattern as bagging: some models improved, and **the gain was biggest for the weak/unstable ones**.
- **Default stump: the biggest gain** (F1 0.79 → 0.84).
- Random forest: slightly better (it had the best absolute score, 0.861).
- Logistic regression: **worse**. Boosting stopped early (~7 stages on average).
- Pruned tree: not better, but its scores became **more consistent across folds** (the spread dropped).
- Deep tree: no change. It's perfect on training at stage 1, so there's nothing to reweight.
- **Lesson:** AdaBoost works with any algorithm, but it's best with "stupid" (weak) ones.

**2. `n_estimators`:** the maximum number of stages, i.e. how long you let it run (default 50).

**3. `learning_rate`:** the step size. It multiplies alpha, so it shrinks both each stage's vote and how much the wrong rows get boosted. A smaller rate gives more careful steps but **needs more stages**. With only 50 stages, a rate of 0.1 was worse than 1.0 (it hadn't finished learning).

### Stopping conditions

AdaBoost is **iterative**, so like other iterative algorithms (k-means: max iterations + convergence) it stops when:
- it reaches **`n_estimators`** (the iteration cap), or
- a stage is **perfect** (error = 0, nothing left to fix), or
- a stage is **no better than guessing** (error ≥ 0.5), which means no progress is possible.

**Don't mess with test scores:** validation chooses the model. Test scores are only a final check.

---

## Gradient Boosting

Also sequential, but **no row weights**. Each tree predicts **what's still wrong**.

1. Start with a prediction: a **probability of class 1** (in the notebook, from a first simple tree).
2. Compute **pseudo-residuals** = **actual target − predicted probability**, for each row.
3. Fit a **regression tree on those residuals**. Its leaves predict *corrections*, not the target.
4. New prediction = old + learning_rate × tree.
5. Repeat. **The residuals/error should get smaller with every tree.**

Everything is optimised against a **loss function**: log-loss by default (`exponential` loss is the other option and gives back AdaBoost). **In both cases the trees are regression trees.**

**Main difference from AdaBoost:** AdaBoost's trees predict the **target** (on reweighted rows). Gradient boosting's trees predict the **difference between the target and the current prediction**, and that's what gets optimised.

### Parameters (`GradientBoostingClassifier`)

Learning rate, `n_estimators`, loss, plus all the decision-tree parameters (`max_depth` default **3**, leaf sizes…) and `subsample` (use all rows or a fraction per tree).

- `n_estimators`: about 0.84 at few trees, **best at 110 (0.864)**.
- At first train and val move together. **The gap grows with more iterations**, so it **can overfit with a high learning rate and many iterations**.
- Rule of thumb: **many iterations + a small learning rate** gives the best results (it just takes longer). Here 0.03 needed ~290 trees to match 0.1 at ~120.

---

## Histogram Gradient Boosting

Gradient boosting that **first groups each numeric column into bins** and only searches splits between bins. That's **much faster**; everything else is gradient boosting. The only new parameter is **`max_bins`**.

The speed opens up so many combinations (110,310) that grid search is impossible, so we used **random search with 60 trials** (≈ 95% chance of at least one top-5% solution).
- Nested CV: **0.862**.
- **The best model on the board is still the random forest (0.866).**

---

## Combining models (voting, stacking, blending)

In bagging/boosting all members are **the same algorithm**. Here we combine **different algorithms** (the 6 earlier winners + the boosting winner).

(Professor: **Naive Bayes is terrible** here. Notice it has **low precision but high recall**: it says "win" too easily.)

- **Voting:** a fixed rule. **Hard** = majority of the labels; **soft** = average of the probabilities. Optional `weights` let some members count more.
- **Stacking:** a **meta-estimator**. The members' predictions become the **input features of a new model** (the final estimator), which *learns* how to combine them. The training predictions must be **out-of-fold** (made by members that didn't see that row), otherwise it leaks.
- **Blending:** like stacking, but it holds back a slice of the training rows (e.g. 25%) just for the final model. Simpler, but the members train on less data.

**Takeaways from class:**
- **ML is trial and error.** Stacking is nothing more than a model on top of models, so try things.
- **Don't only combine the best models.** Weaker, *different* models can help. For stacking and blending, mixing more **kinds** of models scored higher.
- Sometimes you won't know *why* a combination is best, **and that's OK**.

**Result:** the combiner search (60 random trials, nested CV 0.867) picked a **stack of pruned tree + random forest**, with a pruned tree as the final model. That's barely above the forest alone (0.866), and lower than the forest on the test rows.

## Check yourself

1. In AdaBoost, how are the weights of the wrong rows updated?
2. Why does AdaBoost help a stump a lot but logistic regression not at all?
3. What are AdaBoost's stopping conditions?
4. What does each tree in gradient boosting predict?
5. Why do a high learning rate and many iterations lead to overfitting?
6. What makes histogram boosting faster?
7. Voting vs stacking: which one *learns* how to combine?
