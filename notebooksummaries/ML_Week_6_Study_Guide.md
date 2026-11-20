# Week 6 — Linear and Logistic Regression

Source: [week 6 solution](../solutions/week_06/week_06_linear_logistic_solution.ipynb)

## The big idea

Both models start from the same thing, a **weighted sum** of the features:

> z = b₀ + b₁·x₁ + b₂·x₂ + … (one weight per feature, plus an intercept b₀)

- **Linear regression** stops there: the prediction *is* z. It's used for numbers (price).
- **Logistic regression** squashes z into a value between 0 and 1 and reads it as a **probability**. It's used for categories (win/lose).

**The type of the target decides the model, before you compute any score.**

## Why not use linear regression for a 0/1 target?

A weighted sum has no limits: it can be −0.8 or 1.56. As a probability that means "155% chance of winning", which is nonsense. In the notebook, about 11% of predictions fell outside [0, 1] **with no error or warning**, and the F1 score looked fine. So the score won't warn you; the argument comes from the range of the output, not from the scoreboard.

Matching the model to the target makes the output **meaningful**. It doesn't promise a better score (here both models scored about the same).

## How logistic regression works

It passes z through the **sigmoid**: σ(z) = 1 / (1 + e^(−z)). Whatever z is, the output is between 0 and 1, so it's a probability **by construction**, not by luck.

Turned around, z = log(p / (1−p)), which is the **log-odds**. So logistic regression is still a linear model, just linear in the log-odds instead of in y.

Being between 0 and 1 doesn't mean the probabilities are *accurate*. Whether "0.8" really comes true 8 times in 10 is a separate property, **calibration** (see week 7).

## Reading the coefficients

- **Linear regression:** a coefficient = how much the prediction changes when that feature goes up one unit, **holding the other features fixed**.
- **Logistic regression:** the coefficient is in log-odds, which is hard to say out loud. Take **e^coefficient = odds ratio**. Above 1 the odds of a win go up with the feature; below 1 they go down. Example: an odds ratio of 19 means "about 19 times the odds".
- **Odds aren't probability.** Doubling the odds doesn't double the probability.
- "One unit" depends on how the column was scaled (for example, one IQR, or the full range).
- If two features carry the same information (mileage and km_per_year), the model **splits the effect between them**, so each coefficient alone looks smaller or weird.

### statsmodels: the statistics table

scikit-learn gives only coefficients. **statsmodels** adds, for each coefficient, a **standard error**, a **95% confidence interval** and a **p-value**.
- p-value < 0.05 → the data pin the coefficient away from zero. p-value > 0.05 → not significant, so it's a candidate for removal.
- A small p-value **doesn't mean causation** and doesn't rank importance.
- A **huge standard error** is a red flag. In the notebook, one feature was carried by a single training athlete, so its coefficient couldn't really be estimated (the fit said `converged: False`).

## Regularisation: penalising big weights

A penalty makes every weight "pay" for being large, which pulls weights toward zero and reduces overfitting.

| Penalty | Effect |
|---|---|
| **L2 (Ridge)** | Shrinks all weights toward zero, but none become exactly 0 |
| **L1 (Lasso)** | Shrinks, and sets the weakest weights to **exactly 0**, which acts as feature selection |
| **Elastic Net** | A mix of both |

- In **logistic regression**, `C` is the **inverse** of the strength: **small C = strong penalty**, large C = weak penalty. With a very strong penalty all weights go to 0 and the model predicts the same thing for everyone.
- In **Ridge/Lasso**, `alpha` is the strength directly: big alpha = strong penalty. Expect a **U-shape**: a little shrinkage helps, too much hurts.

## Other settings (and what they taught)

- **`class_weight='balanced'`** gives the rarer class more weight. Here the *positive* class was the majority, so balancing made the model say YES less often: precision went up, recall went down, F1 went down. That's a **trade, not a failure**. Which one you want depends on which mistake costs more.
- **`fit_intercept=False`** forces the baseline prediction through zero. It made almost no difference here because other weights absorbed it.
- **`positive=True`** forces all weights ≥ 0. That's wrong when a feature really lowers the price (mpg, mileage), so it cost a lot. Use it only when a negative effect is impossible.

## Transforming the target: log(price)

The biggest improvement on cars came **not from any setting but from predicting log(price)**. Why: prices are skewed (a few very expensive cars) and depreciation is **multiplicative** (a car worth twice as much loses about twice as many euros). A straight line in euros can't follow that, but a straight line in log-euros can. You predict the log, then convert back to euros. This cut the error by about 15%.

## The tuning workflow (used from here on)

1. Change each setting **alone** to understand what it does (for learning only).
2. Run **one grid search over all combinations** together (in a project, go straight here). The best combination is often one no single-setting test visited.
3. Use **nested CV** to check if the search actually beats the default.

On the athletes, the tuned model was **not** better than the default under nested CV. On the cars, the log target was what paid off. **What tuning is worth depends on the dataset, and sometimes the answer is "nothing".**

**When a complex model only matches a simple one, choose the simple one**: it's faster, easier to explain, and there's less to go wrong.

## Check yourself

1. Why can't a linear model predict probabilities, even if its predictions happen to land in [0, 1]?
2. What is an odds ratio, and how do you get it from a logistic coefficient?
3. What's the difference between L1 and L2? What does small `C` mean?
4. Why did predicting log(price) help so much?
5. Why can a p-value be tiny without the feature causing anything?
