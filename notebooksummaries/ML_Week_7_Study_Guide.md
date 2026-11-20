# Week 7 — Bayes: Naive Bayes and Bayesian Regression

Sources: [Naive Bayes solution](../solutions/week_07/week_07_naive_bayes_solution.ipynb) · [Bayesian regression solution](../solutions/week_07/week_07_bayesian_regression_solution.ipynb)

## The big idea

Logistic regression learns a **boundary** between classes directly. Bayes takes another route: model **what each class looks like**, then use Bayes' theorem to flip that around into "given what I see, which class is it?".

---

## Part A — Naive Bayes (classification)

### Bayes' theorem in words

> **posterior = likelihood × prior / evidence**

- **Prior** P(class): how common the class is *before* looking at the features (for example, 60% of athletes win).
- **Likelihood** P(features | class): how typical these feature values are *among* athletes of that class.
- **Evidence** P(features): the same for every class, so it only rescales the probabilities to add up to 1.
- **Posterior** P(class | features): what we want. **Predict the class with the biggest posterior.**

### Why we can't do "full" Bayes

To get the likelihood exactly, you'd count how many athletes of each class have **exactly this whole profile**. But the number of possible profiles explodes. It's a *product* of the number of values in each column: 12 columns gave ~870,000 combinations against only 3,200 athletes. Most new profiles were **never seen**, so the count is 0 and the answer is 0/0. With all columns, the table would need more memory than exists.

### The three "naive" assumptions

Naive Bayes fixes this by assuming:

1. **Independence given the class:** once you know the class, the features don't tell you anything about each other. So the likelihood becomes a simple **product of one factor per feature**.
2. **Equal weight:** every feature contributes equally (no feature counts more than another).
3. **A known shape:** each feature follows a simple distribution within each class (for example, a bell curve), so you only need a few numbers per feature.

Now training is just **counting and averaging**: class shares, plus a mean and variance per feature per class. No optimisation, very fast.

These assumptions are often **false** (training-volume columns were strongly correlated, and many columns are 0/1, not bell-shaped), yet the classifier can still work reasonably. It just isn't the strongest here.

### Versions of Naive Bayes (differ only in assumption 3)

| Version | Feature type |
|---|---|
| **GaussianNB** | Numeric, roughly bell-shaped |
| **BernoulliNB** | 0/1 features |
| **CategoricalNB** | Categories |
| **MultinomialNB** | Counts (for example, word counts) |

**Smoothing (α):** add a small number to every count so that a value never seen in training doesn't get probability exactly 0 and wipe out the whole product.

### Dropping the independence assumption: LDA / QDA

If you allow features to vary **together** (covariance):
- **QDA:** each class has its own covariance.
- **LDA:** one covariance shared by all classes. This gives a **linear boundary**, the same form as logistic regression.

LDA did much better than Naive Bayes here, close to logistic regression.

### Settings of GaussianNB

- **`var_smoothing`:** adds a small amount to every variance so that a feature that barely varies doesn't create a huge spike that dominates everything. Too much and the model collapses to "predict the majority class".
- **`priors`:** set the class shares by hand. Raising the prior on WIN shifts **every** athlete's odds the same way, so the boundary moves: **more YES → higher recall, lower precision**. It works like moving the threshold.

### Calibration: are the probabilities honest?

A model is **calibrated** if, among cases it gives ~70%, about 70% really happen. Naive Bayes is famously **overconfident**: it said ~95% on average while being right ~70% of the time.

`CalibratedClassifierCV` fixes this by learning, on rows the model didn't train on, a mapping from "model's probability" → "real frequency":
- **sigmoid (Platt):** an S-shaped curve.
- **isotonic:** any curve that only goes up (more flexible).

**F1 can't tell you if probabilities are calibrated.** F1 only checks labels, not how confident the model was.

---

## Part B — Bayesian regression (numbers)

### From one answer to a range

Normal linear regression gives **one** set of coefficients and **one** predicted price. Bayesian regression keeps a **distribution** over possible coefficients, so every prediction comes with a **range** (uncertainty).

- **Prior:** before seeing data, "coefficients are probably small" (a bell curve centred on 0).
- **Likelihood:** how well a set of coefficients explains the observed prices.
- **Posterior:** prior updated by the data. More data → narrower, more confident.

### The key insight: the prior IS a ridge penalty

A prior that says "coefficients are probably near zero" pulls coefficients toward zero, which is exactly what Ridge's penalty does. The most likely Bayesian coefficients **are the Ridge coefficients**, with the penalty strength learned from the data instead of searched for.

### The prediction interval has two parts

1. **Noise:** how much prices scatter around the line even if we knew the perfect coefficients. Same for every car, and usually the biggest part.
2. **Coefficient uncertainty:** how much the plausible models disagree about *this* car. It grows the further the car is from typical training cars.

**Coverage check:** a "95% interval" should contain about 95% of true prices. Here it did (~96%), so the model's uncertainty claim was honest.

### What the interval doesn't tell you

- Far from the data (a car from a very old year), the interval **widens**, which is good, but the prediction can become **impossible** (a negative price). The interval doesn't check if the input makes sense; a straight line just keeps going.
- Within normal data, interval width barely varied between cars, so it's a weak guide to which individual predictions are off.

### Settings (hyperpriors)

BayesianRidge **learns** its noise level and prior strength from data. What you set are beliefs *about* those (hyperpriors):
- Insisting too hard on small coefficients → all coefficients go to 0 → it predicts the average for everyone.
- Insisting the noise is tiny → MAE barely changes but **intervals become way too narrow** (coverage fell to ~3%).

**MAE can't see interval failures.** If you tune only on MAE, you can end up with a model whose uncertainty is wrong. The grid winner here had intervals that only covered ~53%.

## Check yourself

1. Say Bayes' theorem in words: prior, likelihood, posterior.
2. Why is "full" Bayes impossible with many columns?
3. What does the "naive" independence assumption buy you?
4. Why can a model have a high F1 and still badly calibrated probabilities?
5. Why is a zero-centred prior the same as a ridge penalty?
6. What does BayesianRidge give you that LinearRegression doesn't?
