# Week 2: What to Retain

This guide condenses:

- [Week 2 machine-learning pipeline](../solutions/week_02/week_02_ml_pipeline_solution.ipynb)

Week 2 introduces the complete machine-learning workflow. The notebook uses
drone measurements from plantations to predict whether a plantation is a
`DrugPlant`, then ranks new plantations for inspection.

> **A useful ML project is a pipeline from a real decision to a deployed
> prediction, not just a fitted model.**

---

## 1. The complete eight-stage pipeline

The Week 2 process is:

```text
business need
-> data
-> exploration
-> preparation
-> model
-> optimisation
-> assessment
-> deployment
```

Every later week deepens one part of this process.

### 0. Identify the business need

Before writing model code, define:

- **observation:** one plantation;
- **features:** drone measurements `BD1` to `BD4`;
- **target:** `DrugPlant`, where 1 means drug plantation and 0 means legal
  crop;
- **prediction moment:** after the drone flight but before inspection;
- **decision:** which plantations inspection teams should visit first.

This matters because the business request determines the output and metric. The
agency does not merely need a yes/no label. It needs a **ranked list** so that
limited inspection capacity can be used on the most suspicious plantations.

---

## 2. Import and understand the data

The Excel file contains two sheets:

- `ClassifiedData`: 300 plantations with confirmed labels;
- `Data2Classify`: 40 new plantations with measurements but no labels.

The labelled sheet is used to train and evaluate the model. The unlabelled
sheet is only used after the model has been selected, to produce the actual
inspection list.

The first inspection commands have different purposes:

- `.head()` checks that the file looks as expected;
- `.info()` checks data types and missing values;
- `.describe()` gives numeric summaries;
- `.value_counts()` checks class balance;
- `.groupby("DrugPlant").mean()` asks whether the features differ between
  classes.

Exploration is not decoration. It tells you whether the modelling assumptions,
data quality, class balance, and possible signals are plausible.

---

## 3. Separate features and target

The model needs two separate objects:

```text
X = input features
y = target to predict
```

The target must not remain inside `X`. If it does, the model is given the answer
while supposedly learning how to predict it.

This is a general rule:

> Features describe what is known at prediction time. The target is what the
> model is trying to learn and must be kept separate.

---

## 4. Split before fitting

The notebook uses a reproducible 70/30 train/validation split with
`stratify=y`.

- **Training data:** used to fit the model.
- **Validation data:** held back to estimate performance on unseen examples and
  choose between hyperparameters.

`stratify=y` keeps the class proportions similar in both portions. This is
important when one class is less common.

`random_state` makes the split reproducible. Without it, different runs could
produce different rows in each part and make debugging or comparison harder.

### The golden rule: avoid leakage

After the split, validation rows imitate the future. They must not influence:

- cleaning decisions;
- imputation values;
- feature choices;
- scaling;
- model fitting;
- hyperparameter selection.

If validation information enters any of these steps, the validation score is
optimistic. The model has indirectly seen the answers it is supposed to be
tested against.

For serious work, a single validation split is not enough. Later weeks use
repeated splits or cross-validation. Week 2 uses one split to make the full
pipeline understandable.

---

## 5. The first model: a decision tree

A decision tree learns a sequence of if/then questions, such as:

```text
Is BD1 greater than a learned threshold?
If yes, ask another question.
If no, follow a different branch.
```

The Week 2 tree can use the raw measurements because:

- it does not require feature scaling;
- it does not require categorical encoding for this numeric dataset;
- changing the units of a feature does not change the ordering of values or the
  threshold logic.

Important methods:

- `.fit(X_train, y_train)`: learn the tree;
- `.predict(X)`: return class labels;
- `.predict_proba(X)`: return class probabilities;
- `.score(X, y)`: return accuracy.

Important hyperparameters:

- `max_depth`: limits how many question levels the tree can grow;
- `criterion`: how candidate splits are evaluated, such as Gini or entropy;
- `min_samples_split`: how many observations a node needs before splitting;
- `random_state`: reproducible tie-breaking and behaviour.

The model is fitted only on training rows. It is then used separately on
training and validation rows so that overfitting can be diagnosed.

---

## 6. Overfitting and model complexity

A fully grown tree can memorise the training rows. This is overfitting:

- training performance is extremely high;
- validation performance is lower;
- the model learned details that do not generalise.

The default tree achieved approximately:

| Metric | Training | Validation |
|---|---:|---:|
| Accuracy | 1.000 | 0.833 |
| F1 | 1.000 | 0.694 |

The gap is evidence that the tree is too flexible.

Limiting `max_depth` makes the tree simpler. A simpler model may make more
training mistakes but can perform better on unseen data because it captures
general patterns rather than memorising individual examples.

This is the central bias/variance trade-off:

- too simple: underfitting, because the model cannot represent the pattern;
- too complex: overfitting, because the model follows noise;
- suitable complexity: enough flexibility to learn useful structure while
  generalising.

---

## 7. Hyperparameter tuning

The notebook tries several `max_depth` values and scores them on validation
rows. This is a basic form of hyperparameter tuning.

The rule is:

1. Fit each candidate on the training rows.
2. Score each candidate on the validation rows.
3. Choose the candidate with the best validation score.
4. Refit the chosen setting on the training rows.

Two depths tied on accuracy. F1 separated them: depth 4 achieved validation F1
of about 0.784, while depth 5 achieved about 0.756. The shallower tree was
therefore preferred.

When candidates are tied, prefer the simpler model. Simplicity can mean:

- smaller depth;
- fewer features;
- fewer parameters;
- lower operational cost.

### Important limitation

Once validation rows choose `max_depth`, they are no longer a completely
neutral final test. The winning validation score is slightly optimistic because
the rows helped choose the winner.

For an unbiased final estimate, use a third untouched test set or nested
cross-validation:

```text
training data -> choose settings
validation data -> compare settings
test data -> report final performance once
```

---

## 8. Metrics: what does “good” mean?

### Accuracy

Accuracy is:

```text
correct predictions / all predictions
```

The tuned tree achieved validation accuracy of about 0.878. However, always
compare this with a baseline. Predicting “legal” for every plantation already
achieved about 0.733 because of the class balance.

Accuracy can hide which class is being missed. A model can look good while
performing poorly on the class the agency cares about.

### F1

F1 combines precision and recall:

```text
F1 = 2 * precision * recall / (precision + recall)
```

- **Precision:** of the plantations predicted positive, how many really are
  positive?
- **Recall:** of the real positive plantations, how many did the model find?

The tuned tree achieved validation F1 of about 0.784. This is more aligned
with the agency's goal of finding drug plantations than accuracy alone.

The metric is part of the problem definition:

> A model is not “good” in the abstract. It is good relative to the decision,
> cost of mistakes, and metric chosen.

---

## 9. Confusion matrix

A confusion matrix shows the kinds of mistakes:

```text
[[TN, FP],
 [FN, TP]]
```

- **TN:** legal plantation correctly predicted as legal;
- **FP:** legal plantation incorrectly flagged;
- **FN:** drug plantation missed;
- **TP:** drug plantation correctly found.

Accuracy reduces these outcomes to one number. The confusion matrix explains
what the inspection agency will actually experience.

For this use case, false negatives may be especially important because they are
drug plantations the system fails to prioritise. False positives still cost
inspection time. The business should decide how those costs compare.

---

## 10. Deployment means using the output

The model's deliverable is not just a score. It is the ranked inspection list.

For the 40 unlabelled plantations:

1. use `predict_proba`;
2. take the probability of class 1;
3. store it as `drug_probability`;
4. sort from highest to lowest;
5. export the CSV for the inspection teams.

### Why `predict_proba` instead of `predict`?

`predict` returns a hard decision such as 0 or 1. That is not enough to rank
cases within the same class.

`predict_proba` returns a confidence-like probability for each class, allowing
the agency to inspect the highest-risk plantations first. The probability is
used for ordering, not automatically treated as a guarantee that a plantation
is truly positive.

### Persist the model

`joblib.dump` saves the fitted model to disk. Loading it again verifies that a
future scoring script can reuse the same fit without retraining.

In a production project, save the complete preprocessing-plus-model pipeline,
not only the final estimator. Otherwise future data may be transformed
differently from the data used during training.

---

## 11. What to remember from the code

The most important code ideas are conceptual:

- `X` and `y` represent predictors and target separately;
- `train_test_split(..., stratify=y, random_state=...)` creates a reproducible,
  class-balanced holdout;
- `.fit()` learns from training data;
- `.predict()` returns labels;
- `.predict_proba()` returns probabilities for ranking;
- `.score()` is model-specific and returns accuracy for this classifier;
- `f1_score(...)` measures positive-class performance more directly;
- `confusion_matrix(...)` shows the error types;
- `joblib` persists the fitted model.

Do not memorise method calls without understanding which rows each method is
allowed to see.

---

## 12. Exam and practical checklist

Before calling a model successful, ask:

1. What real decision will the prediction support?
2. What is one observation?
3. Which columns are features and which is the target?
4. What is known at prediction time?
5. Did I inspect class balance and data quality?
6. Did I split before fitting or learning anything?
7. Is the validation set protected from leakage?
8. Is the model overfitting?
9. Is accuracy appropriate, or do I need F1, precision, recall, or another
   metric?
10. What does the confusion matrix say about the actual mistakes?
11. Did I compare against a simple baseline?
12. Was the hyperparameter chosen using validation data?
13. Do I need an untouched test set for the final estimate?
14. Does deployment require labels, probabilities, a ranking, or a report?
15. Can I save and reload the exact model or full pipeline?

## One-sentence memory aid

> **Start with the decision, protect the holdout, control complexity, choose a
> metric that reflects the cost of mistakes, and deploy predictions in the form
> the user actually needs.**
