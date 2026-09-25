# Week 2 Study Guide: The Machine-Learning Process

This guide explains the complete pipeline built in `week_02_ml_pipeline.ipynb`.
The example predicts whether a plantation is a drug plantation from four drone measurements.

## The Main Idea

A machine-learning project is not only about training a model. The complete process is:

1. Define the decision.
2. Load the data.
3. Explore the data.
4. Prepare the data.
5. Train a model.
6. Tune the model.
7. Assess the model.
8. Use the model on new data.

The most important general lesson is this:

> A model is useful when it performs well on new, unseen data, not only on the data it memorised during training.

---

## 0. Identify the Business Need

The agency wants to decide which plantations should be inspected first.

The data contains:

- One row per plantation.
- Four input measurements: `BD1`, `BD2`, `BD3`, and `BD4`.
- A target column called `DrugPlant`.
- `DrugPlant = 1`: drug plantation.
- `DrugPlant = 0`: legal crop.

There are 300 labelled plantations and 40 new plantations without labels.

The model's real purpose is therefore not simply to print `0` or `1`. It should help rank the 40 new plantations by their probability of being drug plantations.

### What to retain

Always identify:

- What one row represents.
- Which columns are features.
- Which column is the target.
- When the prediction is made.
- What decision will be made from the prediction.

---

## 1. Import the Libraries

The notebook imports the tools needed for the pipeline:

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

RANDOM_STATE = 42
```

### What each library does

- `pandas`: reads and manipulates tables.
- `DecisionTreeClassifier`: creates the classification model.
- `train_test_split`: separates training and validation data.
- `confusion_matrix`: counts the different types of predictions.
- `f1_score`: measures performance using precision and recall.
- `matplotlib` and `seaborn`: create plots.
- `joblib`: saves and reloads the trained model.

### Why use `RANDOM_STATE`?

Random operations are used when splitting the data and when some model decisions are tied. Setting:

```python
RANDOM_STATE = 42
```

makes the results reproducible. Running the notebook again produces the same split and the same model behaviour.

---

## 2. Import the Data

The workbook contains two sheets:

```python
PLANTS = "../../data/raw/drug_plantations.xlsx"

plants_truth = pd.read_excel(
    PLANTS,
    sheet_name="ClassifiedData",
    index_col="ID",
)

plants_2classify = pd.read_excel(
    PLANTS,
    sheet_name="Data2Classify",
    index_col="ID",
)
```

### The two data objects

`plants_truth` contains labelled examples. It includes the features and the known target, so it is used for learning and evaluation.

`plants_2classify` contains new examples without a `DrugPlant` target. It is used only after the model is ready.

### Why use `ID` as the index?

The ID identifies a plantation, but it is not a measurement that should influence the prediction. Making it the index prevents the model from treating the ID as a feature.

### What to retain

Keep labelled training data and unlabelled future data separate. Never use the unlabelled data to train or tune the model.

---

## 3. Explore the Data

Before training a model, inspect the dataset.

### View rows

```python
plants_truth.head()
```

This checks whether the table looks as expected.

### Check types and missing values

```python
plants_truth.info()
```

This shows the number of rows, column names, data types, and non-null values.

### Check statistics

```python
plants_truth.describe()
```

This gives values such as count, mean, standard deviation, minimum, maximum, and quartiles.

### Check the class distribution

```python
plants_truth["DrugPlant"].value_counts()
```

The dataset is imbalanced: legal crops are the majority class, approximately 73% of the labelled data.

### Why imbalance matters

A model that always predicts `0` could achieve approximately 73% accuracy without detecting any drug plantations. Therefore, accuracy alone can be misleading.

This is the baseline to remember:

> Always compare the model with a simple baseline, such as always predicting the majority class.

### Compare feature means by class

```python
plants_truth.groupby("DrugPlant").mean()
```

This compares the average `BD1` to `BD4` values for legal and drug plantations. Features whose averages differ substantially may be useful for classification.

In this notebook, the exploratory comparison suggested that `BD4` and `BD1` would be especially informative.

### What to retain

Exploration helps you discover:

- Missing values.
- Incorrect data types.
- Unusual values.
- Class imbalance.
- Potentially useful features.
- Problems that could make the model unreliable.

Exploration does not prove that a feature causes the target. It only helps identify patterns worth testing.

---

## 4. Prepare the Data

### Separate features and target

```python
X = plants_truth.drop(columns="DrugPlant")
y = plants_truth["DrugPlant"]
```

- `X` contains the input measurements: `BD1`, `BD2`, `BD3`, and `BD4`.
- `y` contains the answers: `0` or `1`.

### Split into training and validation data

```python
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    train_size=0.7,
    shuffle=True,
    stratify=y,
    random_state=42,
)
```

For 300 labelled rows, this produces:

```python
len(X_train), len(X_val)
# (210, 90)
```

### Meaning of each argument

`train_size=0.7` uses 70% of the data for training and 30% for validation.

`shuffle=True` mixes the rows before splitting so that their original order does not determine the split.

`stratify=y` keeps approximately the same proportion of legal and drug plantations in both sets. This is especially important because the classes are imbalanced.

`random_state=42` makes the split repeatable.

### The golden rule: avoid leakage

The model may learn from:

```python
X_train, y_train
```

The validation data:

```python
X_val, y_val
```

must imitate future unseen data. Decisions about model settings should not be based on the validation data before evaluation.

Using information from validation data to train or prepare the model is called data leakage. Leakage makes the evaluation look better than the model's real future performance.

---

## 5. Create the First Model

The notebook uses a decision tree:

```python
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree.fit(X_train, y_train)
```

### How a decision tree works

A decision tree learns a sequence of questions about the features, such as:

- Is `BD4` greater than a particular threshold?
- Is `BD1` less than another threshold?
- Which branch should this row follow?

The final leaf produces a class prediction.

### Make predictions

```python
predictions_train = tree.predict(X_train)
predictions_val = tree.predict(X_val)
```

The model predicts both the rows it learned from and the rows it did not see during training.

### Measure accuracy and F1

```python
accuracy_train = tree.score(X_train, y_train)
f1_train = f1_score(y_train, predictions_train)

accuracy_val = tree.score(X_val, y_val)
f1_val = f1_score(y_val, predictions_val)
```

The first model produced approximately:

```text
Training accuracy:   1.0000
Validation accuracy: 0.8333
Training F1:         1.0000
Validation F1:       0.6939
```

### What this tells us

The unlimited tree classified every training row correctly, but it performed worse on unseen validation rows. This is evidence of overfitting.

Overfitting means the model learned the specific training examples too closely instead of learning patterns that generalise to new data.

### What to retain

A perfect training score is not automatically good news. Always compare training and validation performance.

---

## 6. Optimize the Model

The default tree has:

```python
max_depth=None
```

This means there is no explicit limit on how deep the tree can grow. It can continue splitting until the training examples are nearly perfectly separated.

That flexibility can cause overfitting.

### Test several depths

The notebook trains one candidate tree for each value:

```python
[1, 2, 3, 4, 5, 6, 8, 10, None]
```

For each candidate, it stores:

- Training accuracy.
- Validation accuracy.
- Training F1.
- Validation F1.

The results are placed in `depth_table`.

### Why compare training and validation curves?

If training accuracy keeps increasing while validation accuracy stops improving or decreases, the tree is becoming too complex and is memorising the training data.

A good choice usually has:

- Strong validation performance.
- A smaller gap between training and validation performance.
- Less complexity when performance is tied.

### Select the best depth

The notebook ranks candidates by:

1. Highest validation accuracy.
2. Smallest depth when validation accuracy is tied.

The final selected depth was:

```python
max_depth = 4
validation accuracy = 0.878
```

Depth 4 and depth 5 had the same validation accuracy, so depth 4 was selected because it is simpler.

### Rebuild the tuned tree

```python
tree_tuned = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=RANDOM_STATE,
)

tree_tuned.fit(X_train, y_train)
```

### Compare the tuned model

The tuned tree produced approximately:

```text
Training accuracy:   0.9619
Validation accuracy: 0.8778
Training F1:         0.9286
Validation F1:       0.7843
```

Compared with the default tree:

```text
Default validation accuracy: 0.833
Default validation F1:       0.694
Tuned validation accuracy:   0.878
Tuned validation F1:         0.784
```

The changes are approximately:

```text
Accuracy: +0.044
F1:       +0.090
```

The tuned tree gives up a little training performance but performs better on validation data. That is the desired trade-off because the goal is generalisation.

### Why choose using validation data?

The candidate trees were all trained on the training rows. Their validation scores show how well they perform on rows they did not learn from.

If we selected the tree using training accuracy, the unlimited tree would usually win because it can memorise the training rows. That would encourage overfitting.

### Important limitation

The validation rows were used to choose `max_depth`, so they are no longer completely neutral. A more rigorous project would use:

- Training data to fit models.
- Validation data to choose settings.
- A separate test set to report final performance.

Cross-validation is another common solution.

---

## 7. Assess the Tuned Model

### Confusion matrix

The tuned model produced this validation confusion matrix:

```text
[[59,  7],
 [ 4, 20]]
```

The layout is:

```text
[[TN, FP],
 [FN, TP]]
```

The rows are the true classes and the columns are the predicted classes.

| Value | Meaning                                                | Result |
| ----- | ------------------------------------------------------ | -----: |
| TN    | Legal crop correctly predicted as legal                |     59 |
| FP    | Legal crop incorrectly flagged as drug plantation      |      7 |
| FN    | Drug plantation incorrectly predicted as legal         |      4 |
| TP    | Drug plantation correctly predicted as drug plantation |     20 |

There are 90 validation rows:

```text
59 + 7 + 4 + 20 = 90
```

Accuracy is:

```text
(59 + 20) / 90 = 0.878
```

### Interpreting the errors

The 7 false positives represent wasted inspections. The agency inspects a legal plantation unnecessarily.

The 4 false negatives represent missed drug plantations. These may be more serious because the agency's main goal is to find drug plantations.

Therefore, the cost of a false negative may be higher than the cost of a false positive. The best metric depends on the real business cost of each error.

### Accuracy versus F1

Accuracy measures all correct predictions:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

F1 focuses on the positive class and balances precision and recall:

$$
F1 = 2 \times \frac{\text{precision} \times \text{recall}}
{\text{precision} + \text{recall}}
$$

Because legal crops are the majority class, accuracy can hide poor detection of drug plantations. F1 gives more useful information about the positive class.

### Feature importance

The tuned tree's feature importances were approximately:

```text
BD4: 0.52
BD1: 0.28
BD2: 0.16
BD3: 0.04
```

The model relied most on `BD4`, followed by `BD1`. This agrees with the earlier exploration of class means.

Feature importance means that a feature helped the tree make decisions. It does not prove that the feature causes drug plantations, and it is specific to this model and dataset.

---

## 8. Use the Model on New Data

The 40 new plantations do not have known labels. The model can still estimate their probability of belonging to class 1.

### Inspect the new data

```python
plants_2classify.head()
```

### Get class-1 probabilities

```python
scored = plants_2classify.copy()

class_1_column = list(tree_tuned.classes_).index(1)
scored["drug_probability"] = (
    tree_tuned.predict_proba(plants_2classify)[:, class_1_column]
)

scored = scored.sort_values("drug_probability", ascending=False)
scored.head(10)
```

`predict_proba` returns one probability column for each class. The columns follow the order in:

```python
tree_tuned.classes_
```

The code finds the column corresponding to class `1`, rather than assuming its position.

The rows are sorted from highest to lowest probability. The plantations at the top of the table should be considered first for inspection.

### Important distinction

A probability is not a guaranteed truth. A row with probability `1.0` is considered highly likely by this model, but it can still be wrong. The model's ranking should support inspection decisions, not replace them completely.

---

## 9. Export the Results

The ranked list is saved as a CSV file:

```python
scored.to_csv("week_02_inspection_ranking.csv")
print("Exported", len(scored), "ranked plantations.")
```

This creates a deliverable that can be opened or sent to the inspection team.

---

## 10. Save and Reload the Model

The fitted model is saved with `joblib`:

```python
joblib.dump(tree_tuned, "week_02_drug_plant_model.joblib")
```

It can later be loaded without training again:

```python
reloaded_model = joblib.load("week_02_drug_plant_model.joblib")
reused = reloaded_model.predict(plants_2classify)
```

This demonstrates a simple form of deployment: a future script can load the trained model and score new data.

---

## Essential Concepts to Remember

### Features and target

- Features are the inputs used to make predictions.
- The target is the answer the model is trying to predict.

### Training and validation

- Training data teaches the model.
- Validation data checks performance on unseen rows.
- Never train on validation data.

### Overfitting

- Training score much higher than validation score usually indicates overfitting.
- Limiting `max_depth` reduces tree complexity.

### Class imbalance

- A majority-class prediction can produce deceptively high accuracy.
- Inspect class counts and use metrics such as F1 when the positive class matters.

### Reproducibility

- Use a fixed `random_state` so results can be repeated.

### Tuning

- A hyperparameter is a model setting chosen before or during training.
- `max_depth` controls the maximum complexity of a decision tree.
- When scores tie, prefer the simpler model.

### Confusion matrix

Remember the order:

```text
[[TN, FP],
 [FN, TP]]
```

For this problem, false negatives are drug plantations that the model misses.

### Feature importance

Feature importance shows which features the tree used most, but it is not proof of causality.

### Deployment

A complete project should produce something usable: here, a ranked CSV and a saved model.

---

## Final Checklist

Before considering a classification project complete, ask:

- What decision will the model support?
- What are the features and target?
- Did I inspect missing values and data types?
- Are the classes balanced?
- Did I split the data before training?
- Did I use `stratify` when necessary?
- Did I keep validation data separate?
- Did I compare training and validation scores?
- Is the model overfitting?
- Is accuracy appropriate for the business problem?
- Did I inspect the confusion matrix?
- Did I tune the model using validation performance?
- Did I select a simpler model when scores tied?
- Did I save the model and the prediction results?

The central lesson from Week 2 is:

> The model is only one part of a machine-learning project. Good results depend on the whole pipeline: a clear decision, trustworthy data, careful evaluation, appropriate metrics, and a usable final output.
