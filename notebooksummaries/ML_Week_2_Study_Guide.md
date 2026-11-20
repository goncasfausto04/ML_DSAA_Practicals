# Week 2 — The ML Pipeline from Start to Finish

Source: [week 2 solution](../solutions/week_02/week_02_ml_pipeline_solution.ipynb)

## The big idea

An ML project is **a pipeline of stages**, and fitting the model is only one of them, usually not the one that decides whether the project is useful:

**business need → get data → explore → prepare → model → tune → assess → deploy**

Every choice you make along the way (how you fill gaps, which columns you drop) is a decision you should be able to explain afterwards.

## Why we split the data

The goal is a model that works on **new** data, not one that memorises the old data. So we hide part of the data from the model and use it to check how well it does.

| Set | Used for |
|---|---|
| **Training** | The model learns from it |
| **Validation** | Comparing settings and models to pick the best one |
| **Test** | One final, honest check of the chosen model. **Never** used to make choices |

Why keep the test set separate? Once you use a set to *choose* something, its score becomes too optimistic, because you picked whatever happened to do best on it. The test set stays clean so it can give an honest number.

**Stratification:** when splitting, keep the same proportion of each class in every part (for example, 60% winners in both train and validation).

**`random_state`:** fixes the randomness so you get the same split every time you run the code (reproducible).

## Parameters vs hyperparameters

- **Parameters** are what the model *learns* from data (for example, the threshold a tree splits on).
- **Hyperparameters** are settings *you choose* before training (for example, the maximum depth of a tree).

## Overfitting and underfitting

- **Overfitting:** the model memorises the training data, including its noise. You see a **great training score and a worse validation score**. Example: a very deep decision tree.
- **Underfitting:** the model is too simple to capture the pattern. The score is **bad on both** training and validation. Example: a tree with only one question.
- The sweet spot is in between, and **we find it using validation scores, never training scores.** If you tuned on training accuracy you would always pick the most complex model.

## Metrics: the confusion matrix

| | Predicted NO | Predicted YES |
|---|---|---|
| **Actually NO** | TN (true negative) | FP (false alarm) |
| **Actually YES** | FN (missed it) | TP (true positive) |

- **Accuracy:** share of all predictions that are right.
- **Precision:** of the ones I flagged YES, how many really were YES? (Am I wasting inspections?)
- **Recall:** of all the real YES cases, how many did I catch? (Am I missing criminals?)
- **F1:** a single number that balances precision and recall.

The same model can score 0.88 accuracy and 0.78 F1. **The metric you pick decides what "good" means**, so pick it based on which mistake costs more.

## Probabilities, not just labels

A classifier can give a **probability** (`predict_proba`) instead of just YES/NO. That's useful when you can only act on a limited number of cases: rank them by probability and inspect the top ones first.

## Deployment

Deployment means **using the model on new, unlabelled data**. You save the model *together with its preprocessing* to disk (with `joblib`), so next month's data gets treated exactly the same way.

## Check yourself

1. Why must the test set stay untouched while you tune?
2. What does "great on training, worse on validation" tell you?
3. What's the difference between precision and recall, in plain words?
4. Why save the preprocessing together with the model?
