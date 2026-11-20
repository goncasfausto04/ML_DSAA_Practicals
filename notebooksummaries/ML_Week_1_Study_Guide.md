# Week 1 — What Machine Learning Is (and When to Use It)

Source: [week 1 solution](../solutions/week_01/week_01_supervised_ml_and_environment_solution.ipynb)

## The big idea

Machine learning means **learning a rule from examples** instead of writing the rule yourself. You show the computer many past cases where you already know the answer, and it works out a pattern it can apply to new cases.

ML only makes sense when **all three** of these are true:

1. There is a real pattern to find (the inputs are actually related to the answer).
2. You can't easily write the rule by hand (otherwise just write the rule).
3. You have past examples **with the answer attached**.

If one is missing, use something simpler.

## Supervised vs unsupervised

- **Supervised:** each example comes with the answer (the *label* or *target*). The model learns to predict it, and you can check if it's right. **This is what the course is about.**
- **Unsupervised:** there's no answer column. You can only look for structure, like groups of similar rows, and you can't directly check if it's "right".

## Classification vs regression

This depends on **what kind of thing the target is**:

- **Classification:** the target is a category (win/lose, legal/illegal plantation).
- **Regression:** the target is a number on a scale (a car's price).

Trap: a category stored as a number (0/1) is still a category. Storing it as a number doesn't make the problem regression.

## Five questions that define any supervised problem

| Question | Meaning |
|---|---|
| **Observation** | What is one row? (one plantation, one athlete, one car) |
| **Features (X)** | What do we know about it? (the inputs) |
| **Target (y)** | What are we trying to predict? |
| **Prediction moment** | *When* do we need the answer? |
| **Decision** | What will someone do differently because of the prediction? |

The **prediction moment** decides which features you're allowed to use. If a piece of information only becomes available *after* the moment you need the prediction, using it is **data leakage**: the model looks great in testing and fails in real life. Example: you can't use the inspection result to decide which plantation to inspect.

## Labels are not perfect truth

Labels are usually decisions made by people (an inspector, a judge). If those people made mistakes or were biased, the model learns the mistakes too, and no score computed on the same data can show you that.

## A score means nothing on its own

Always compare against a **baseline**, a "dumb" model that does no work. If 73% of plantations are legal, a model that always says "legal" is 73% accurate but catches **zero** illegal ones. So 73% accuracy sounds good but is actually worthless. Ask: *better than what?* and *which mistakes matter?*

## Check yourself

1. What three conditions make a problem a good fit for supervised ML?
2. Why is "Outcome = 0/1" still a classification problem?
3. Why does the prediction moment matter for choosing features?
4. How can a model with high accuracy be useless?
