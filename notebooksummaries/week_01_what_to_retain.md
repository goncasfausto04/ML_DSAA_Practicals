# Week 1: What to Retain

This guide condenses:

- [Week 1 supervised machine learning and environment](../solutions/week_01/week_01_supervised_ml_and_environment_solution.ipynb)

Week 1 is about understanding what a supervised-learning problem actually is
before writing modelling code.

> **Frame the decision first. A model predicts a target from features at a
> defined moment; it does not explain causes or make policy decisions.**

---

## 1. When is machine learning appropriate?

Supervised machine learning is appropriate when all three conditions hold:

1. A plausible pattern connects what is known to what should be predicted.
2. The rule cannot be written down reliably by hand, or it changes over time.
3. Past examples exist with the correct answer attached.

If one condition is missing, another tool may be better:

- a fixed formula for a known relationship;
- a lookup table for a stable mapping;
- a business rule for a clearly defined policy;
- collecting more labelled data when no examples with answers exist.

The goal is not to use ML because data exists. The goal is to use it when a
learnable, useful prediction is needed.

---

## 2. Supervised versus unsupervised learning

The difference is whether the training examples include the answer.

### Supervised learning

Each example has:

- input features;
- a known target label or value.

The model learns a relationship from the examples and applies it to new cases
where the target is not yet known.

Examples:

- classifying a plantation as drug or legal;
- predicting whether a customer will cancel;
- predicting a car price.

Because the historical answers are known, supervised learning can be evaluated:
we can compare predictions with the true targets.

### Unsupervised learning

There is no target column. The algorithm looks for structure, such as:

- groups of similar customers;
- unusual observations;
- lower-dimensional patterns;
- measurements that move together.

There is no labelled answer that automatically tells us whether the discovered
groups are correct. The result must be interpreted and judged for usefulness.

---

## 3. The five objects of every supervised problem

Learn to name these five objects:

1. **Observation:** the thing one row represents.
2. **Features:** information available to the model.
3. **Target:** what the model should predict.
4. **Prediction moment:** when the prediction must be made.
5. **Decision:** what someone does with the prediction.

### Week 1 example

For the plantation problem:

- observation: one plantation;
- features: drone measurements `BD1` to `BD4`;
- target: `DrugPlant`, drug plantation or legal crop;
- prediction moment: after the drone flight but before an inspection;
- decision: which plantations should inspection teams visit first.

The prediction moment controls which features are legitimate. An inspector's
report cannot be used as a feature if the model is supposed to rank plantations
before anyone visits them.

This is the foundation of preventing data leakage:

> Never give the model information that would not exist when the real
> prediction is made.

---

## 4. Prediction, explanation, cause, and decision are different

A model can identify a predictive pattern without explaining why it exists.

For example, if plantations with a certain drone measurement are more often
labelled as drug plantations, the model may use that relationship. That does
not prove the measurement causes the plantation to be a drug plantation.

Keep these ideas separate:

- **prediction:** what is likely to happen or what label is likely;
- **explanation:** why the pattern appears;
- **causation:** whether changing one thing changes another;
- **decision:** what an organisation chooses to do.

A predictive model can support a decision, but it does not make the policy or
prove a causal story by itself.

---

## 5. Where labels come from

Features may come directly from instruments, databases, or measurements. Labels
usually come from a human or organisational process.

In the plantation example, the `DrugPlant` label came from an inspector's
site-visit decision. Supervised learning learns to reproduce those decisions,
including any:

- mistakes;
- inconsistent standards;
- recording bias;
- missing context;
- outdated definitions.

Before trusting a target, ask:

- Who produced the label?
- What evidence did they use?
- Was the labelling rule consistent?
- Is the label actually the business outcome we care about?

A high model score can mean the model predicts the labelling process well. It
does not automatically mean it has discovered objective truth.

---

## 6. Classification versus regression

The target type defines the basic supervised task:

| Task | Target | Example |
|---|---|---|
| Classification | A category or class | Drug plantation versus legal crop |
| Regression | A numeric value | Used-car price or temperature |

Classification outputs may be labels or class probabilities. Regression outputs
numbers on a continuous scale.

The task type affects:

- which models are suitable;
- which metrics make sense;
- how predictions are used;
- what kinds of mistakes matter.

For example, later weeks use F1 for the imbalanced plantation classification
problem and MAE for price regression.

---

## 7. Class imbalance and the accuracy trap

The plantation data contains roughly three legal crops for every drug
plantation. The classes are imbalanced.

This matters because:

- a model can appear accurate by mostly predicting the majority class;
- there are fewer minority examples from which to learn;
- overall accuracy may hide failure to find the class the project actually
  cares about.

### The baseline or floor

Always compare a model with a simple do-nothing strategy. If a model always
predicts the majority class, its accuracy is the **baseline/floor** for that
problem.

For the plantation example, always answering “legal” achieves roughly 73%
accuracy. A reported accuracy is meaningful only when compared with that
baseline and with the costs of the two error types.

The baseline does not need to be sophisticated. It answers:

> Is the model doing more than the simplest available strategy?

---

## 8. The first practical data workflow

The Week 1 notebook also establishes the course environment:

- course repository containing the notebooks and data;
- Conda environment created from `environment.yml`;
- Jupyter Notebook for running cells;
- pandas for loading and inspecting data;
- scikit-learn for modelling;
- matplotlib for visualisation.

The important notebook habits are:

- run cells in order;
- read the output rather than treating code as the answer;
- keep the repository folder structure because data paths are relative;
- use the environment specified by the course;
- distinguish setup problems from ML reasoning problems.

The environment is the tool setup. It is not itself a machine-learning concept,
but a reproducible environment prevents “works on my machine” confusion.

---

## 9. What to remember from the plantation example

The example is deliberately simple and partly manual. The point is to see the
logic of supervised learning:

1. Start with labelled historical plantations.
2. Compare the measurements for the known classes.
3. Work out a rule that maps measurements to a class.
4. Apply the rule to new plantations without labels.
5. Evaluate the rule on known examples that stand in for future cases.
6. Use the result to support an inspection decision.

This is supervised learning even before an algorithm automates the rule.

The code and future models make the process more systematic; they do not change
the underlying logic.

---

## 10. Questions to ask before modelling any project

Write down answers to these before opening a model library:

1. What decision will the prediction support?
2. What is one observation?
3. What information is available at prediction time?
4. What is the target?
5. Is the target a category or a number?
6. Who created the labels, and can they be trusted?
7. Are the classes balanced?
8. What is the simplest baseline?
9. Which mistakes are more costly?
10. What metric reflects those costs?
11. What information would constitute leakage?
12. What will the prediction look like when delivered: label, number,
    probability, ranking, or report?

If these answers are unclear, fitting a model is premature.

## One-sentence memory aid

> **Define the decision, observation, features, target, and prediction moment;
> then choose the task, baseline, and metric before choosing a model.**
