# Machine Learning — Week 1: Supervised Learning and Problem Framing

## Glossary

| Key term | Meaning |
|---|---|
| Machine learning | Learning patterns from examples instead of manually specifying every prediction rule |
| Supervised learning | Learning from examples that contain a known target |
| Observation / sample | One row or case in a dataset |
| Feature / predictor (`X`) | Information available to the model for making a prediction |
| Target / label (`y`) | The outcome the model learns to predict |
| Classification | Predicting a discrete class or category |
| Regression | Predicting a numerical value |
| Prediction moment | The time when the prediction must be available |
| Decision | The action informed by a prediction |
| Baseline | A simple reference strategy that a model should beat |
| Generalization | Performing well on new, unseen examples |
| Data leakage | Allowing unavailable or held-out information to influence learning |
| Class imbalance | When one target class occurs much more often than another |
| Predictive association | A relationship useful for prediction that does not prove causation |

# Week 1 — Supervised Learning and Problem Framing

## The takeaways

1. **Start with the decision, not the algorithm.** ML is appropriate when historical labelled examples exist, a learnable pattern is plausible, and a reliable fixed rule is unavailable or insufficient.
2. **Know the five objects:** observation, features, target, prediction moment, and decision.
3. **Classification predicts a class; regression predicts a numeric value.** The target determines the task and appropriate evaluation metrics.
4. **Prediction is not causation.** A pattern can predict an outcome without explaining why it happens.
5. **Labels are not necessarily ground truth.** They may reproduce human judgement, error, and bias.
6. **Accuracy needs context.** An imbalanced dataset can give a high accuracy to a useless majority-class predictor.

## Concepts to understand

| Concept | Meaning | Course example |
|---|---|---|
| Machine learning | Learning patterns from examples rather than manually specifying every prediction rule | Learning from labelled plantations |
| Supervised learning | Training examples contain a known target | Drone measurements plus `DrugPlant` |
| Unsupervised learning | No known target; discover groups, structure, or unusual cases | Grouping similar plantations without labels |
| Observation / sample | One row or case | One plantation |
| Feature / predictor (`X`) | Information used to predict | `BD1`–`BD4` drone readings |
| Target / label (`y`) | The answer the model learns to predict | `DrugPlant` = 0 or 1 |
| Classification | Predict a discrete class | Drug versus legal plantation |
| Regression | Predict a numerical quantity | Car price in euros |
| Prediction moment | The time when the prediction must be available | After drone flight, before inspection |
| Baseline | Simple reference strategy a model should beat | Always predict the majority class |
| Class imbalance | One class occurs much more often | Legal plantations outnumber drug plantations |
| Generalization | Working on new examples, not only historical ones | Correctly ranking unseen plantations |
| Data leakage | Information unavailable at prediction time or from held-out data enters learning | Using inspection results as pre-inspection features |

### The five-object exercise

For the plantation problem:

- **Observation:** a plantation.
- **Features:** `BD1`, `BD2`, `BD3`, `BD4`.
- **Target:** `DrugPlant` (1 = drug, 0 = legal).
- **Prediction moment:** after measurements, before the inspector visits.
- **Decision:** which plantations inspectors should prioritize.

**Understand why the prediction moment matters:** An inspection report might predict the target extremely well, but it cannot be used to decide which plantation to inspect *before* inspection. That would be leakage.

### Accuracy trap

If roughly 73% of plantations are legal, a classifier that always says “legal” already achieves about **73% accuracy**, yet finds **zero** drug plantations. Thus accuracy alone is not proof of usefulness. Compare against a baseline and ask which mistakes matter.

### What you should be able to explain

- Why a lookup table or fixed rule can sometimes be better than ML.
- Why supervised learning requires historical targets.
- Why a numerical-looking column can still represent categories.
- Why predictive association does not prove causation.
- Why human-produced labels can be inconsistent.
- Why high accuracy on imbalanced data can be misleading.

### Practical knowledge

Know the purpose of `pandas`, `scikit-learn`, `matplotlib`, Jupyter, and the course Conda environment. You should be able to load a dataset, inspect its columns and values, and identify `X` and `y`. The environment setup supports reproducibility; it is not itself the learning algorithm.

### Check yourself

1. What are the five objects of a supervised problem?
2. Is predicting €15,000 a classification or regression task? What about predicting “expensive” versus “cheap”?
3. Why can a 73% accurate classifier still be useless?
4. What is wrong with using a feature recorded only after the prediction is supposed to happen?

**Week 1 in one sentence:** Define what is being predicted, from which available features, at what moment, for which decision, and against which baseline.

---

