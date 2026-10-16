# Machine Learning — Week 1: Supervised Learning and Problem Framing

Source: [Supervised ML and environment solution](../solutions/week_01/week_01_supervised_ml_and_environment_solution.ipynb).

## Key learning points

- **Machine learning** learns patterns from examples. Use it when labelled history and a plausible predictive pattern exist; a reliable fixed rule may be enough for simpler problems.
- **Supervised learning** uses a known target. **Unsupervised learning** looks for structure without target labels.
- **Classification** predicts a category; **regression** predicts a numerical quantity. Numeric category codes do not turn classification into regression.
- **Generalization** means performing well on unseen examples, rather than remembering training cases.
- Predictive association does not prove causation. Labels can contain human errors and bias.

## Frame the problem before modelling

| Question | Plantation example |
|---|---|
| What is one observation? | One plantation |
| What are the features (`X`)? | Drone readings `BD1`–`BD4` |
| What is the target (`y`)? | `DrugPlant`: legal or drug plantation |
| When must the prediction be available? | After the drone flight, before inspection |
| What decision does it support? | Which plantations to inspect first |

**Data leakage:** using information unavailable at prediction time, or allowing held-out information to influence training. An inspection result cannot help decide which plantation to inspect beforehand.

## The baseline lesson

If about 73% of plantations are legal, always predicting “legal” gives about **73% accuracy** while finding no drug plantations. Compare a model with a simple baseline and ask which mistakes matter.

Know the tools' roles: `pandas` handles tables, `scikit-learn` fits and evaluates models, `matplotlib` plots, and Jupyter combines code with explanations. A consistent environment makes work reproducible.

## Check yourself

1. Can you identify the observation, features, target, prediction moment, and decision?
2. Why can a high-accuracy classifier be useless?
3. Why is a feature recorded after the prediction moment a leakage risk?
