# Week 3 — Exploring and Cleaning Data

Sources: [classification solution](../solutions/week_03/week_03_deepen_exploration_solution.ipynb) · [regression solution](../solutions/week_03/week_03_deepen_exploration_regression_solution.ipynb)

## The big idea

A raw file is not a dataset yet. **You** turn it into one through cleaning decisions, and those decisions affect the results as much as the model does. Good cleaning comes from **understanding what each column means**, not from applying automatic rules.

## Four different problems, four different fixes

Don't lump these together:

| Problem | What it is | Example |
|---|---|---|
| **Missing** | No value recorded | Empty cell |
| **Invalid** | A value that *can't* be true | Age = 0, a misspelt category, a placeholder like 999 |
| **Extreme (outlier)** | Unusual but possibly **real** | A very expensive car, an athlete who trains a lot |
| **Repeated** | Same entity appears more than once | One athlete with several rows |

Find the invalid values **first**. They often need to become "missing" before you count what's missing.

## Outliers: investigate, don't auto-delete

- The **IQR rule** (boxplot) flags values far from the middle 50% of the data. It's a *screening tool* ("look at these"), **not a deletion rule**.
- Auto-deleting outliers can throw away real, important cases. In the cars data the rule would have deleted about 30% of the file, including the whole cheap end of the market, and the "weird" mpg values turned out to be hybrids.
- **A better option:** `log1p` (log of 1 + x). It squashes long tails so extreme values matter less, **without removing any rows**.

## Missing values: filling them in (imputation)

| Method | Idea |
|---|---|
| Median / most common value | Simple and often good enough |
| "Missing" as its own category | Keeps the information that the value wasn't recorded |
| KNN imputation | Copy from similar rows |
| Iterative imputation | Predict the missing value from the other columns |

Fancier isn't automatically better. On the athletes the simple median won; on the cars KNN won. **Same technique, different result on different data.** You have to test.

## The golden rule: learn only from training rows

Any value you *compute* while cleaning (a median, the neighbours used by KNN) must be computed **from the training rows only**, then applied to validation/test rows. If you compute the median on the whole file, information from the test rows leaks into training and your score is too optimistic.

So what you save is the **recipe** ("fill with the training median"), not the fixed number.

## Repeated entities cause leakage

If the same athlete appears in both training and test, the model is partly tested on someone it has already seen. `drop_duplicates()` only catches *identical* rows, so you need to think about repeated **entities**, not just repeated rows.

## How we compare cleaning choices

Try each strategy on **the same set of repeated random splits** (20 here) and compare the average difference. A difference only means something if it's large compared to its uncertainty (the standard error, SEM). One split alone can't tell a real gain from luck.

## Check yourself

1. How is an invalid value different from a real outlier?
2. Why is the IQR rule a screening tool and not a deletion rule?
3. Why must the median be computed on training rows only?
4. Why might the athletes and the cars end up with different cleaning recipes?
