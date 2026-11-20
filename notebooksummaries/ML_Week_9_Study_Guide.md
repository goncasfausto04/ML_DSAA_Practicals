# Week 9 — Decision Trees

Sources: [classification solution](../solutions/week_09/week_09_decision_trees_solution.ipynb) · [regression solution](../solutions/week_09/week_09_regression_trees_solution.ipynb)

## The big idea

A decision tree is a sequence of **yes/no questions** ("training hours ≤ 9?"). Each question splits the rows into two groups, and the tree keeps asking questions until the groups are "pure enough". A new row follows the questions down to a **leaf**, and the leaf gives the answer.

## How the tree picks a question

At each node the tree tries **every possible threshold of every feature** and picks the one that makes the two groups **least mixed**.

### Measuring "mixed": impurity

- **Gini impurity:** 0 = everyone in the group has the same class (pure). 0.5 = a perfect 50/50 mix (worst, for two classes).
- **Entropy:** a different formula with the same idea (0 = pure). Usually picks the same splits as Gini. `log_loss` is literally the same as entropy.
- **Regression trees:** use **squared error** (how spread out the values in the group are) instead.

The tree picks the split with the biggest **impurity decrease**: the impurity before, minus the impurity of the two children (weighted by their size).

It's **greedy**: it picks the best question *right now* without looking ahead.

## How a tree predicts

- **Classification:** the leaf answers with its majority class, and the probability = the share of each class in the leaf.
- **Regression:** the leaf answers with the **average** of the training values in it.

## Trees don't care about scale

A tree only asks "is the value above or below this threshold?", so only the **order** of values matters. Scaling or a log transform gives the same groups. **Trees don't need scaling.**

## The default tree overfits badly

By default the tree keeps splitting until every leaf is pure. Here that gave 21 levels and 504 leaves, many with **just 1 or 2 athletes**. Result: **training F1 = 1.0**, but validation was only about 0.80. It memorised the training data. A probability of "0" or "1" from such a tree really means "1 or 2 training athletes in this leaf", not certainty.

## Limiting the tree: pruning

### Pre-pruning: stop it growing

| Setting | What it limits |
|---|---|
| `max_depth` | How many questions in a row (at most 2^depth leaves) |
| `min_samples_leaf` | Each leaf must keep at least this many rows, so answers rest on more data |
| `min_samples_split` | A node needs at least this many rows to be allowed to split |

### Post-pruning: grow fully, then cut back

**`ccp_alpha` (cost-complexity pruning):** each leaf has a "price". A branch is removed if it doesn't reduce impurity enough to be worth its extra leaves. Bigger alpha = smaller tree. Unlike `max_depth`, it cuts weak branches **wherever they are**, at any depth.

All of these made the tree **smaller and better on validation**. A pruned tree has a **worse training score but a better validation score**, because it stopped memorising noise.

## Other settings

- **`splitter='random'`:** picks random thresholds instead of the best one. Faster, but a single tree gets worse. (This idea comes back in Extra Trees, week 10.)
- **`max_features`:** each split only looks at a random subset of the features. Worse for one tree. (This idea comes back in Random Forests, week 10.)
- **`class_weight='balanced'`:** makes the rarer class count more.

## What a tree can't do

- **Coarse answers:** only as many different predictions as there are leaves.
- **No "I don't know":** a strange new row still lands in some leaf and gets a confident-looking answer.
- **Can't extrapolate (regression):** a leaf predicts an average of training values, so it can never go beyond the training range.
- **Feature importance ≠ cause:** importance only says how much a feature reduced impurity in *this* tree. If two features are correlated, one may "steal" the importance of the other.

## Instability (key for week 10)

Change a few training rows and a deep tree can end up asking **completely different questions**. Trees are **unstable / high variance**, which is exactly why averaging many trees (bagging, forests) works so well.

## Check yourself

1. How does a tree choose which question to ask?
2. What does a Gini impurity of 0 mean? Of 0.5?
3. Why don't trees need feature scaling?
4. Why does the unpruned tree score 1.0 on training data but worse on validation?
5. What's the difference between pre-pruning and post-pruning?
6. Why can't a regression tree predict above the highest training price?
