# Week 8 — K-Nearest Neighbours (KNN)

Sources: [classification solution](../solutions/week_08/week_08_similarity_knn_solution.ipynb) · [regression solution](../solutions/week_08/week_08_knn_regression_solution.ipynb)

## The big idea

KNN **learns nothing** during training. It just **stores** the training rows. To predict a new row it:

1. finds the **k closest** stored rows (the "neighbours"),
2. **classification:** takes a vote of their classes (the probability = share of neighbours in each class),
3. **regression:** averages their values (for example, their prices).

So everything depends on three things: **how you measure distance**, **how many neighbours vote**, and **how you find them**.

## Measuring "close": distance

- **Euclidean (p = 2):** the straight-line distance. Square each column's difference, add them, take the square root.
- **Manhattan (p = 1):** just add the absolute differences, like walking city blocks.
- **Minkowski:** the general formula; `p` picks which one. A bigger `p` lets the single largest difference dominate.

### Scaling is essential

Distance adds up column differences, so a column in **big units** (mileage in km) would completely dominate a column in small units (number of owners), regardless of which one actually matters. **Always scale features before KNN.**

## Choosing k (`n_neighbors`)

- **k = 1:** copy the single nearest row. Very flexible, so it **overfits**. Training score is perfect by construction (every training row is its own nearest neighbour).
- **Large k:** averages over a wide area. Smoother, but too large and it **underfits**.
- Typical pattern: training score falls as k grows, while validation rises, then flattens or falls.

**Small k = complex model, large k = simple model.**

## Weighting neighbours

- **`uniform`:** each neighbour gets one vote.
- **`distance`:** closer neighbours count more (weight = 1 / distance). Training score becomes perfect, since a row at distance 0 takes all the weight, so the training score tells you nothing here.

## What KNN can't do

- **Coarse probabilities:** with k = 5, the only possible probabilities are 0, 0.2, 0.4, 0.6, 0.8, 1.
- **No "I don't know":** a new row unlike anything in training still gets an answer from its (far away) neighbours, with no warning.
- **Can't extrapolate (regression):** a prediction is an average of training prices, so it can **never go outside the range of training values**.

## How the neighbours are found (`algorithm`)

This only changes **speed**, never the answer:
- **brute:** compute the distance to every stored row.
- **k-d tree / ball tree:** organise stored rows into boxes/balls so that whole regions can be skipped.

Trees help with **few columns and many rows**. With many columns (like ours), almost nothing can be skipped, so brute force is faster. `auto` picks for you (it chose brute here).

## Radius neighbours (the alternative)

Instead of "the k nearest", use "**everyone within distance r**". Problem: in crowded areas you get hundreds of neighbours, in empty areas none. **One radius can't suit crowded and sparse regions at the same time**, so it did worse than plain KNN.

## Result

The tuned KNN (Manhattan distance, distance weighting, larger k) beat the default in every fold and was competitive with logistic regression.

## Check yourself

1. What does KNN actually do during `fit`?
2. Why must you scale features before KNN?
3. Why does k = 1 always score perfectly on training data?
4. Why can't KNN regression predict a price higher than any training price?
5. Why didn't the k-d tree save time on our data?
6. Why does a fixed radius work worse than a fixed number of neighbours?
