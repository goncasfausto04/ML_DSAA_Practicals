"""Rebuild a logged preprocessing recipe as unfitted, refittable objects.

A recipe is a JSON log naming the cleaning decisions, the categorical and
numeric columns, the encoding, the outlier treatment, the scaler and a feature
selection rule. `load_classification` and `load_regression` read one and return
it beside its frame; `classification_preprocessor` and `regression_preprocessor`
turn it into an unfitted transformer.

Everything here returns UNFITTED objects. Put one inside the model you fit, so
the fill values, the encoder levels, the scale and the selected columns are
learned from each fold's training rows alone. A preprocessor fitted on a whole
frame and split afterwards has let the held-out rows influence all four.

The log stores the selection RULE rather than the columns it chose, so refitting
it on different rows can keep a different set.
"""

from __future__ import annotations

import ast
import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency, pearsonr, pointbiserialr
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold, check_cv
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.utils import get_tags
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from course_helpers import CleaningLog


# Shape of the repeated cross-validation `outer_splitter` builds: five folds,
# two repeats, ten scores. Both constants are read there and nowhere else.


# -------------------------------------------------------------------------
# Reading a recipe and its frame
# -------------------------------------------------------------------------

def _decision(log, *required):
    """Return the single logged decision carrying every required field."""
    matches = [
        step.carries
        for step in log.steps
        if step.carries is not None
        and all(key in step.carries for key in required)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one decision carrying {required}, got {len(matches)}")
    return matches[0]

def _strategy(log):
    """Return the logged selection strategy, or None for a log without one."""
    matches = [step.carries for step in log.steps
               if step.carries is not None and "strategy" in step.carries]
    if len(matches) > 1:
        raise ValueError(f"expected one selection strategy, got {len(matches)}")
    return matches[0]["strategy"] if matches else None

def _recipe(log, *, task):
    """Assemble one reusable preprocessing recipe from logged decisions."""
    fill = _decision(log, "fill", "numeric", "categorical")
    transform = _decision(log, "transform", "columns")
    representation = _decision(log, "encoding", "scaler", "fitted_per_split")
    outliers = _decision(log, "outliers", "rule_tested")
    categorical = representation.get("categorical")
    if categorical is None:
        categorical = fill["categorical"]
    dropped = set(representation.get("dropped", []))
    transformation = transform["transform"]
    if transformation not in (None, "none", "identity", "log1p"):
        raise ValueError(f"unknown transform: {transformation!r}")
    return {
        "task": task,
        "numeric": [column for column in representation.get("numeric", fill["numeric"])
                    if column not in dropped],
        "categorical": [column for column in categorical if column not in dropped],
        "fill": dict(fill["fill"]),
        "log1p": list(transform["columns"]) if transformation == "log1p" else [],
        "encoding": representation["encoding"],
        "scaler": representation["scaler"],
        "outliers": outliers["outliers"],
        "drop": representation["drop"],
        "fitted_per_split": representation["fitted_per_split"],
        "dropped": list(representation.get("dropped", [])),
        "strategy": _strategy(log),
    }

def load_classification(data_path, log_path):
    """Load a classification frame with its logged dtypes and recipe."""
    log = CleaningLog.load(log_path)
    cast = _decision(log, "cast", "replace")["cast"]
    frame = pd.read_csv(
        data_path,
        dtype={column: "boolean" if dtype == "Boolean" else dtype
               for column, dtype in cast.items()},
    )
    return frame, _recipe(log, task="classification")

def load_regression(data_path, log_path):
    """Load a regression frame with its logged dtypes and recipe."""
    log = CleaningLog.load(log_path)
    dtypes = _decision(log, "dtypes")["dtypes"]
    frame = pd.read_csv(data_path, dtype=dtypes)
    return frame, _recipe(log, task="regression")


# -------------------------------------------------------------------------
# Turning a recipe into an unfitted preprocessor
# -------------------------------------------------------------------------

def _scaler(name):
    """Return a fresh scaler selected by its reusable configuration name."""
    if hasattr(name, "fit") and hasattr(name, "transform"):
        return clone(name)
    choices = {
        "none": None,
        None: None,
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "min-max": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if name not in choices:
        raise ValueError(f"unknown scaler: {name!r}")
    return choices[name]

def _encoder(name, *, drop="first"):
    """Return a fresh, unfitted encoder for a logged encoding name.

    `name` is "one-hot", "ordinal", "count" or "target", or an encoder object,
    which is cloned. `drop` is passed to the one-hot encoder only. "count" and
    "target" need `category_encoders`, imported inside their own branch so the
    other two work without the package installed.
    """
    if hasattr(name, "fit") and hasattr(name, "transform"):
        return clone(name)
    if name == "one-hot":
        return OneHotEncoder(
            handle_unknown="ignore", drop=drop, sparse_output=False
        )
    if name == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    if name == "count":
        from category_encoders import CountEncoder

        return CountEncoder(
            normalize=True, handle_unknown=0, handle_missing="value"
        )
    if name == "target":
        from category_encoders import TargetEncoder

        return TargetEncoder(handle_unknown="value", handle_missing="value")
    raise ValueError(f"unknown encoding: {name!r}")

def _outlier_limits(name):
    """Return the (low, high) quantile pair a logged treatment clips to.

    `None` means clip nothing. Treatments that DROP rows cannot be expressed
    here and raise: `transform` returns one row per row it is given and never
    sees `y`, so it cannot remove a row and its label together. Refusing is
    safer than ignoring the logged decision.
    """
    if name in (None, False, "keep", "none", "keep everything"):
        return None
    if name in ("winsorize", "winsorize 1/99"):
        return 0.01, 0.99
    if isinstance(name, (tuple, list)) and len(name) == 2:
        return name
    if isinstance(name, str) and name.startswith("drop"):
        raise ValueError(
            f"{name!r} removes training rows, which a transformer cannot do."
            " Clip instead, or drop the rows inside the fold loop itself,"
            " where the labels are in reach."
        )
    raise ValueError(f"unknown outlier treatment: {name!r}")

class FoldPreprocessor(BaseEstimator):
    """Fit the logged fill, encoding, transform, scale and selection on one fold."""

    def __init__(
        self,
        recipe,
        *,
        scaler="inherit",
        encoding="inherit",
        outliers="inherit",
        selection="inherit",
        log1p=True,
        feature_function=None,
    ):
        """Store the recipe and the choices that may override it.

        `recipe` is a loaded recipe mapping. Each of `scaler`, `encoding`,
        `outliers` and `selection` defaults to `"inherit"`, meaning use the
        logged choice; passing a value overrides it for this object only, which
        is what makes them searchable as `prepare__scaler` and so on.
        `selection=None` keeps every encoded column. `log1p` applies the logged
        log1p columns, and `feature_function` is an optional stateless callable
        applied to the frame before anything else.
        """
        self.recipe = recipe
        self.scaler = scaler
        self.encoding = encoding
        self.outliers = outliers
        self.selection = selection
        self.log1p = log1p
        self.feature_function = feature_function

    def _frame(self, X):
        """Apply an optional stateless feature function to a fresh frame."""
        return X if self.feature_function is None else self.feature_function(X)

    def _categorical_block(self, frame):
        """Fill categorical columns with values learned during fitting."""
        block = frame[self.categorical_].astype(object).copy()
        if self.recipe["fill"]["categorical"] == "own level":
            # A missing label and a nullable boolean need one common type.
            block = block.astype("string").astype(object)
        for column, value in self.categorical_fill_.items():
            block[column] = block[column].fillna(value)
        return block

    def _numeric_block(self, frame):
        """Fill, clip, and transform numeric columns with fold decisions."""
        block = frame[self.numeric_].apply(pd.to_numeric, errors="coerce")
        for column, value in self.numeric_fill_.items():
            block[column] = block[column].fillna(value)
        if self.outlier_bounds_ is not None:
            low, high = self.outlier_bounds_
            block = block.clip(lower=low, upper=high, axis=1)
        if self.log1p:
            block[self.log1p_] = np.log1p(block[self.log1p_])
        return block.to_numpy(dtype=float)

    def fit(self, X, y=None):
        """Learn fill values, one-hot levels, and scale from training rows only."""
        X = self._frame(X)
        self.categorical_ = list(self.recipe["categorical"])
        self.numeric_ = list(self.recipe["numeric"])
        expected = set(self.categorical_ + self.numeric_)
        missing = expected.difference(X.columns)
        if missing:
            raise ValueError(f"recipe columns missing from frame: {sorted(missing)}")

        fill = self.recipe["fill"]
        if fill["categorical"] == "mode":
            self.categorical_fill_ = {
                column: (X[column].mode(dropna=True).iloc[0]
                         if not X[column].mode(dropna=True).empty else "(missing)")
                for column in self.categorical_
            }
        elif fill["categorical"] == "own level":
            self.categorical_fill_ = {
                column: "(missing)" for column in self.categorical_
            }
        else:
            raise ValueError(f"unknown categorical fill: {fill['categorical']!r}")

        numeric = X[self.numeric_].apply(pd.to_numeric, errors="coerce")
        if fill["numeric"] == "median":
            self.numeric_fill_ = numeric.median().to_dict()
        elif fill["numeric"] == "mean":
            self.numeric_fill_ = numeric.mean().to_dict()
        elif fill["numeric"] == "zero":
            self.numeric_fill_ = dict.fromkeys(self.numeric_, 0)
        else:
            raise ValueError(f"unknown numeric fill: {fill['numeric']!r}")

        # Quantiles come off the raw training cells, before the fill runs. A
        # median fill piles identical values at the centre of a column, so a
        # quantile taken afterwards is measured partly over invented cells.
        # `quantile` skips gaps, which is what allows this order.
        treatment = (self.recipe.get("outliers", "keep")
                     if self.outliers == "inherit" else self.outliers)
        limits = _outlier_limits(treatment)
        self.outlier_bounds_ = None
        if limits is not None:
            lower, upper = limits
            self.outlier_bounds_ = (numeric.quantile(lower),
                                    numeric.quantile(upper))

        self.log1p_ = [column for column in self.recipe["log1p"]
                       if column in self.numeric_]

        self.encoding_ = (self.recipe["encoding"] if self.encoding == "inherit"
                          else self.encoding)
        self.encoder_ = (_encoder(self.encoding_, drop=self.recipe["drop"])
                         if self.categorical_ else None)
        block = self._categorical_block(X)
        if self.encoder_ is not None:
            if self.encoding_ == "target":
                if y is None:
                    raise ValueError(
                        "target encoding reads the labels, so it has to be"
                        " fitted inside a fold that supplies y"
                    )
                self.encoder_.fit(block, y)
            else:
                self.encoder_.fit(block)
        chosen = self.recipe["scaler"] if self.scaler == "inherit" else self.scaler
        self.scaler_ = _scaler(chosen) if self.numeric_ else None
        numeric_block = self._numeric_block(X)
        if self.scaler_ is not None:
            self.scaler_.fit(numeric_block)
        # A number an encoder made is still a number. Ordinal, count and target
        # encoding put a level on a scale of their own, so the chosen scaler is
        # fitted on those columns too, on the same training rows. One-hot
        # indicators stay 0/1: their spread only records how common a level is.
        self.label_scaler_ = None
        if self.categorical_ and self.encoding_ != "one-hot":
            self.label_scaler_ = _scaler(chosen)
        if self.label_scaler_ is not None:
            self.label_scaler_.fit(
                np.asarray(self.encoder_.transform(block), dtype=float)
            )
        self.n_features_in_ = X.shape[1]
        self.encoded_names_ = self._encoded_names()
        self.selection_ = self._fit_selection(X, y)
        self.support_ = (
            np.ones(len(self.encoded_names_), dtype=bool)
            if self.selection_ is None else np.asarray(self.selection_.get_support())
        )
        return self

    def _encoded_names(self):
        """Encoded label names followed by the logged numeric names."""
        labels = (self.encoder_.get_feature_names_out(self.categorical_)
                  if isinstance(self.encoder_, OneHotEncoder)
                  else self.categorical_)
        return np.asarray([*labels, *self.numeric_], dtype=object)

    def _encoded(self, frame):
        """The full encoded, transformed and scaled matrix, before selection."""
        labels = np.asarray(
            self.encoder_.transform(self._categorical_block(frame)), dtype=float
        ) if self.encoder_ is not None else np.empty((len(frame), 0))
        if self.label_scaler_ is not None:
            labels = self.label_scaler_.transform(labels)
        numbers = self._numeric_block(frame)
        if self.scaler_ is not None:
            numbers = self.scaler_.transform(numbers)
        return np.hstack([labels, numbers])

    def _fit_selection(self, X, y):
        """Fit the selection the log names on THIS fold's training rows.

        The log carries the rule, not the columns Week 4 kept, so the rule is
        rebuilt and refitted here. A different fold may keep a different set,
        which is the reason a rule travels and a column list does not.
        """
        chosen = (self.recipe.get("strategy") if self.selection == "inherit"
                  else self.selection)
        if chosen is None:
            return None
        if y is None:
            raise ValueError(
                "the logged selection reads the target, so it has to be fitted"
                " inside a fold that supplies y"
            )
        width = len(self.encoded_names_) - len(self.numeric_)
        scores = effect_scores_for(self.recipe["task"], width)
        selector = (strategy_from_spec(chosen, scores=scores)
                    if isinstance(chosen, dict) else copy.deepcopy(chosen))
        frame = pd.DataFrame(self._encoded(X), columns=self.encoded_names_)
        fitted = selector.fit(frame, y)
        if not np.asarray(fitted.get_support()).any():
            raise ValueError(
                "the logged selection kept no columns on these rows. A penalty"
                " in the logged rule is expressed in the target's own units, so"
                " a week that rescales or transforms the target has to refit"
                " that rule rather than inherit it, or switch selection off and"
                " say why"
            )
        return fitted

    def transform(self, X):
        """Apply training-fold decisions without learning from these rows."""
        check_is_fitted(self, "encoder_")
        return self._encoded(self._frame(X))[:, self.support_]

    def fit_transform(self, X, y=None):
        """Fit on supplied rows and return their transformed matrix."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return the names of the columns transform keeps, in matrix order."""
        check_is_fitted(self, "encoder_")
        return self.encoded_names_[self.support_]

def classification_preprocessor(
    recipe,
    *,
    scaler="inherit",
    encoding="inherit",
    outliers="inherit",
    selection="inherit",
    log1p=True,
    feature_function=None,
):
    """Return an unfitted preprocessor for a classification recipe.

    `recipe` must be a classification recipe. The keyword arguments override the
    logged choices and are documented on `FoldPreprocessor`.
    """
    if recipe["task"] != "classification":
        raise ValueError("classification_preprocessor needs a classification recipe")
    return FoldPreprocessor(
        recipe,
        scaler=scaler,
        encoding=encoding,
        outliers=outliers,
        selection=selection,
        log1p=log1p,
        feature_function=feature_function,
    )

def regression_preprocessor(
    recipe,
    *,
    scaler="inherit",
    encoding="inherit",
    outliers="inherit",
    selection="inherit",
    log1p=True,
    feature_function=None,
):
    """Return an unfitted preprocessor for a regression recipe.

    `recipe` must be a regression recipe. The keyword arguments override the
    logged choices and are documented on `FoldPreprocessor`.
    """
    if recipe["task"] != "regression":
        raise ValueError("regression_preprocessor needs a regression recipe")
    return FoldPreprocessor(
        recipe,
        scaler=scaler,
        encoding=encoding,
        outliers=outliers,
        selection=selection,
        log1p=log1p,
        feature_function=feature_function,
    )


# -------------------------------------------------------------------------
# The selection rule the log carries
# -------------------------------------------------------------------------

def _cramers_v(column, outcome):
    """Chi-square on a contingency table, rescaled to a [0, 1] effect size."""
    table = pd.crosstab(column, outcome)
    if min(table.shape) < 2:
        return 0.0
    statistic, _, _, _ = chi2_contingency(table, correction=False)
    total = table.to_numpy().sum()
    return float(np.sqrt(statistic / (total * (min(table.shape) - 1))))

def classification_effect_scores(matrix, outcome, categorical_width=0):
    """One comparable [0, 1] effect per encoded column, read by its type.

    A 0/1 dummy is categorical evidence and takes Cramer's V; anything else is
    a measurement and takes |point-biserial r|. An encoder that replaces a level
    with a number produces measurements, so those columns are read as such.
    """
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(outcome, dtype=int)
    scores = []
    for position in range(matrix.shape[1]):
        values = matrix[:, position]
        if position < categorical_width and np.isin(values, [0.0, 1.0]).all():
            effect = _cramers_v(pd.Series(values), pd.Series(target))
        else:
            effect = abs(pointbiserialr(target, values).statistic)
        scores.append(effect)
    return np.nan_to_num(np.asarray(scores, dtype=float))

def regression_effect_scores(matrix, outcome, categorical_width=0):
    """|Pearson r| against a numeric target, for every encoded column."""
    target = np.asarray(outcome, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.abs([pearsonr(matrix[:, position], target).statistic
                         for position in range(matrix.shape[1])])
    return np.nan_to_num(scores)

def effect_scores_for(task, categorical_width=0):
    """The relevance index one task reads its encoded columns with."""
    index = (classification_effect_scores if task == "classification"
             else regression_effect_scores)
    return lambda matrix, outcome: index(matrix, outcome, categorical_width)

class CorrelationFilter:
    """Rank by relevance, then walk the ranking and skip repeats.

    A candidate is skipped when it correlates above `redundancy` with a column
    already kept. With `redundancy=None` this is the ranking alone, and with
    `k=None` there is no budget to exhaust, so the redundancy rule runs by
    itself.
    """

    def __init__(self, k, redundancy=None, scores=None):
        """Store the budget, the redundancy limit, and the relevance index."""
        self.k = k
        self.redundancy = redundancy
        self.scores = scores

    def fit(self, frame, outcome):
        """Rank these training rows, then keep what repeats nothing kept."""
        matrix = frame.to_numpy(dtype=float)
        scores = self.scores or effect_scores_for("regression")
        with np.errstate(invalid="ignore", divide="ignore"):
            relevance = scores(matrix, outcome)
            between = np.nan_to_num(np.abs(np.corrcoef(matrix, rowvar=False)))
        keep = []
        for candidate in np.argsort(relevance)[::-1]:
            if len(keep) == self.k:
                break
            if self.redundancy is not None and any(
                between[candidate, chosen] > self.redundancy for chosen in keep
            ):
                continue
            keep.append(int(candidate))
        self.support_ = np.zeros(matrix.shape[1], dtype=bool)
        self.support_[keep] = True
        return self

    def get_support(self):
        """The mask of columns this fit keeps."""
        return self.support_

class Sequence:
    """Apply selectors one after another, each on what the previous one left."""

    def __init__(self, *stages):
        """Store the named stages in the order they run."""
        self.stages = stages

    def fit(self, frame, outcome):
        """Fit each stage on the columns its predecessors left standing."""
        self.feature_names_in_ = frame.columns.to_numpy()
        support = np.ones(frame.shape[1], dtype=bool)
        for _, stage in self.stages:
            survivors = frame.loc[:, support]
            kept = stage.fit(survivors, outcome).get_support()
            support[np.flatnonzero(support)] = kept
        self.support_ = support
        return self

    def get_support(self):
        """The mask left after every stage has run."""
        return self.support_

class Vote:
    """Keep the columns that at least `minimum` of the members select."""

    def __init__(self, members, minimum):
        """Store the named members and how many of them a column needs."""
        self.members = members
        self.minimum = minimum

    def fit(self, frame, outcome):
        """Fit every member on the same rows and count the agreements."""
        self.feature_names_in_ = frame.columns.to_numpy()
        self.decisions_ = pd.DataFrame(
            {name: member.fit(frame, outcome).get_support()
             for name, member in self.members.items()},
            index=frame.columns,
        )
        self.support_ = (self.decisions_.sum(axis=1) >= self.minimum).to_numpy()
        return self

    def get_support(self):
        """The mask of columns enough members agreed on."""
        return self.support_

# Where a logged name is looked up. The log names the techniques, so nothing
# here is a list of the ones this course happens to use: a name is resolved
# against these libraries and the selectors above, and anything else is refused
# by name rather than executed.
SELECTION_LIBRARIES = (
    "sklearn.feature_selection",
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.tree",
    "sklearn.svm",
    "sklearn.neighbors",
)

def _resolve(name, local):
    """Find one logged name among the local selectors or the libraries."""
    if name in local:
        return local[name]
    for library in SELECTION_LIBRARIES:
        found = getattr(importlib.import_module(library), name, None)
        if found is not None:
            return found
    raise ValueError(
        f"the log asks for {name}, which is neither defined here nor found in"
        f" {', '.join(SELECTION_LIBRARIES)}"
    )

def _construct(node, local):
    """Build one parsed constructor call, and refuse everything else."""
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"a logged selector is one call: {ast.unparse(node)}")
        if node.args:
            raise ValueError("a logged selector is written with keywords only")
        return _resolve(node.func.id, local)(
            **{keyword.arg: _construct(keyword.value, local)
               for keyword in node.keywords}
        )
    if isinstance(node, ast.Name):
        return _resolve(node.id, local)
    return ast.literal_eval(node)

def selector_from_text(text, *, scores=None):
    """Rebuild one unfitted selector from the call the log recorded."""
    local = {
        "CorrelationFilter":
            lambda k, redundancy=None: CorrelationFilter(k, redundancy, scores=scores),
        "Sequence": Sequence,
        "Vote": Vote,
    }
    if scores is not None:
        local["encoded_effect_scores"] = scores
    return _construct(ast.parse(text, mode="eval").body, local)

def strategy_from_spec(spec, *, scores=None):
    """Rebuild an unfitted selector from an exported strategy specification.

    `spec["kind"]` is "vote", "sequence" or "single", and the remaining keys
    name the members: a mapping plus a `minimum` for a vote, an ordered list of
    stages for a sequence, one call string for a single selector. `scores` is
    the relevance index passed to any member that takes one.
    """
    kind = spec["kind"]
    if kind == "vote":
        members = {name: selector_from_text(text, scores=scores)
                   for name, text in spec["members"].items()}
        return Vote(members, spec["minimum"])
    if kind == "sequence":
        return Sequence(*[(stage["stage"],
                           selector_from_text(stage["selector"], scores=scores))
                          for stage in spec["stages"]])
    if kind == "single":
        return selector_from_text(spec["selector"], scores=scores)
    raise ValueError(f"unknown strategy kind: {kind!r}")


# -------------------------------------------------------------------------
# Pairing preprocessing with an estimator
# -------------------------------------------------------------------------

def _estimator_provides(attribute):
    """True when the wrapped estimator can answer `attribute`.

    Scikit-learn picks a scorer's response method with `hasattr`, so a wrapper
    that defines `decision_function` for every estimator makes that check lie.
    A forest then gets asked for decision scores it does not have, the scorer
    raises inside `cross_validate`, and the fold's score becomes NaN with only
    a warning. Guarding the attribute keeps the answer honest.
    """
    def check(self):
        return hasattr(self.estimator, attribute)

    return check

class PreparedEstimator(BaseEstimator):
    """Fit one unfitted preprocessor and one estimator on the same training rows."""

    def __init__(
        self,
        preprocessor,
        estimator,
        *,
        preprocessor_name="prepare",
        estimator_name="model",
    ):
        """Store unfitted preprocessing and estimation components."""
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.preprocessor_name = preprocessor_name
        self.estimator_name = estimator_name

    def __sklearn_tags__(self):
        """Expose the wrapped estimator type to splitters and calibration."""
        tags = super().__sklearn_tags__()
        model_tags = get_tags(self.estimator)
        tags.estimator_type = model_tags.estimator_type
        tags.target_tags = copy.deepcopy(model_tags.target_tags)
        tags.classifier_tags = copy.deepcopy(model_tags.classifier_tags)
        tags.regressor_tags = copy.deepcopy(model_tags.regressor_tags)
        return tags

    def get_params(self, deep=True):
        """Expose component parameters under stable configurable aliases."""
        params = super().get_params(deep=deep)
        if deep:
            for name, value in self.preprocessor.get_params(deep=True).items():
                params[f"{self.preprocessor_name}__{name}"] = value
            for name, value in self.estimator.get_params(deep=True).items():
                params[f"{self.estimator_name}__{name}"] = value
        return params

    def set_params(self, **params):
        """Translate stable aliases before assigning component parameters."""
        translated = {}
        for name, value in params.items():
            if name.startswith(f"{self.preprocessor_name}__"):
                suffix = name.split("__", 1)[1]
                translated[f"preprocessor__{suffix}"] = value
            elif name.startswith(f"{self.estimator_name}__"):
                suffix = name.split("__", 1)[1]
                translated[f"estimator__{suffix}"] = value
            else:
                translated[name] = value
        return super().set_params(**translated)

    @property
    def named_steps(self):
        """Expose fitted or unfitted components by their configured aliases."""
        preprocessor = getattr(self, "preprocessor_", self.preprocessor)
        estimator = getattr(self, "estimator_", self.estimator)
        return {
            self.preprocessor_name: preprocessor,
            self.estimator_name: estimator,
        }

    def fit(self, X, y):
        """Fit preprocessing and then the estimator on the transformed rows."""
        self.preprocessor_ = clone(self.preprocessor).fit(X, y)
        transformed = self.preprocessor_.transform(X)
        self.estimator_ = clone(self.estimator).fit(transformed, y)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        """Predict after applying the fitted preprocessing component."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self.preprocessor_.transform(X))

    @available_if(_estimator_provides("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities after fitted preprocessing."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(self.preprocessor_.transform(X))

    @available_if(_estimator_provides("decision_function"))
    def decision_function(self, X):
        """Return decision scores after fitted preprocessing."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(self.preprocessor_.transform(X))

    def score(self, X, y):
        """Score transformed rows with the fitted estimator's default metric."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.score(self.preprocessor_.transform(X), y)

class FittedModel:
    """A fitted fold preprocessor and estimator with a prediction interface."""

    def __init__(self, preprocessor, estimator):
        """Store one fitted preprocessing and estimation pair."""
        self.preprocessor = preprocessor
        self.estimator = estimator

    def predict(self, X):
        """Predict after applying the stored preprocessing object."""
        return self.estimator.predict(self.preprocessor.transform(X))

    @available_if(_estimator_provides("predict_proba"))
    def predict_proba(self, X):
        """Predict probabilities after applying stored preprocessing."""
        return self.estimator.predict_proba(self.preprocessor.transform(X))

    @available_if(_estimator_provides("decision_function"))
    def decision_function(self, X):
        """Return decision scores after applying stored preprocessing."""
        return self.estimator.decision_function(self.preprocessor.transform(X))


# -------------------------------------------------------------------------
# The shared cross-validation design
# -------------------------------------------------------------------------

OUTER_SPLITS = 5

OUTER_REPEATS = 2

def outer_splitter(task, *, random_state=42):
    """Return a repeated K-fold splitter, stratified for classification.

    `task` is "classification" or "regression" and picks
    RepeatedStratifiedKFold or RepeatedKFold. `random_state` seeds the fold
    assignment, so two calls with the same task and seed yield the same folds
    and the scores they produce are paired.
    """
    if task not in ("classification", "regression"):
        raise ValueError(
            f"task must be classification or regression, not {task!r}"
        )
    design = (
        RepeatedStratifiedKFold if task == "classification" else RepeatedKFold
    )
    return design(
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=random_state
    )

def outer_folds(splitter, X, y):
    """Return the row labels each fold scores, as one list per fold.

    `splitter` is any sklearn splitter, `X` and `y` the rows to divide. Labels
    come from `X.index` when it has one and are positions otherwise. Two runs
    that produce equal lists divided the same rows the same way.
    """
    return [
        X.index[score_at].tolist() if hasattr(X, "index") else list(score_at)
        for _, score_at in splitter.split(X, y)
    ]


# -------------------------------------------------------------------------
# Preparing each fold once, and reusing it
# -------------------------------------------------------------------------

def _take_rows(values, positions):
    """Select positional rows from pandas or array-like values."""
    return values.iloc[positions] if hasattr(values, "iloc") else values[positions]

def prepared_folds(preprocessor, splitter, X, y):
    """Fit preprocessing once per fold and cache matrices for every candidate."""
    folds = []
    splits = splitter.split(X, y) if hasattr(splitter, "split") else splitter
    for fit_at, score_at in splits:
        X_fit, X_score = _take_rows(X, fit_at), _take_rows(X, score_at)
        y_fit, y_score = _take_rows(y, fit_at), _take_rows(y, score_at)
        fitted = clone(preprocessor).fit(X_fit, y_fit)
        folds.append({
            "fit_at": fit_at,
            "score_at": score_at,
            "X_fit": fitted.transform(X_fit),
            "X_score": fitted.transform(X_score),
            "y_fit": y_fit,
            "y_score": y_score,
        })
    return folds

class FoldCache:
    """Cache independently prepared matrices for one dataset and splitter."""

    def __init__(self, splitter, X, y):
        """Split `X` and `y` once with `splitter` and cache nothing yet."""
        self.X = X
        self.y = y
        self.splits = list(splitter.split(X, y))
        self._folds = {}

    @staticmethod
    def _key(preprocessor):
        """Create a stable cache key from preprocessing parameters."""
        return tuple(
            sorted(
                (name, repr(value))
                for name, value in preprocessor.get_params(deep=True).items()
            )
        )

    def get(self, preprocessor):
        """Return cached fold matrices, preparing them only when absent."""
        key = self._key(preprocessor)
        if key not in self._folds:
            self._folds[key] = prepared_folds(
                preprocessor, self.splits, self.X, self.y
            )
        return self._folds[key]

class SplitCache:
    """Cache one training-fitted representation for a fixed holdout split."""

    def __init__(self, X_fit, y_fit, X_score):
        """Store raw fit and score rows behind an empty preparation cache."""
        self.X_fit = X_fit
        self.y_fit = y_fit
        self.X_score = X_score
        self._prepared = {}

    def get(self, preprocessor):
        """Return shared fitted preprocessing and matrices for one choice."""
        key = FoldCache._key(preprocessor)
        if key not in self._prepared:
            fitted = clone(preprocessor).fit(self.X_fit, self.y_fit)
            self._prepared[key] = {
                "preprocessor": fitted,
                "X_fit": fitted.transform(self.X_fit),
                "y_fit": self.y_fit,
            }
        return self._prepared[key]

def fit_on_cached_split(model, cache):
    """Fit one prepared estimator without repeating cached preprocessing."""
    if not isinstance(model, PreparedEstimator):
        raise TypeError("fit_on_cached_split requires a PreparedEstimator")
    prepared = cache.get(model.preprocessor)
    fitted_estimator = clone(model.estimator).fit(
        prepared["X_fit"], prepared["y_fit"]
    )
    return FittedModel(prepared["preprocessor"], fitted_estimator)


# -------------------------------------------------------------------------
# Searching over candidates
# -------------------------------------------------------------------------

def candidate_scores(estimator, parameter_grid, folds, *, scoring):
    """Score all parameter combinations on cached, independently prepared folds."""
    scorer = get_scorer(scoring)
    rows = []
    for parameters in ParameterGrid(parameter_grid):
        train_scores = []
        scores = []
        for fold in folds:
            fitted = clone(estimator).set_params(**parameters).fit(
                fold["X_fit"], fold["y_fit"]
            )
            train_scores.append(scorer(fitted, fold["X_fit"], fold["y_fit"]))
            scores.append(scorer(fitted, fold["X_score"], fold["y_score"]))
        train_scores = np.asarray(train_scores, dtype=float)
        scores = np.asarray(scores, dtype=float)
        rows.append({
            "params": parameters,
            "train_scores": train_scores,
            "scores": scores,
            "mean_train_score": train_scores.mean(),
            "mean_score": scores.mean(),
        })
    return sorted(rows, key=lambda row: row["mean_score"], reverse=True)

class CachedSearchCV(BaseEstimator):
    """Search candidates, preparing each preprocessing choice once per fold.

    The saving is in the cache: candidates that share a preprocessing
    configuration reuse one fitted preprocessor per fold, however many estimator
    settings they try. `estimator` must be a `PreparedEstimator` and `scoring`
    one metric name.

    Three ways to propose candidates:

    * exhaustive, the default, over `param_grid` as `ParameterGrid` reads it;
    * random, when `n_iter` is set, over `param_grid` as `ParameterSampler`
      reads it, so values may be lists or scipy distributions;
    * anything sequential, when `strategy` is given: an object with
      `ask(n) -> list of parameter dicts` and `tell(params, score)`, called
      until `ask` returns nothing. Scores are means over the folds, higher being
      better, which matches an Optuna study asked to maximise.

    `cv` takes any sklearn splitter, an integer, or an iterable of index pairs.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        cv=None,
        refit=True,
        n_jobs=4,
        return_train_score=False,
        n_iter=None,
        random_state=None,
        strategy=None,
    ):
        """Store the estimator, the candidate source, and the fold design."""
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score
        self.n_iter = n_iter
        self.random_state = random_state
        self.strategy = strategy

    def _candidate_batches(self):
        """Yield lists of parameter dicts until the candidate source is spent."""
        if self.strategy is not None:
            while True:
                batch = [dict(candidate)
                         for candidate in self.strategy.ask(self.n_jobs)]
                if not batch:
                    return
                yield batch
        elif self.n_iter is not None:
            yield list(ParameterSampler(
                self.param_grid, self.n_iter, random_state=self.random_state
            ))
        else:
            yield list(ParameterGrid(self.param_grid))

    def __sklearn_tags__(self):
        """Preserve the estimator type through the search wrapper."""
        return copy.deepcopy(get_tags(self.estimator))

    def fit(self, X, y):
        """Search cached folds and optionally refit the best configuration."""
        if not isinstance(self.estimator, PreparedEstimator):
            raise TypeError("CachedSearchCV requires a PreparedEstimator")
        if not isinstance(self.scoring, str):
            raise TypeError("CachedSearchCV requires one named scoring metric")

        splitter = check_cv(self.cv, y=y, classifier=is_classifier(self.estimator.estimator))
        cache = FoldCache(splitter, X, y)
        rows = []
        for parameter_sets in self._candidate_batches():
            configured = [
                clone(self.estimator).set_params(**parameters)
                for parameters in parameter_sets
            ]
            fold_sets = [cache.get(model.preprocessor) for model in configured]
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(candidate_scores)(
                    model.estimator, {}, folds, scoring=self.scoring
                )
                for model, folds in zip(configured, fold_sets)
            )
            batch = [
                {**result[0], "params": parameters}
                for result, parameters in zip(results, parameter_sets)
            ]
            if self.strategy is not None:
                for row in batch:
                    self.strategy.tell(row["params"], row["mean_score"])
            rows.extend(batch)
        if not rows:
            raise ValueError("the candidate source proposed nothing to score")

        means = np.asarray([row["mean_score"] for row in rows])
        ranks = np.asarray([1 + np.sum(means > value) for value in means])
        self.cv_results_ = {
            "params": [row["params"] for row in rows],
            "mean_test_score": means,
            "std_test_score": np.asarray([row["scores"].std(ddof=0) for row in rows]),
            "rank_test_score": ranks,
        }
        if self.return_train_score:
            self.cv_results_["mean_train_score"] = np.asarray(
                [row["mean_train_score"] for row in rows]
            )
        for split_index in range(len(rows[0]["scores"])):
            self.cv_results_[f"split{split_index}_test_score"] = np.asarray(
                [row["scores"][split_index] for row in rows]
            )
            if self.return_train_score:
                self.cv_results_[f"split{split_index}_train_score"] = np.asarray(
                    [row["train_scores"][split_index] for row in rows]
                )
        for parameter in sorted({key for row in rows for key in row["params"]}):
            self.cv_results_[f"param_{parameter}"] = np.asarray(
                [row["params"].get(parameter) for row in rows], dtype=object
            )

        self.best_index_ = int(np.argmax(self.cv_results_["mean_test_score"]))
        self.best_params_ = self.cv_results_["params"][self.best_index_]
        self.best_score_ = float(self.cv_results_["mean_test_score"][self.best_index_])
        self.n_splits_ = len(rows[0]["scores"])
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(
                **self.best_params_
            ).fit(X, y)
            if hasattr(self.best_estimator_, "classes_"):
                self.classes_ = self.best_estimator_.classes_
        return self

    def predict(self, X):
        """Predict with the refitted best estimator."""
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities with the refitted best estimator."""
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):
        """Return decision scores from the refitted best estimator."""
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.decision_function(X)

    def score(self, X, y):
        """Score the refitted best estimator with the configured metric."""
        check_is_fitted(self, "best_estimator_")
        return get_scorer(self.scoring)(self.best_estimator_, X, y)


# -------------------------------------------------------------------------
# Fitting several estimators, and stacking them
# -------------------------------------------------------------------------

def fit_estimators(preprocessor, estimators, X, y):
    """Fit one preprocessor, then every named estimator on its shared matrix."""
    fitted_preprocessor = clone(preprocessor).fit(X, y)
    transformed = fitted_preprocessor.transform(X)
    fitted_estimators = [
        (name, clone(estimator).fit(transformed, y))
        for name, estimator in estimators
    ]
    return fitted_preprocessor, fitted_estimators

def estimator_predictions(preprocessor, estimators, X, *, method="predict"):
    """Return one prediction column per fitted named estimator."""
    transformed = preprocessor.transform(X)
    columns = []
    for _, estimator in estimators:
        predict = getattr(estimator, method)
        values = predict(transformed)
        if method == "predict_proba":
            values = values[:, 1]
        columns.append(np.asarray(values))
    return np.column_stack(columns)

def out_of_fold_predictions(
    preprocessor,
    estimators,
    splitter,
    X,
    y,
    *,
    method="predict",
):
    """Build predictions where each row is held out exactly once.

    Repeated and partial splitters do not define the complete training matrix
    required by a stack, so reject them before fitting any member.
    """
    splits = list(splitter.split(X, y))
    coverage = np.zeros(len(X), dtype=int)
    for fit_at, score_at in splits:
        if np.intersect1d(fit_at, score_at).size:
            raise ValueError("out-of-fold fit and score rows must be disjoint")
        np.add.at(coverage, score_at, 1)
    if not np.all(coverage == 1):
        raise ValueError("out-of-fold predictions require each row held out exactly once")
    predictions = np.empty((len(X), len(estimators)), dtype=float)
    for fold in prepared_folds(preprocessor, splits, X, y):
        for column, (_, estimator) in enumerate(estimators):
            fitted = clone(estimator).fit(fold["X_fit"], fold["y_fit"])
            predict = getattr(fitted, method)
            values = predict(fold["X_score"])
            if method == "predict_proba":
                values = values[:, 1]
            predictions[fold["score_at"], column] = values
    return predictions

class FittedStack:
    """A fitted preprocessor, member estimators and meta-estimator."""

    def __init__(self, preprocessor, estimators, meta_estimator, *, method="predict"):
        """Store fitted stack components and their prediction method."""
        self.preprocessor = preprocessor
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.method = method

    def predict(self, X):
        """Predict from member outputs transformed by the fitted stack."""
        features = estimator_predictions(
            self.preprocessor,
            self.estimators,
            X,
            method=self.method,
        )
        return self.meta_estimator.predict(features)

def fit_stacked(preprocessor, estimators, meta_estimator, splitter, X, y, *, method="predict"):
    """Fit a stack from fold-local shared preprocessing and out-of-fold features."""
    features = out_of_fold_predictions(
        preprocessor, estimators, splitter, X, y, method=method
    )
    fitted_meta = clone(meta_estimator).fit(features, y)
    fitted_preprocessor, fitted_estimators = fit_estimators(
        preprocessor, estimators, X, y
    )
    return FittedStack(
        fitted_preprocessor,
        fitted_estimators,
        fitted_meta,
        method=method,
    )
