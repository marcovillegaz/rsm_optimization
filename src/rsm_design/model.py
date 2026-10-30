"""Quadratic response-surface model: term naming, design matrix, prediction and
fitting. The model is valid for any number of factors:

    y = b0 + sum(bi*xi) + sum(bij*xi*xj, i <= j)
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

DPI = 600


class RankDeficientDesignError(ValueError):
    """The experimental design cannot estimate all the requested model terms."""


########################################################################
# TERM NAMES AND QUADRATIC MODEL (shared core)
########################################################################


def term_names(n_factors: int, prefix: str = "b") -> list[str]:
    """Names of the terms of the full quadratic model.

    The order is the one of sklearn PolynomialFeatures(degree=2): the intercept,
    the linear terms, and then the pairs (i <= j) as squares and interactions.
    For 3 factors: b0, b1, b2, b3, b11, b12, b13, b22, b23, b33.

    params:
        n_factors (int): number of factors of the design
        prefix (str): letter used to name the coefficients
    return:
        names (list): term names; pairs use "_" when there are 10 or more factors
    """
    sep = "_" if n_factors >= 10 else ""
    names = [f"{prefix}0"]
    names += [f"{prefix}{i}" for i in range(1, n_factors + 1)]
    for i, j in combinations_with_replacement(range(1, n_factors + 1), 2):
        names.append(f"{prefix}{i}{sep}{j}")
    return names


def n_terms(n_factors: int) -> int:
    """Number of terms of the full quadratic model, (N + 1)(N + 2) / 2."""
    return (n_factors + 1) * (n_factors + 2) // 2


def n_factors_from_terms(n: int) -> int:
    """Number of factors that gives a full quadratic model with n terms.

    Raises ValueError if n is not a valid number of terms.
    """
    for n_factors in range(1, n + 1):
        if n_terms(n_factors) == n:
            return n_factors
    raise ValueError(f"{n} is not the number of terms of a quadratic model")


def design_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Columns of the full quadratic model for each experiment.

    params:
        X (array): factors, one row per experiment and one column per factor
    return:
        D (np.ndarray): [1, x_i, x_i * x_j (i <= j)] in the term_names order
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_factors = X.shape[1]

    columns = [np.ones(X.shape[0])]
    columns += [X[:, i] for i in range(n_factors)]
    for i, j in combinations_with_replacement(range(n_factors), 2):
        columns.append(X[:, i] * X[:, j])
    return np.column_stack(columns)


def predict(
    X: pd.DataFrame | np.ndarray, coefficients: pd.Series | Sequence[float]
) -> np.ndarray:
    """Evaluate the response surface for any number of factors.

    params:
        X (array): factors, one row per point (a 1-D input is one point)
        coefficients (array): full coefficient vector in the term_names order
    return:
        y (np.ndarray): predicted response, one value per row of X
    """
    coefficients = np.asarray(coefficients, dtype=float)
    D = design_matrix(X)
    if D.shape[1] != coefficients.size:
        raise ValueError(
            f"Got {coefficients.size} coefficients but a model with "
            f"{np.atleast_2d(X).shape[1]} factors has {D.shape[1]} terms"
        )
    return D @ coefficients


def quadratic_form(
    coefficients: pd.Series | Sequence[float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Write the model as y = b0 + b.x + x'Bx.

    B is symmetric: squares on the diagonal and half of each interaction
    coefficient in the off-diagonal entries. The optimizer uses it to get the
    analytic gradient (b + 2Bx) and the stationary point.

    params:
        coefficients (array): full coefficient vector in the term_names order
    return:
        b0 (float), b (np.ndarray, shape n), B (np.ndarray, shape n x n)
    """
    coefficients = np.asarray(coefficients, dtype=float)
    n_factors = n_factors_from_terms(coefficients.size)

    b0 = float(coefficients[0])
    b = coefficients[1 : n_factors + 1].copy()
    B = np.zeros((n_factors, n_factors))
    pairs = combinations_with_replacement(range(n_factors), 2)
    for k, (i, j) in enumerate(pairs, start=n_factors + 1):
        if i == j:
            B[i, i] = coefficients[k]
        else:
            B[i, j] = B[j, i] = coefficients[k] / 2
    return b0, b, B


########################################################################
# FITTING (implemented by the core stream)
########################################################################


@dataclass(frozen=True)
class FitResult:
    """Everything produced by fit_model.

    params:
        response (str): name of the response
        factors (tuple): factor names, in the order of the design matrix
        terms (tuple): names of ALL the terms of the full model
        dropped_terms (tuple): terms removed from the model
        ols (Any): statsmodels RegressionResults of the fitted model
        coefficients (pd.Series): full vector indexed by terms; dropped terms = 0.0
        X (pd.DataFrame): design matrix columns kept in the fit
        y (pd.Series): experimental response
    """

    response: str
    factors: tuple[str, ...]
    terms: tuple[str, ...]
    dropped_terms: tuple[str, ...]
    ols: Any
    coefficients: pd.Series
    X: pd.DataFrame
    y: pd.Series

    @property
    def fitted(self) -> pd.Series:
        """Predicted response for the experimental points."""
        return pd.Series(self.ols.fittedvalues, index=self.y.index)

    @property
    def residuals(self) -> pd.Series:
        """Experimental minus predicted response."""
        return self.y - self.fitted

    @property
    def r_squared(self) -> float:
        return float(self.ols.rsquared)

    @property
    def adj_r_squared(self) -> float:
        return float(self.ols.rsquared_adj)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))


def file_stem(response: str) -> str:
    """File-name stem of a response: only letters, digits, '-' and '_'.

    Spaces are removed so every output of one response shares a name such as
    EnrichmentFactor_model_params.csv.
    """
    stem = re.sub(r"[^0-9A-Za-z_-]", "", str(response))
    return stem if stem else "response"


def check_estimability(X_design: np.ndarray, terms: Sequence[str]) -> list[str]:
    """Terms that do not increase the rank of the design matrix.

    Columns are added one at a time; a term whose column is a linear
    combination of the previous ones cannot be estimated by the design.
    For example, a duplicated 2^2 factorial has only 4 distinct points, so
    b11 and b22 are aliased with b0 and cannot be separated from it.

    params:
        X_design (np.ndarray): design matrix columns, one per term
        terms (list): names of the columns of X_design
    return:
        non_estimable (list): names of the terms that are linear combinations
            of the previous ones (empty if the design is full rank)
    """
    X_design = np.asarray(X_design, dtype=float)
    if X_design.shape[1] != len(terms):
        raise ValueError("X_design must have one column per term")

    non_estimable = []
    kept = []  # Columns that increased the rank
    rank = 0
    for k, term in enumerate(terms):
        trial = kept + [k]
        trial_rank = np.linalg.matrix_rank(X_design[:, trial])
        if trial_rank > rank:
            kept = trial
            rank = trial_rank
        else:
            non_estimable.append(term)
    return non_estimable


def fit_model(
    df: pd.DataFrame,
    factors: Sequence[str],
    response: str,
    drop_terms: Sequence[str] = (),
    on_rank_deficient: Literal["raise", "warn"] = "raise",
) -> FitResult:
    """Fit the quadratic model by ordinary least squares (statsmodels OLS).

    OLS from statsmodels is used because it gives the full summary and ANOVA
    tables needed to decide which terms to drop. The rank of the kept columns
    is checked first: with replicated or too few distinct points some
    coefficients are not identifiable and the fit would be meaningless.

    params:
        df (pd.DataFrame): experiments, one row each, with numeric columns
        factors (list): names of the factor columns
        response (str): name of the response column
        drop_terms (list): terms removed from the model, e.g. ["b12", "b22"]
        on_rank_deficient (str): "raise" or "warn" if the kept terms cannot
            be estimated by the design
    return:
        result (FitResult): fitted model; dropped terms have coefficient 0.0
    """
    if on_rank_deficient not in ("raise", "warn"):
        raise ValueError("on_rank_deficient must be 'raise' or 'warn'")

    factors = list(factors)
    for column in factors + [response]:
        if column not in df.columns:
            raise KeyError(
                f"Column {column!r} not in data. Available: {list(df.columns)}"
            )

    terms = term_names(len(factors))
    drop_terms = list(drop_terms)
    unknown = [t for t in drop_terms if t not in terms]
    if unknown:
        raise ValueError(f"Unknown terms {unknown}. Valid terms are {terms}")

    logger.info(f"Fitting model to {response} response.")

    # DESIGN MATRIX OF THE KEPT TERMS
    D = design_matrix(df[factors])
    X_df = pd.DataFrame(D, columns=terms, index=df.index)
    kept_terms = [t for t in terms if t not in drop_terms]
    X_df = X_df[kept_terms]
    y = df[response].astype(float)

    # RANK CHECK
    non_estimable = check_estimability(X_design=X_df.to_numpy(), terms=kept_terms)
    if non_estimable:
        message = (
            f"The design cannot estimate the terms {non_estimable} of the "
            f"{response} model (rank {X_df.shape[1] - len(non_estimable)} of "
            f"{X_df.shape[1]} terms). Remove them with drop_terms=..."
        )
        if on_rank_deficient == "raise":
            raise RankDeficientDesignError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    # OLS FIT
    ols = sm.OLS(y, X_df).fit()
    coefficients = pd.Series(0.0, index=terms, name=response)
    coefficients[kept_terms] = ols.params
    logger.info(
        f"    R2: {ols.rsquared:.4f}, adjusted R2: {ols.rsquared_adj:.4f}, "
        f"{int(ols.nobs)} experiments, {len(kept_terms)} terms."
    )

    return FitResult(
        response=response,
        factors=tuple(factors),
        terms=tuple(terms),
        dropped_terms=tuple(drop_terms),
        ols=ols,
        coefficients=coefficients,
        X=X_df,
        y=y,
    )


def save_results(
    result: FitResult, folder: str | Path, plots: bool = True, dpi: int = DPI
) -> dict[str, Path]:
    """Write {stem}_model_params.csv, {stem}_model_stats.txt and the plots.

    The parameters file is the headerless name,value table that the optimizer
    and the surface plots read back with load_coefficients.

    params:
        result (FitResult): output of fit_model
        folder (str | Path): results folder, created if missing
        plots (bool): also save the diagnostic plots
        dpi (int): resolution of the plots
    return:
        paths (dict): "params", "stats" and, with plots, "plots" (list of paths)
    """
    from rsm_design.data import save_coefficients

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = file_stem(result.response)
    logger.info(f"Saving results of {result.response} in {folder}.")

    paths: dict[str, Any] = {}

    # MODEL COEFFICIENTS
    paths["params"] = folder / f"{stem}_model_params.csv"
    logger.info("    Saving model coefficients.")
    save_coefficients(coefficients=result.coefficients, path=paths["params"])

    # STATISTICAL SUMMARY
    paths["stats"] = folder / f"{stem}_model_stats.txt"
    logger.info("    Saving model statistics.")
    text = result.ols.summary().as_text()
    dropped = ", ".join(result.dropped_terms) if result.dropped_terms else "none"
    text += "\n\nDropped terms: " + dropped + "\n"
    paths["stats"].write_text(text, encoding="utf-8")

    # PLOTS (imported here: plots.py is only needed when plotting)
    if plots:
        from rsm_design.plots import plot_diagnostics

        paths["plots"] = plot_diagnostics(result, folder, dpi=dpi)

    return paths
