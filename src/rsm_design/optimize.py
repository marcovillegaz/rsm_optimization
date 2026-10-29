"""Optimum of a fitted response surface inside the experimental region.

The model of a central composite design (CCD) is only valid in the region
circumscribed by the experimental points, so the optimum is searched inside it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rsm_design.model import n_factors_from_terms, predict, quadratic_form

logger = logging.getLogger(__name__)

########################################################################
# DESIGN REGION
########################################################################


@dataclass(frozen=True)
class DesignRegion:
    """Region of the factor space where the model is valid.

    For a CCD the experimental points (factorial, axial and centre points) lie
    on an ellipsoid around the replicated centre point, and the fitted model
    is only trustworthy inside it: outside there are no data, so the quadratic
    extrapolates freely and its optimum is an artefact. The default shape is
    therefore the ellipsoid; "box" only keeps the factor ranges.

    params:
        factors (tuple): factor names, in the order of the coordinates
        center (np.ndarray): centre of the region (centre point of the design)
        semi_axes (np.ndarray): semi-axis of the ellipsoid along each factor
        lower (np.ndarray): minimum experimental value of each factor
        upper (np.ndarray): maximum experimental value of each factor
        shape (str): "ellipsoid" or "box"
    """

    factors: tuple[str, ...]
    center: np.ndarray
    semi_axes: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    shape: Literal["ellipsoid", "box"] = "ellipsoid"

    @classmethod
    def from_data(
        cls,
        df: pd.DataFrame,
        factors: Sequence[str],
        center: Literal["mode", "midrange"] = "mode",
        shape: Literal["ellipsoid", "box"] = "ellipsoid",
    ) -> DesignRegion:
        """Build the region from the experimental factor values.

        The centre is the per-column mode, i.e. the replicated centre point of
        a CCD. If no value repeats in a column (no replicates) the mode is not
        a centre and the midrange is used instead, with a warning.

        params:
            df (pd.DataFrame): experiments, one row per run
            factors (list): names of the factor columns
            center (str): "mode" or "midrange"
            shape (str): "ellipsoid" or "box"
        return:
            region (DesignRegion): centre, semi-axes and bounds of the design
        """
        if center not in ("mode", "midrange"):
            raise ValueError(f"center must be 'mode' or 'midrange', not {center!r}")
        if shape not in ("ellipsoid", "box"):
            raise ValueError(f"shape must be 'ellipsoid' or 'box', not {shape!r}")

        factors = tuple(factors)
        data_df = df[list(factors)].astype(float)
        lower = data_df.min().to_numpy()
        upper = data_df.max().to_numpy()
        midrange = (lower + upper) / 2

        if center == "mode":
            centre_point = midrange.copy()
            for i, name in enumerate(factors):
                counts = data_df[name].value_counts()
                if counts.iloc[0] > 1:
                    centre_point[i] = float(counts.index[0])
                else:
                    logger.warning(
                        "No repeated value in '%s'; using the midrange as centre", name
                    )
        else:
            centre_point = midrange

        semi_axes = upper - centre_point  # Distance centre to the highest level
        if np.any(semi_axes <= 0):
            raise ValueError("A factor has no variation above the centre point")

        return cls(
            factors=factors,
            center=centre_point,
            semi_axes=semi_axes,
            lower=lower,
            upper=upper,
            shape=shape,
        )

    def ellipsoid_norm(self, x: Sequence[float]) -> float:
        """Normalised squared distance to the centre: <= 1 inside the ellipsoid.

        params:
            x (array): point in factor space
        return:
            norm (float): sum(((x - center) / semi_axes)^2)
        """
        z = (np.asarray(x, dtype=float) - self.center) / self.semi_axes
        return float(np.sum(z**2))

    def contains(self, x: Sequence[float], tol: float = 1e-9) -> bool:
        """True if x is inside the region (ellipsoid and bounds, or bounds only).

        params:
            x (array): point in factor space
            tol (float): numerical tolerance
        return:
            inside (bool)
        """
        x = np.asarray(x, dtype=float)
        in_bounds = bool(
            np.all(x >= self.lower - tol) and np.all(x <= self.upper + tol)
        )
        if self.shape == "box":
            return in_bounds
        return in_bounds and self.ellipsoid_norm(x) <= 1 + tol


########################################################################
# OPTIMIZATION
########################################################################


@dataclass(frozen=True)
class OptimizationResult:
    """Optimum found by optimize.

    params:
        x (np.ndarray): optimal factor values
        factors (dict): factor name -> optimal value
        response (float): predicted response at x
        success (bool): the solver converged
        message (str): solver message
        inside_region (bool): x satisfies the region constraints
    """

    x: np.ndarray
    factors: dict[str, float]
    response: float
    success: bool
    message: str
    inside_region: bool


def _starting_points(region: DesignRegion, n_starts: int | None) -> list[np.ndarray]:
    """Centre plus points at +-0.5 semi-axis along each axis (all feasible)."""
    starts = [region.center.copy()]
    for i in range(len(region.factors)):
        for sign in (1.0, -1.0):
            point = region.center.copy()
            point[i] += sign * 0.5 * region.semi_axes[i]
            starts.append(point)
    if n_starts is not None:
        starts = starts[: max(n_starts, 1)]
    return starts


def optimize(
    coefficients: pd.Series | Sequence[float],
    region: DesignRegion,
    maximize: bool = True,
    n_starts: int | None = None,
) -> OptimizationResult:
    """Optimum of the response surface inside the design region.

    The optimum of a CCD model must stay inside the ellipsoid of the
    experimental points: the model is only valid in the region circumscribed
    by the design, and outside it the quadratic is an extrapolation.

    The old code enforced this by returning 1e6 when the point left the
    ellipsoid. That objective is discontinuous (a cliff, zero gradient on both
    sides), which gradient methods cannot handle: L-BFGS-B stalls at the
    starting point or at the cliff. Here the ellipsoid is a real inequality
    constraint, 1 - sum(((x - c) / a)^2) >= 0, solved with SLSQP together with
    the factor bounds and the analytic gradient b + 2Bx. Several starting
    points are tried and the best feasible result is kept, since the surface
    can be a saddle or have several local optima.

    params:
        coefficients (array): full coefficient vector in the term_names order
        region (DesignRegion): where the model is valid
        maximize (bool): maximize (True) or minimize (False) the response
        n_starts (int): number of starting points to use (default: all, i.e.
            the centre and +-0.5 semi-axis along each factor)
    return:
        result (OptimizationResult): optimal factors and predicted response
    """
    coefficients = np.asarray(coefficients, dtype=float)
    n_factors = n_factors_from_terms(coefficients.size)
    if n_factors != len(region.factors):
        raise ValueError(
            f"The coefficients are for {n_factors} factors but the region has "
            f"{len(region.factors)}"
        )

    b0, b, B = quadratic_form(coefficients)
    sign = -1.0 if maximize else 1.0  # SLSQP minimizes

    def objective(x: np.ndarray) -> float:
        return sign * (b0 + b @ x + x @ B @ x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return sign * (b + 2 * B @ x)

    constraints = []
    if region.shape == "ellipsoid":
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: 1 - region.ellipsoid_norm(x),
                "jac": lambda x: -2 * (x - region.center) / region.semi_axes**2,
            }
        )
    bounds = list(zip(region.lower, region.upper, strict=True))

    logger.info("Optimizing the response surface...")
    best = None
    for x0 in _starting_points(region, n_starts):
        res = minimize(
            objective,
            x0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        if not region.contains(res.x, tol=1e-6):
            continue  # Discard solutions that leave the valid region
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("No feasible optimum found inside the design region")

    x_opt = np.asarray(best.x, dtype=float)
    response = float(predict(x_opt, coefficients)[0])
    logger.info("Optimum of the response surface:")
    for name, value in zip(region.factors, x_opt, strict=True):
        logger.info("\t%s: %s", name, value)
    logger.info("\tResponse: %s", response)

    return OptimizationResult(
        x=x_opt,
        factors=dict(zip(region.factors, map(float, x_opt), strict=True)),
        response=response,
        success=bool(best.success),
        message=str(best.message),
        inside_region=region.contains(x_opt, tol=1e-6),
    )


def stationary_point(
    coefficients: pd.Series | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary point of the quadratic and the eigenvalues of B (canonical analysis).

    x_s = -0.5 B^-1 b. All eigenvalues negative: maximum; all positive:
    minimum; mixed signs: saddle point.

    params:
        coefficients (array): full coefficient vector in the term_names order
    return:
        x_s (np.ndarray): stationary point
        eigenvalues (np.ndarray): eigenvalues of B, ascending
    """
    _, b, B = quadratic_form(coefficients)
    x_s = -0.5 * np.linalg.solve(B, b)
    return x_s, np.linalg.eigvalsh(B)
