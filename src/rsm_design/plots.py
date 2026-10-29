"""Plots of a fitted response surface: diagnostics, surface and contour."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm, t

from rsm_design.model import DPI, FitResult, predict

logger = logging.getLogger(__name__)


########################################################################
# HELPERS
########################################################################


def _stem(response: str) -> str:
    """File-name stem of a response: only letters, digits, '-' and '_'.

    params:
        response (str): name of the response, e.g. "Enrichment Factor"
    return:
        stem (str): e.g. "EnrichmentFactor"
    """
    return re.sub(r"[^A-Za-z0-9_-]", "", response)


def _finish(
    fig: Figure, folder: str | Path | None, file_name: str, dpi: int = DPI
) -> Path | None:
    """Save the figure if there is a folder, else show it; always close it.

    Closing the figure avoids the memory leak (and the matplotlib warning)
    of generating many plots in one run.

    params:
        fig (Figure): figure to finish
        folder (str | Path | None): output folder (created if missing); None shows
        file_name (str): name of the PNG file
        dpi (int): resolution of the saved image
    return:
        path (Path | None): saved file, None when the figure was only shown
    """
    path = None
    try:
        if folder is not None:
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / file_name
            logger.info("\tSaving %s", file_name)
            fig.savefig(path, dpi=dpi)
        else:
            plt.show()
    finally:
        plt.close(fig)
    return path


########################################################################
# DIAGNOSTIC PLOTS
########################################################################


def coeff_plot(
    result: FitResult,
    whiskers: Literal["ci", "stderr"] = "ci",
    normalize: bool = False,
    alpha: float = 0.05,
    folder: str | Path | None = None,
    dpi: int = DPI,
) -> Path | None:
    """Coefficient plot with whiskers. If the whiskers cross the zero line the
    coefficient is not significantly different from zero.

    When normalized, the t-values (coefficient / standard error) are plotted
    with lines at +-t_crit, so a coefficient is significant when its marker is
    outside them. This keeps the reference lines consistent with the quantity
    shown (the legacy plot kept standard-error bars on t-values).

    params:
        result (FitResult): fitted model
        whiskers (str): "ci" for confidence intervals, "stderr" for standard error
        normalize (bool): plot t-values and the +-t_crit limits instead
        alpha (float): significance level of the confidence interval and t_crit
        folder (Path | None): output folder; None shows the figure
        dpi (int): resolution of the saved image
    return:
        path (Path | None): saved file, None when only shown
    """
    logger.info("coeff_plot()")
    if whiskers not in ("ci", "stderr"):
        raise ValueError(f"whiskers must be 'ci' or 'stderr', got {whiskers!r}")

    # PROCESSING DATA
    coeff = np.asarray(result.ols.params, dtype=float)
    coeff_names = list(result.ols.params.index)
    stderr = np.asarray(result.ols.bse, dtype=float)
    t_crit = float(t.ppf(1 - alpha / 2, result.ols.df_resid))

    if normalize:
        values = coeff / stderr  # t-values
        error = None
        ylabel = "t-value"
    elif whiskers == "ci":
        ci = np.asarray(result.ols.conf_int(alpha=alpha), dtype=float)
        values = coeff
        error = np.array([coeff - ci[:, 0], ci[:, 1] - coeff])
        ylabel = f"Coefficients ({100 * (1 - alpha):.0f}% CI)"
    else:
        values = coeff
        error = stderr
        ylabel = "Coefficients (standard error)"

    # MAKING PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(values))
    ax.errorbar(positions, values, yerr=error, fmt="s", capsize=5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    if normalize:
        ax.axhline(t_crit, color="gray", linestyle=":", linewidth=1)
        ax.axhline(-t_crit, color="gray", linestyle=":", linewidth=1)
        logger.info(
            "\tt_crit (alpha=%g, df=%g): %.3f", alpha, result.ols.df_resid, t_crit
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(coeff_names)
    ax.set_xlabel("Variables")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    # SAVE OR SHOW
    return _finish(fig, folder, f"{_stem(result.response)}_coeff_plot.png", dpi)


def residual_plot(
    result: FitResult,
    normalize: bool = True,
    bins: int = 10,
    folder: str | Path | None = None,
    dpi: int = DPI,
) -> Path | None:
    """Residuals vs predicted response, with histogram and normal curve.

    The residuals of a good model look like noise around zero and their
    histogram follows the normal curve. The curve is scaled by the standard
    deviation of the residuals, so it is also right when they are not normalized.

    params:
        result (FitResult): fitted model
        normalize (bool): divide the residuals by their standard deviation
        bins (int): number of bars of the histogram
        folder (Path | None): output folder; None shows the figure
        dpi (int): resolution of the saved image
    return:
        path (Path | None): saved file, None when only shown
    """
    logger.info("residual_plot()")
    # PROCESSING DATA
    y_pred = result.fitted.to_numpy(dtype=float)
    resd = result.residuals.to_numpy(dtype=float)  # Residual
    logger.info("\tRoot Mean Squared Error (RMSE): %s", result.rmse)

    std = float(np.std(resd))
    if normalize and std > 0:
        resd = resd / std
        std = 1.0
        ylabel = "Normalized residuals"
    else:
        ylabel = "Residuals"

    # MAKING PLOT
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 1))

    ## Residual plot
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(y_pred, resd)
    ax.axhline(y=0)
    ax.set_xlabel("y predicted")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    ## Normal distribution with histogram
    n_points = len(resd)
    ax_marg = fig.add_subplot(gs[0, 1], sharey=ax)
    counts, edges, _ = ax_marg.hist(x=resd, bins=bins, orientation="horizontal")
    bin_width = edges[1] - edges[0]
    if std > 0:
        y = np.linspace(edges[0], edges[-1], 1000)
        ax_marg.plot(norm.pdf(y, loc=0.0, scale=std) * n_points * bin_width, y)
    ax_marg.tick_params(labelleft=False)

    # SAVE OR SHOW
    return _finish(fig, folder, f"{_stem(result.response)}_rsd_plot.png", dpi)


def prediction_plot(
    result: FitResult, folder: str | Path | None = None, dpi: int = DPI
) -> Path | None:
    """Experimental vs predicted response; the points should follow y = x.

    params:
        result (FitResult): fitted model
        folder (Path | None): output folder; None shows the figure
        dpi (int): resolution of the saved image
    return:
        path (Path | None): saved file, None when only shown
    """
    logger.info("prediction_plot()")
    # PROCESSING DATA
    y_exp = result.y.to_numpy(dtype=float)
    y_pred = result.fitted.to_numpy(dtype=float)
    low = min(y_exp.min(), y_pred.min())
    high = max(y_exp.max(), y_pred.max())
    margin = 0.05 * (high - low) if high > low else 1.0
    limits = (low - margin, high + margin)

    # MAKING PLOT
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x=y_pred, y=y_exp)
    ax.plot(limits, limits, color="gray", linestyle="--")  # y = x in data coordinates
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("Experimental")
    ax.set_xlabel("Predicted")
    ax.grid(True)

    # SAVE OR SHOW
    return _finish(fig, folder, f"{_stem(result.response)}_prediction_plot.png", dpi)


def plot_diagnostics(
    result: FitResult, folder: str | Path | None, dpi: int = DPI
) -> list[Path]:
    """The three diagnostic plots: coefficients, residuals and prediction.

    params:
        result (FitResult): fitted model
        folder (Path): output folder
        dpi (int): resolution of the saved images
    return:
        paths (list[Path]): the saved files
    """
    paths = [
        coeff_plot(result=result, folder=folder, dpi=dpi),
        residual_plot(result=result, folder=folder, dpi=dpi),
        prediction_plot(result=result, folder=folder, dpi=dpi),
    ]
    return [path for path in paths if path is not None]


########################################################################
# RESPONSE SURFACE
########################################################################


def _centre(data: pd.DataFrame | None, factor: str) -> float:
    """Centre point of a factor: the mode of the data (the replicated centre
    point of a CCD), or the midrange when no value repeats."""
    if data is None:
        raise ValueError(f"No data to take the fixed value of {factor!r}; pass fixed")
    values = data[factor].dropna()
    mode = values.mode()
    if len(mode) >= 1 and (values == mode.iloc[0]).sum() > 1:
        return float(mode.iloc[0])
    return float((values.min() + values.max()) / 2)


def _resolve_grid_inputs(
    factors: Sequence[str],
    vary: tuple[str, str],
    data: pd.DataFrame | None,
    fixed: Mapping[str, float] | None,
    limits: Mapping[str, tuple[float, float]] | None,
    extend: float,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Fixed values and axis limits with their defaults filled in."""
    for name in vary:
        if name not in factors:
            raise ValueError(f"Factor {name!r} not in {list(factors)}")
    if vary[0] == vary[1]:
        raise ValueError("vary needs two different factors")

    fixed = dict(fixed or {})
    limits = dict(limits or {})
    for factor in factors:
        if factor in vary:
            if factor not in limits:
                if data is None:
                    raise ValueError(f"No data to take the limits of {factor!r}")
                lo, hi = float(data[factor].min()), float(data[factor].max())
                dif = hi - lo
                limits[factor] = (lo - extend * dif, hi + extend * dif)
        elif factor not in fixed:
            fixed[factor] = _centre(data, factor)
    return fixed, limits


def response_grid(
    coefficients: pd.Series | Sequence[float],
    factors: Sequence[str],
    vary: tuple[str, str],
    fixed: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
    grid: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the response on a 2-D grid of two factors; the rest are constant.

    The grid is built with the two varied factors only and the fixed factors
    enter as constant columns, so it works for any number of factors.

    params:
        coefficients (array): full coefficient vector in the term_names order
        factors (list): factor names, in the order of the design matrix
        vary (tuple): the two factors on the axes (x, y)
        fixed (dict): factor -> constant value of every factor not in vary
        limits (dict): factor -> (low, high) of the two varied factors
        grid (int): points per axis
    return:
        X1, X2, Y (np.ndarray): arrays of shape (grid, grid); Y[i, j] is the
        response at (X1[i, j], X2[i, j])
    """
    axis_1 = np.linspace(*limits[vary[0]], grid)
    axis_2 = np.linspace(*limits[vary[1]], grid)
    X1, X2 = np.meshgrid(axis_1, axis_2)

    columns = []
    for factor in factors:
        if factor == vary[0]:
            columns.append(X1.ravel())
        elif factor == vary[1]:
            columns.append(X2.ravel())
        else:
            columns.append(np.full(X1.size, float(fixed[factor])))
    Y = predict(np.column_stack(columns), coefficients).reshape(X1.shape)
    return X1, X2, Y


def _labels(vary: tuple[str, str], labels: Mapping[str, str] | None) -> tuple[str, str]:
    """Axis labels of the varied factors (factor name if not in labels)."""
    labels = labels or {}
    return labels.get(vary[0], vary[0]), labels.get(vary[1], vary[1])


def _points_residual(
    data: pd.DataFrame,
    response: str,
    coefficients: pd.Series | Sequence[float],
    factors: Sequence[str],
) -> np.ndarray:
    """Experimental minus predicted response at the experimental points."""
    y_pred = predict(data[list(factors)].to_numpy(dtype=float), coefficients)
    return data[response].to_numpy(dtype=float) - y_pred


def _optimum_xy(
    optimum: Any, factors: Sequence[str], vary: tuple[str, str]
) -> tuple[float, float, float]:
    """Coordinates (x1, x2, response) of an optimum with .x and .response."""
    x = np.asarray(optimum.x, dtype=float)
    return (
        float(x[list(factors).index(vary[0])]),
        float(x[list(factors).index(vary[1])]),
        float(optimum.response),
    )


def _grid_for_plot(
    coefficients: pd.Series | Sequence[float],
    factors: Sequence[str],
    vary: tuple[str, str],
    data: pd.DataFrame | None,
    fixed: Mapping[str, float] | None,
    limits: Mapping[str, tuple[float, float]] | None,
    extend: float,
    grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve defaults, log them and build the grid shared by both plots."""
    fixed, limits = _resolve_grid_inputs(factors, vary, data, fixed, limits, extend)
    logger.info("\tVaried factors: %s, %s", *vary)
    for name, value in fixed.items():
        if name not in vary:
            logger.info("\tFixed %s = %g", name, value)
    return response_grid(coefficients, factors, vary, fixed, limits, grid)


def surface_plot(
    coefficients: pd.Series | Sequence[float],
    factors: Sequence[str],
    vary: tuple[str, str],
    data: pd.DataFrame | None = None,
    response: str | None = None,
    fixed: Mapping[str, float] | None = None,
    limits: Mapping[str, tuple[float, float]] | None = None,
    extend: float = 0.25,
    grid: int = 100,
    optimum: Any = None,
    labels: Mapping[str, str] | None = None,
    view: tuple[float, float] = (20, 45),
    cmap: str = "jet",
    folder: str | Path | None = None,
    file_name: str | None = None,
    dpi: int = DPI,
) -> Path | None:
    """3-D response surface of two factors with the experimental points.

    With more than two factors the others are held constant (by default at the
    centre point, the mode of the data). The points below the surface are drawn
    behind it and those above in front, so the depth is read correctly.

    params:
        coefficients (array): full coefficient vector in the term_names order
        factors (list): factor names, in the order of the design matrix
        vary (tuple): the two factors on the axes
        data (DataFrame): experimental data (factors + response); None = no points
        response (str): response column of data, also the z label
        fixed (dict): factor -> constant value; default the centre point of data
        limits (dict): factor -> (low, high); default data range plus extend
        extend (float): fraction of the data range the surface goes beyond it
        grid (int): points per axis
        optimum: object with .x (all the factors) and .response; drawn as a star
        labels (dict): factor -> axis label, e.g. to add units
        view (tuple): (elevation, azimuth) in degrees
        cmap (str): matplotlib colormap of the surface
        folder (Path | None): output folder; None shows the figure
        file_name (str): name of the PNG; default {response}_surface.png
        dpi (int): resolution of the saved image
    return:
        path (Path | None): saved file, None when only shown
    """
    logger.info("surface_plot()")
    # PROCESSING DATA
    X1, X2, Y = _grid_for_plot(
        coefficients, factors, vary, data, fixed, limits, extend, grid
    )
    xlabel, ylabel = _labels(vary, labels)

    # MAKING PLOT
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    fig.subplots_adjust(left=0.0, right=1, bottom=0.0, top=1.1)
    ax.plot_surface(X1, X2, Y, cmap=cmap, alpha=0.6, antialiased=True, zorder=5)

    ## Experimental points
    if data is not None and response is not None:
        resd = _points_residual(data, response, coefficients, factors)
        y_exp = data[response].to_numpy(dtype=float)
        below = resd < 0  # Point under the surface: drawn behind it
        for mask, zorder in ((below, 2), (~below, 10)):
            ax.scatter(
                xs=data.loc[mask, vary[0]],
                ys=data.loc[mask, vary[1]],
                zs=y_exp[mask],
                marker=".",
                c="red",
                zorder=zorder,
                depthshade=False,
            )

    ## Optimum
    if optimum is not None:
        x1, x2, z = _optimum_xy(optimum, factors, vary)
        ax.scatter(xs=x1, ys=x2, zs=z * 1.05, marker="*", c="green", zorder=1)

    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(response or "Response")

    # SAVE OR SHOW
    if file_name is None:
        file_name = f"{_stem(response or 'response')}_surface.png"
    return _finish(fig, folder, file_name, dpi)


def contour_plot(
    coefficients: pd.Series | Sequence[float],
    factors: Sequence[str],
    vary: tuple[str, str],
    data: pd.DataFrame | None = None,
    response: str | None = None,
    fixed: Mapping[str, float] | None = None,
    limits: Mapping[str, tuple[float, float]] | None = None,
    extend: float = 0.0,
    grid: int = 100,
    optimum: Any = None,
    labels: Mapping[str, str] | None = None,
    levels: int = 20,
    cmap: str = "summer",
    filled: bool = True,
    folder: str | Path | None = None,
    file_name: str | None = None,
    dpi: int = DPI,
) -> Path | None:
    """Contour map of the response as a function of two factors.

    Same inputs as surface_plot, but drawn as contours with a colour bar. By
    default it does not extend beyond the data range, since a map is read
    inside the experimental domain.

    params:
        levels (int): number of contour levels
        cmap (str): matplotlib colormap
        filled (bool): filled contours (contourf) or only lines (contour)
        (the rest as in surface_plot; file_name default {response}_contour.png)
    return:
        path (Path | None): saved file, None when only shown
    """
    logger.info("contour_plot()")
    # PROCESSING DATA
    X1, X2, Y = _grid_for_plot(
        coefficients, factors, vary, data, fixed, limits, extend, grid
    )
    xlabel, ylabel = _labels(vary, labels)

    # MAKING PLOT
    fig, ax = plt.subplots(figsize=(6, 5))
    draw = ax.contourf if filled else ax.contour
    contour = draw(X1, X2, Y, levels=levels, cmap=cmap)
    colorbar = fig.colorbar(contour, ax=ax, shrink=1, aspect=10)
    if response is not None:
        colorbar.set_label(response)

    if data is not None:
        ax.scatter(x=data[vary[0]], y=data[vary[1]], marker=".", c="red")
    if optimum is not None:
        x1, x2, _ = _optimum_xy(optimum, factors, vary)
        ax.scatter(x=x1, y=x2, marker="*", c="green", s=120)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # SAVE OR SHOW
    if file_name is None:
        file_name = f"{_stem(response or 'response')}_contour.png"
    return _finish(fig, folder, file_name, dpi)
