"""Tests of the plots: files are written, figures are closed, grids are right."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from matplotlib import pyplot as plt

from rsm_design import plots
from rsm_design.model import FitResult, design_matrix, predict, term_names

DPI_TEST = 50


def make_result(factors, seed=0, n=20):
    """FitResult built directly with OLS on synthetic quadratic data."""
    rng = np.random.default_rng(seed)
    k = len(factors)
    X = rng.uniform(-1, 1, size=(n, k))
    X[:3] = 0.0  # replicated centre point
    terms = term_names(k)
    true = rng.normal(size=len(terms))
    y = design_matrix(X) @ true + rng.normal(scale=0.1, size=n)
    D = pd.DataFrame(design_matrix(X), columns=terms)
    ols = sm.OLS(y, D).fit()
    data = pd.DataFrame(X, columns=factors)
    data["Enrichment Factor"] = y
    result = FitResult(
        response="Enrichment Factor",
        factors=tuple(factors),
        terms=tuple(terms),
        dropped_terms=(),
        ols=ols,
        coefficients=pd.Series(ols.params.to_numpy(), index=terms),
        X=D,
        y=pd.Series(y),
    )
    return result, data


@pytest.fixture(params=[["A", "B"], ["A", "B", "C"]], ids=["2f", "3f"])
def fit(request):
    return make_result(request.param)


def assert_png(path):
    assert path is not None and path.exists() and path.stat().st_size > 0
    assert path.read_bytes()[:4] == b"\x89PNG"
    assert plt.get_fignums() == []


def test_diagnostics(fit, tmp_path):
    result, _ = fit
    paths = plots.plot_diagnostics(result, folder=tmp_path / "new", dpi=DPI_TEST)
    assert [p.name for p in paths] == [
        "EnrichmentFactor_coeff_plot.png",
        "EnrichmentFactor_rsd_plot.png",
        "EnrichmentFactor_prediction_plot.png",
    ]
    for path in paths:
        assert_png(path)


@pytest.mark.parametrize("whiskers", ["ci", "stderr"])
@pytest.mark.parametrize("normalize", [False, True])
def test_coeff_plot_variants(fit, tmp_path, whiskers, normalize):
    result, _ = fit
    path = plots.coeff_plot(
        result, whiskers=whiskers, normalize=normalize, folder=tmp_path, dpi=DPI_TEST
    )
    assert_png(path)


def test_coeff_plot_bad_whiskers(fit, tmp_path):
    with pytest.raises(ValueError):
        plots.coeff_plot(fit[0], whiskers="nope", folder=tmp_path)
    assert plt.get_fignums() == []


def test_residual_plot_not_normalized(fit, tmp_path):
    path = plots.residual_plot(fit[0], normalize=False, folder=tmp_path, dpi=DPI_TEST)
    assert_png(path)


def test_prediction_identity_line(fit, monkeypatch):
    captured = {}
    monkeypatch.setattr(plt, "show", lambda: captured.setdefault("fig", plt.gcf()))
    assert plots.prediction_plot(fit[0], folder=None) is None
    ax = captured["fig"].axes[0]
    xs, ys = ax.lines[0].get_data()
    assert np.allclose(xs, ys)
    assert plt.get_fignums() == []


def test_response_grid_shapes_and_fixed():
    factors = ["A", "B", "C"]
    coeff = pd.Series(np.arange(1, 11, dtype=float), index=term_names(3))
    limits = {"A": (0, 2), "C": (-1, 1)}
    X1, X2, Y = plots.response_grid(
        coeff, factors, vary=("A", "C"), fixed={"B": 0.7}, limits=limits, grid=7
    )
    assert X1.shape == X2.shape == Y.shape == (7, 7)
    i, j = 2, 5
    expected = predict(np.array([X1[i, j], 0.7, X2[i, j]]), coeff)[0]
    assert Y[i, j] == pytest.approx(expected)
    assert X1[0, 0] == 0 and X1[0, -1] == 2 and X2[0, 0] == -1 and X2[-1, 0] == 1


def test_defaults_centre_and_extension(fit):
    result, data = fit
    factors = list(result.factors)
    fixed, limits = plots._resolve_grid_inputs(
        factors, tuple(factors[:2]), data, None, None, 0.25
    )
    for name in factors[2:]:
        assert fixed[name] == 0.0  # mode = replicated centre point
    lo, hi = data["A"].min(), data["A"].max()
    assert limits["A"][0] == pytest.approx(lo - 0.25 * (hi - lo))


def test_surface_and_contour(fit, tmp_path):
    result, data = fit
    factors = list(result.factors)
    optimum = SimpleNamespace(x=np.zeros(len(factors)), response=1.0)
    common = dict(
        coefficients=result.coefficients,
        factors=factors,
        vary=("A", "B"),
        data=data,
        response="Enrichment Factor",
        optimum=optimum,
        labels={"A": "A (mL)"},
        grid=20,
        folder=tmp_path,
        dpi=DPI_TEST,
    )
    surface = plots.surface_plot(**common)
    contour = plots.contour_plot(**common)
    assert surface.name == "EnrichmentFactor_surface.png"
    assert contour.name == "EnrichmentFactor_contour.png"
    assert_png(surface)
    assert_png(contour)


def test_contour_lines_without_data(tmp_path):
    result, _ = make_result(["A", "B"])
    path = plots.contour_plot(
        result.coefficients,
        ["A", "B"],
        ("A", "B"),
        limits={"A": (-1, 1), "B": (-1, 1)},
        filled=False,
        grid=10,
        folder=tmp_path,
        file_name="lines.png",
        dpi=DPI_TEST,
    )
    assert path.name == "lines.png"
    assert_png(path)


def test_show_when_no_folder(fit, monkeypatch):
    result, data = fit
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    out = plots.surface_plot(
        result.coefficients, list(result.factors), ("A", "B"), data=data, grid=10
    )
    assert out is None and shown == [True]
    assert plt.get_fignums() == []
