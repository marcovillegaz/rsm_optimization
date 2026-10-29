"""Tests for the constrained optimizer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsm_design.model import predict, term_names
from rsm_design.optimize import DesignRegion, optimize, stationary_point

DLLME_FACTORS = [
    "Extractant-Dispersant Ratio",
    "Sample volume",
    "Extraction mixture volume",
]


def make_coefficients(n_factors, peak, curvature=-1.0):
    """y = 10 + curvature * sum((x - peak)^2) in the term_names order."""
    peak = np.asarray(peak, dtype=float)
    coefficients = pd.Series(0.0, index=term_names(n_factors))
    coefficients["b0"] = 10 + curvature * np.sum(peak**2)
    for i in range(n_factors):
        coefficients[f"b{i + 1}"] = -2 * curvature * peak[i]
        coefficients[f"b{i + 1}{i + 1}"] = curvature
    return coefficients


def make_region(n_factors, shape="ellipsoid"):
    """Region centred at 0 with semi-axis 1 (CCD-like data with replicates)."""
    rows = [np.zeros(n_factors)] * 3
    for i in range(n_factors):
        for s in (1.0, -1.0):
            row = np.zeros(n_factors)
            row[i] = s
            rows.append(row)
    df = pd.DataFrame(rows, columns=[f"x{i + 1}" for i in range(n_factors)])
    return DesignRegion.from_data(df, list(df.columns), shape=shape)


@pytest.mark.parametrize("n_factors", [2, 3])
def test_peak_outside_ends_on_boundary(n_factors):
    coefficients = make_coefficients(n_factors, np.full(n_factors, 3.0))
    region = make_region(n_factors)
    result = optimize(coefficients, region)
    assert region.ellipsoid_norm(result.x) <= 1 + 1e-6
    assert region.ellipsoid_norm(result.x) == pytest.approx(1.0, abs=1e-4)
    assert result.inside_region


@pytest.mark.parametrize("n_factors", [2, 3])
def test_peak_inside_found(n_factors):
    peak = np.full(n_factors, 0.3)
    coefficients = make_coefficients(n_factors, peak)
    result = optimize(coefficients, make_region(n_factors))
    assert np.allclose(result.x, peak, atol=1e-4)
    assert np.allclose(result.x, stationary_point(coefficients)[0])
    assert result.response == pytest.approx(10.0, abs=1e-6)


def test_minimize():
    coefficients = -make_coefficients(2, [0.2, -0.4])  # Convex, minimum at peak
    result = optimize(coefficients, make_region(2), maximize=False)
    assert np.allclose(result.x, [0.2, -0.4], atol=1e-4)


def test_box_respects_bounds():
    coefficients = make_coefficients(2, [3.0, 3.0])
    region = make_region(2, shape="box")
    result = optimize(coefficients, region)
    assert np.allclose(result.x, [1.0, 1.0], atol=1e-6)  # Corner, outside ellipsoid
    assert np.all(result.x <= region.upper + 1e-9)


def test_midrange_fallback_warns(caplog):
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 3.0, 5.0]})
    with caplog.at_level("WARNING"):
        region = DesignRegion.from_data(df, ["a", "b"])
    assert np.allclose(region.center, [1.0, 3.0])
    assert "midrange" in caplog.text


def test_dllme(dllme_xlsx):
    raw_df = pd.read_excel(dllme_xlsx, sheet_name="RSM_3", header=1)
    raw_df = raw_df.drop(raw_df.index[:2])  # Drop the units rows
    data_df = raw_df[DLLME_FACTORS].apply(pd.to_numeric)
    coeff_file = (
        Path(__file__).resolve().parents[1]
        / "examples/dllme/results/rsm_simplified/EnrichmentFactor_model_params.csv"
    )
    coefficients = pd.read_csv(coeff_file, header=None, index_col=0)[1]

    region = DesignRegion.from_data(data_df, DLLME_FACTORS)
    result = optimize(coefficients, region)
    print("DLLME optimum:", result.factors, result.response)

    assert region.contains(result.x, tol=1e-6)  # SLSQP tolerance
    centre_response = predict(region.center, coefficients)[0]
    assert result.response >= centre_response
