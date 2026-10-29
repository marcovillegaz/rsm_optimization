"""Refit of the DLLME models of the paper: coefficients must not change."""

import pytest

from rsm_design.data import load_experiments
from rsm_design.model import fit_model

FACTORS = ["Extractant-Dispersant Ratio", "Sample volume", "Extraction mixture volume"]
TERMS = ["b0", "b1", "b2", "b3", "b11", "b12", "b13", "b22", "b23", "b33"]


def _load(path, response, scale=1.0):
    return load_experiments(
        path=path,
        factors=FACTORS,
        response=response,
        sheet="RSM_3",
        header_row=1,
        skip_rows=2,
        scale=scale,
    )


def _check(result, expected):
    for term in TERMS:
        if term in expected:
            assert result.coefficients[term] == pytest.approx(expected[term], rel=1e-6)
        else:
            assert result.coefficients[term] == 0.0


def test_enrichment_factor(dllme_xlsx):
    data_df = _load(dllme_xlsx, "Enrichment Factor")
    result = fit_model(
        df=data_df,
        factors=FACTORS,
        response="Enrichment Factor",
        drop_terms=["b2", "b12", "b13", "b23", "b22", "b33"],
    )
    expected = {
        "b0": 77.04742942964698,
        "b1": -47.02551840202079,
        "b3": -0.04967289354553598,
        "b11": 13.649456917604395,
    }
    _check(result, expected)
    assert result.r_squared == pytest.approx(0.803, abs=1e-3)
    assert int(result.ols.nobs) == 19


def test_recovery(dllme_xlsx):
    data_df = _load(dllme_xlsx, "Recovery", scale=100)
    result = fit_model(
        df=data_df,
        factors=FACTORS,
        response="Recovery",
        drop_terms=["b1", "b12", "b13", "b23", "b11", "b22"],
    )
    expected = {
        "b0": 27.485764944569105,
        "b2": -3.953133226741123,
        "b3": 0.32126863128094674,
        "b33": -0.0004231588706901417,
    }
    _check(result, expected)
