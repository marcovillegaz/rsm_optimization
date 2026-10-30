"""Loading experiments and coefficient files."""

import logging

import numpy as np
import pandas as pd
import pytest

from rsm_design.data import (
    load_coefficients,
    load_experiments,
    read_units,
    save_coefficients,
)
from rsm_design.model import file_stem, fit_model, save_results, term_names


@pytest.fixture
def sheet_df() -> pd.DataFrame:
    """Notation row, units row and 4 experiments under the header."""
    return pd.DataFrame(
        {
            "A": ["x1", "mL", 1.0, 2.0, 3.0, 4.0],
            "B": ["x2", "uL", 10.0, 20.0, 30.0, 40.0],
            "Rec": ["-", "%", 0.1, 0.2, 0.3, 0.4],
        }
    )


@pytest.fixture
def xlsx_path(tmp_path, sheet_df):
    path = tmp_path / "data.xlsx"
    sheet_df.to_excel(path, index=False, sheet_name="S1")
    return path


@pytest.fixture
def csv_path(tmp_path, sheet_df):
    path = tmp_path / "data.csv"
    sheet_df.to_csv(path, index=False)
    return path


def test_csv_and_xlsx_agree(xlsx_path, csv_path):
    from_xlsx = load_experiments(
        path=xlsx_path, factors=["A", "B"], response="Rec", sheet="S1", skip_rows=2
    )
    from_csv = load_experiments(
        path=csv_path, factors=["A", "B"], response="Rec", skip_rows=2
    )
    pd.testing.assert_frame_equal(from_xlsx, from_csv)
    assert len(from_csv) == 4
    assert from_csv["Rec"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_scale_multiplies_response(csv_path):
    data_df = load_experiments(
        path=csv_path, factors=["A", "B"], response="Rec", skip_rows=2, scale=100
    )
    np.testing.assert_allclose(data_df["Rec"], [10, 20, 30, 40])
    np.testing.assert_allclose(data_df["A"], [1, 2, 3, 4])


def test_nan_rows_dropped_with_warning(tmp_path, caplog):
    path = tmp_path / "nan.csv"
    pd.DataFrame({"A": [1, 2, "bad"], "Rec": [1, np.nan, 3]}).to_csv(path, index=False)
    with caplog.at_level(logging.WARNING, logger="rsm_design.data"):
        data_df = load_experiments(path=path, factors=["A"], response="Rec")
    assert len(data_df) == 1
    assert "2 rows" in caplog.text


def test_read_units(csv_path):
    units = read_units(path=csv_path, columns=["A", "B", "Rec"], units_row=1)
    assert units == {"A": "mL", "B": "uL", "Rec": "%"}


def test_bad_column_raises_keyerror(csv_path):
    with pytest.raises(KeyError, match="Available columns"):
        load_experiments(path=csv_path, factors=["A", "C"], response="Rec")
    with pytest.raises(KeyError, match="Did you mean"):
        load_experiments(path=csv_path, factors=["A", "B"], response="Rec ")


def test_unsupported_extension(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        load_experiments(path=path, factors=["A"])


def test_coefficients_round_trip_headerless(tmp_path):
    coefficients = pd.Series(np.arange(6) * 0.5 + 0.25, index=term_names(2))
    path = save_coefficients(coefficients=coefficients, path=tmp_path / "c.csv")
    assert path.read_text().splitlines()[0].startswith("b0,")  # no header
    loaded = load_coefficients(path=path)
    pd.testing.assert_series_equal(loaded, coefficients, check_names=False)


def test_coefficients_with_header(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("term,value\nb0,1.5\nb1,2\nb2,3\nb11,0\nb12,0\nb22,-1\n")
    loaded = load_coefficients(path=path, n_factors=2)
    assert loaded["b22"] == -1.0
    assert list(loaded.index) == term_names(2)


def test_coefficients_wrong_names(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("b0,1\nb1,2\nb2,3\nb11,0\nb22,0\nb12,0\n")
    with pytest.raises(ValueError, match="names"):
        load_coefficients(path=path)
    with pytest.raises(ValueError):
        load_coefficients(path=path, n_factors=3)


def test_file_stem():
    assert file_stem("Enrichment Factor") == "EnrichmentFactor"


def test_save_results_without_plots(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.uniform(-1, 1, size=(12, 2)), columns=["A", "B"])
    df["my y"] = 1 + df.A - df.B**2 + rng.normal(scale=0.01, size=12)
    result = fit_model(df=df, factors=["A", "B"], response="my y", drop_terms=["b12"])
    paths = save_results(result=result, folder=tmp_path / "out", plots=False)
    assert paths["params"].name == "myy_model_params.csv"
    assert paths["stats"].exists()
    assert "b12" in paths["stats"].read_text()
    loaded = load_coefficients(path=paths["params"])
    pd.testing.assert_series_equal(loaded, result.coefficients, check_names=False)
