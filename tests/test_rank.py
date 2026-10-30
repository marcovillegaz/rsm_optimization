"""Rank check of the design before fitting."""

import itertools

import numpy as np
import pandas as pd
import pytest

from rsm_design.model import RankDeficientDesignError, fit_model


@pytest.fixture
def duplicated_factorial() -> pd.DataFrame:
    """2^2 factorial run twice: only 4 distinct points for 6 terms."""
    points = list(itertools.product([-1.0, 1.0], repeat=2)) * 2
    df = pd.DataFrame(points, columns=["A", "B"])
    df["y"] = [1.0, 2.0, 3.5, 4.0, 1.2, 2.1, 3.4, 4.2]
    return df


def test_duplicated_factorial_raises(duplicated_factorial):
    with pytest.raises(RankDeficientDesignError) as error:
        fit_model(df=duplicated_factorial, factors=["A", "B"], response="y")
    message = str(error.value)
    assert "b11" in message and "b22" in message
    assert "drop_terms" in message


def test_duplicated_factorial_warns(duplicated_factorial):
    with pytest.warns(UserWarning, match="b22"):
        fit_model(
            df=duplicated_factorial,
            factors=["A", "B"],
            response="y",
            on_rank_deficient="warn",
        )


def test_dropping_aliased_terms_fits(duplicated_factorial):
    result = fit_model(
        df=duplicated_factorial,
        factors=["A", "B"],
        response="y",
        drop_terms=["b11", "b22"],
    )
    assert result.coefficients[["b11", "b22"]].eq(0.0).all()
    assert len(result.ols.params) == 4


def test_ccd_two_factors_fits():
    a = np.sqrt(2)
    points = [(-1, -1), (1, -1), (-1, 1), (1, 1), (-a, 0), (a, 0), (0, -a), (0, a)]
    points += [(0, 0)] * 3
    df = pd.DataFrame(points, columns=["A", "B"])
    df["y"] = 5 + df.A - 2 * df.B + 0.5 * df.A * df.B - df.A**2 + 0.1 * df.B**2
    result = fit_model(df=df, factors=["A", "B"], response="y")
    assert result.coefficients["b0"] == pytest.approx(5.0)
    assert result.coefficients["b12"] == pytest.approx(0.5)
    assert result.r_squared == pytest.approx(1.0)


def test_unknown_drop_term_rejected(duplicated_factorial):
    with pytest.raises(ValueError, match="b99"):
        fit_model(
            df=duplicated_factorial,
            factors=["A", "B"],
            response="y",
            drop_terms=["b99"],
        )
