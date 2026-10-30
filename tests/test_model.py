"""Term names, design matrix and prediction of the quadratic model."""

import numpy as np
import pytest

from rsm_design.model import (
    design_matrix,
    n_factors_from_terms,
    n_terms,
    predict,
    quadratic_form,
    term_names,
)


def test_term_names_2_and_3_factors():
    assert term_names(2) == ["b0", "b1", "b2", "b11", "b12", "b22"]
    assert term_names(3) == [
        "b0",
        "b1",
        "b2",
        "b3",
        "b11",
        "b12",
        "b13",
        "b22",
        "b23",
        "b33",
    ]


def test_term_names_10_factors_unique():
    names = term_names(10)
    assert len(names) == 66
    assert len(set(names)) == 66
    assert "b1_10" in names


def test_design_matrix_equals_old_three_factor_formula():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(15, 3))
    x1, x2, x3 = X.T
    old = np.column_stack(
        [np.ones(15), x1, x2, x3, x1**2, x1 * x2, x1 * x3, x2**2, x2 * x3, x3**2]
    )
    np.testing.assert_allclose(design_matrix(X), old)


def test_predict_equals_quadratic_form():
    rng = np.random.default_rng(1)
    for n_factors in (2, 3, 4):
        coefficients = rng.normal(size=n_terms(n_factors))
        X = rng.uniform(-1, 1, size=(8, n_factors))
        b0, b, B = quadratic_form(coefficients)
        expected = [b0 + b @ x + x @ B @ x for x in X]
        np.testing.assert_allclose(predict(X, coefficients), expected)


def test_predict_wrong_coefficient_count():
    with pytest.raises(ValueError):
        predict(np.zeros((2, 3)), np.zeros(6))


@pytest.mark.parametrize("n", [2, 4, 5, 7, 100])
def test_n_factors_from_terms_rejects_invalid(n):
    with pytest.raises(ValueError):
        n_factors_from_terms(n)


@pytest.mark.parametrize("n_factors", [1, 2, 3, 10])
def test_n_factors_from_terms_round_trip(n_factors):
    assert n_factors_from_terms(n_terms(n_factors)) == n_factors
