# Methodology

## Quadratic model

For factors x1..xN the response is modelled as

```text
y = b0 + sum_i b_i x_i + sum_{i<=j} b_ij x_i x_j + error
```

fitted by ordinary least squares (`statsmodels.OLS`), which provides the
coefficient table, ANOVA, confidence intervals and t-values. Term names and
ordering are described in [data format](data_format.md#term-naming). Terms can
be pruned with `drop_terms`; the stored coefficient vector keeps all terms,
with dropped ones set to `0.0`.

## Design-rank check

A quadratic model with N factors has (N+1)(N+2)/2 coefficients, but the design
must contain enough distinct factor levels to estimate them. `fit_model` builds
the design matrix and compares its rank with the number of terms
(`check_estimability`), raising `RankDeficientDesignError` (or warning with
`on_rank_deficient="warn"`) and naming the terms that cannot be separated.

Example: a duplicated 2^2 factorial has only 4 distinct points and two levels
per factor. With two levels, x^2 takes the same two values as a function of x,
so the column x^2 is a linear combination of the constant column and the
column x. The full quadratic model has 6 terms but the design matrix has rank
4, so `b11` and `b22` are not identifiable. Replicating the runs adds degrees
of freedom for the error estimate, not new levels. The estimable model is
obtained with `drop_terms=["b11", "b22"]`. A central composite design fixes
this by adding axial and centre points (three or more levels per factor).

## Validity ellipsoid

A CCD explores a roughly spherical region around its centre. Outside that
region the polynomial is an extrapolation and the predicted optimum is not
supported by data; quadratic surfaces in particular often have optima far away
(saddles or ridges). `DesignRegion.from_data` builds an ellipsoid centred on
the design centre (the modal point, or the mid-range when there are no centre
replicates, with a warning) with semi-axes taken from the factor ranges. The
optimum must stay inside it. `shape="box"` uses plain bounds instead.

## Optimiser: SLSQP vs the legacy penalty

The 2024 scripts minimised the negative response with L-BFGS-B and added a
penalty of 1e6 when the point left the ellipsoid. The penalty creates a
discontinuity, which can stall a gradient-based optimiser.

`optimize` uses SLSQP with the ellipsoid as a real inequality constraint, plus
bounds and an analytic gradient, started from several points (`n_starts`).
`stationary_point()` provides the canonical analysis of the unconstrained
surface.

**Optimum values may differ from the 2024 scripts.** The old procedure could
stop early at a penalised point; the new one converges to the constrained
optimum. This is a correction, not a regression. Fitted coefficients are
unchanged and are checked against the original results in the test suite.

Back to [index](index.md).
