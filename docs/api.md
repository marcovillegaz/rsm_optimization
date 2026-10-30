# API reference

All public names are re-exported from `rsm_design`. Signatures follow the
design plan; see the source docstrings for the final word.

## `rsm_design.model`

| Name | Description |
|---|---|
| `term_names(n_factors, prefix="b")` | `["b0", "b1", ..., "b11", "b12", ...]`; `b1_10` style for N >= 10 |
| `n_terms(n_factors)` / `n_factors_from_terms(n)` | Term count and its inverse |
| `design_matrix(X)` | Quadratic design matrix (numpy) |
| `predict(X, coefficients)` | Model response for N factors |
| `quadratic_form(coefficients)` | Returns `(b0, b, B)` for `y = b0 + b.x + x'Bx` |
| `check_estimability(X_design, terms)` | List of non-estimable term names |
| `RankDeficientDesignError` | `ValueError` raised for non-estimable designs |
| `FitResult` | Frozen dataclass: `response, factors, terms, dropped_terms, ols, coefficients, X, y`; properties `r_squared, adj_r_squared, rmse, residuals, fitted` |
| `fit_model(df, factors, response, drop_terms=(), on_rank_deficient="raise")` | Fit by OLS; `"warn"` is the alternative; unknown term names are rejected |
| `save_results(result, folder, plots=True, dpi=DPI)` | Writes params CSV, stats TXT, plots; returns `dict[str, Path]` |
| `file_stem(response)` | Response name without spaces, used in file names |

## `rsm_design.data`

| Name | Description |
|---|---|
| `read_table(path, ...)` | Read xlsx or csv depending on extension |
| `load_experiments(path, factors, response=None, sheet=0, header_row=0, skip_rows=0, scale=1.0, sep, decimal)` | Load runs into a DataFrame |
| `read_units(...)` | Units row for axis labels |
| `save_coefficients(coefficients, path)` | Headerless `name,value` CSV |
| `load_coefficients(path)` | Reads with or without header; validates names |

## `rsm_design.optimize`

| Name | Description |
|---|---|
| `DesignRegion` | Feasible region; `DesignRegion.from_data(df, factors, center="mode"\|"midrange", shape="ellipsoid"\|"box")` |
| `OptimizationResult` | Optimum factors and predicted response |
| `optimize(coefficients, region, maximize=True, n_starts=None)` | Multi-start SLSQP with ellipsoid constraint |
| `stationary_point(...)` | Optional canonical analysis |

## `rsm_design.plots`

| Name | Description |
|---|---|
| `coeff_plot` | Coefficients with CI/standard-error bars; normalised version shows t-values with +/- t-critical |
| `residual_plot` | Residual histogram with scaled normal curve |
| `prediction_plot` | Predicted vs experimental with the y = x line |
| `plot_diagnostics(result, folder)` | The three plots above |
| `response_grid(coefficients, factors, vary, fixed, limits, grid)` | 2-D meshgrid of predictions |
| `surface_plot(...)`, `contour_plot(...)` | Generic 3-D and contour plots with `fixed`, `extend=0.25`, `optimum`, unit `labels` |

Figures are saved when a folder is given, otherwise shown; they are closed afterwards.

## `rsm_design.univariate`

| Name | Description |
|---|---|
| `load_groups(path, sheet, group_column, response_column, header_row=1, skip_rows=2)` | Grouped data |
| `one_way_anova(...)` | Returns `UnivariateResult` (ANOVA + Tukey) |
| `group_boxplot(...)` | Box plot by group |

## `rsm_design.cli`

`main(argv=None) -> int`. See [CLI](cli.md).

Back to [index](index.md).
