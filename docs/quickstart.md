# Quickstart

This example uses the public DLLME workbook, sheet `RSM_3`.

```python
from rsm_design import (
    load_experiments, fit_model, save_results,
    DesignRegion, optimize, surface_plot, contour_plot,
)

path = "data/dllme/experimental_results.xlsx"
factors = ["Extractant-Dispersant Ratio", "Sample volume", "Extraction mixture volume"]
response = "Enrichment Factor"

# 1. Load: row 1 holds the headers, the next two rows (Notation, Units) are skipped
df = load_experiments(
    path, factors=factors, response=response,
    sheet="RSM_3", header_row=1, skip_rows=2,
)

# 2. Fit the full quadratic model, optionally pruning terms
result = fit_model(df, factors, response, drop_terms=["b2", "b12", "b13", "b23", "b22", "b33"])
print(result.r_squared, result.adj_r_squared, result.rmse)
print(result.ols.summary())

# 3. Write coefficients CSV, statistics TXT and diagnostic PNGs
paths = save_results(result, "results", plots=True)

# 4. Optimise inside the validity ellipsoid
region = DesignRegion.from_data(df, factors, center="mode", shape="ellipsoid")
best = optimize(result.coefficients, region, maximize=True)

# 5. Plot
surface_plot(result.coefficients, factors, vary=factors[:2], optimum=best, folder="results")
contour_plot(result.coefficients, factors, vary=factors[:2], optimum=best, folder="results")
```

Notes:

- For a percentage response stored as a fraction, pass `scale=100` to `load_experiments` (used for Recovery in the DLLME study).
- If the design cannot estimate the model, `fit_model` raises `RankDeficientDesignError`; see [methodology](methodology.md#design-rank-check).
- The exact optional arguments of the plot functions are listed in the [API reference](api.md).

See also: [CLI](cli.md), [data format](data_format.md). Back to [index](index.md).
