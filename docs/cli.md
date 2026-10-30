# Command-line interface

Installed as `rsm-design`; also available as `python -m rsm_design`.

```text
rsm-design {fit,optimize,plot} DATA [options]
```

## Shared options

| Flag | Meaning |
|---|---|
| `DATA` | Path to an `.xlsx` or delimited text file |
| `--sheet` | Excel sheet name or index (default: first sheet) |
| `--factors F1 F2 ...` | Column names of the factors (at least one) |
| `--response R` | Column name of the response |
| `--header-row N` | Zero-based row holding column names (default 0) |
| `--skip-rows N` | Rows to skip right after the header, e.g. Notation/Units (default 0) |
| `--scale X` | Multiply the response by X (default 1.0; use 100 for fractions to percent) |
| `--sep` | Field separator for CSV/text input |
| `--decimal` | Decimal mark for CSV/text input |
| `-v`, `-q` | More / less logging |

## fit

```text
rsm-design fit DATA ... [--drop b1 b12 ...] [--out DIR] [--no-plots]
                        [--whiskers ci|stderr] [--allow-rank-deficient] [--dpi N]
```

| Flag | Meaning |
|---|---|
| `--drop` | Term names to remove from the model (see [term naming](data_format.md#term-naming)) |
| `--out DIR` | Output folder |
| `--no-plots` | Skip the three diagnostic PNGs |
| `--whiskers ci\|stderr` | Error bars in the coefficient plot |
| `--allow-rank-deficient` | Warn instead of failing when the design cannot estimate the model |
| `--dpi N` | Figure resolution |

Writes `{stem}_model_params.csv`, `{stem}_model_stats.txt` and (unless `--no-plots`) the coefficient, residual and prediction plots, where `{stem}` is the response name without spaces. Logs R2, adjusted R2 and RMSE.

## optimize

```text
rsm-design optimize DATA ... --coefficients CSV [--minimize]
                             [--region ellipsoid|box] [--center mode|midrange] [--out DIR]
```

| Flag | Meaning |
|---|---|
| `--coefficients CSV` | Coefficient file written by `fit` |
| `--minimize` | Minimise instead of maximise |
| `--region` | Feasible region: `ellipsoid` (default) or `box` |
| `--center` | Region centre: `mode` of the data or `midrange` |
| `--out DIR` | Writes `{stem}_optimum.csv` |

Prints the optimum factor levels and predicted response.

## plot

```text
rsm-design plot DATA ... --coefficients CSV --vary F1 F2 [--fixed "F3=0.5" ...]
                         [--kind surface|contour|both] [--extend X] [--no-points]
                         [--optimum] [--azim A --elev E] [--out DIR | --show]
```

| Flag | Meaning |
|---|---|
| `--vary F1 F2` | The two factors on the axes |
| `--fixed "F=value"` | Value of each remaining factor (quote it) |
| `--kind` | `surface`, `contour` or `both` |
| `--extend X` | Fractional margin beyond the data range |
| `--no-points` | Hide the experimental points |
| `--optimum` | Mark the constrained optimum |
| `--azim`, `--elev` | 3-D view angles |
| `--out DIR` / `--show` | Save PNGs, or open a window |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Known error (unknown factor, missing file, rank-deficient design, ...): one-line message on stderr |

The Agg backend is used unless `--show` is given.

## PowerShell quoting

Quote any name containing spaces; a `--fixed` assignment is one quoted token. The backtick continues a line.

```powershell
rsm-design fit data\dllme\experimental_results.xlsx --sheet RSM_3 `
    --header-row 1 --skip-rows 2 `
    --factors "Extractant-Dispersant Ratio" "Sample volume" "Extraction mixture volume" `
    --response "Enrichment Factor" --drop b2 b12 b13 b23 b22 b33 --out results

rsm-design plot data\dllme\experimental_results.xlsx --sheet RSM_3 `
    --header-row 1 --skip-rows 2 `
    --factors "Extractant-Dispersant Ratio" "Sample volume" "Extraction mixture volume" `
    --response "Enrichment Factor" --coefficients "results\EnrichmentFactor_model_params.csv" `
    --vary "Extractant-Dispersant Ratio" "Sample volume" `
    --fixed "Extraction mixture volume=0.5" --out results
```

In bash/zsh use `\` for line continuation and the same quoting.

Back to [index](index.md).
