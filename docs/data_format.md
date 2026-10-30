# Data format

## Experiment spreadsheet

One sheet per experiment. Each row is a run; columns are factors and
responses. The DLLME workbook (`data/dllme/experimental_results.xlsx`) uses:

| Row | Content |
|---|---|
| 0 | Title / blank |
| 1 | **Column names** (header) |
| 2 | Notation (e.g. `X1`) |
| 3 | Units (e.g. `mL`) |
| 4+ | Data |

Read it with `header_row=1, skip_rows=2`:

- `header_row`: zero-based row index holding the column names.
- `skip_rows`: number of rows removed immediately after the header (Notation and Units here). `read_units(...)` can recover the units row for axis labels.
- `scale`: multiplier applied to the response after loading (`scale=100` turns a recovery fraction into percent).

A plain table with names in the first row uses the defaults (`header_row=0, skip_rows=0`). CSV or text files accept `sep` and `decimal`.

## Term naming

Coefficients are named by position in the quadratic model with N factors:

1. `b0` intercept
2. `b1 ... bN` linear terms
3. Pairs `(i, j)` with `i <= j` in `combinations_with_replacement` order: `b11, b12, ..., b1N, b22, b23, ..., bNN`. This equals the scikit-learn `PolynomialFeatures(degree=2)` order.

For 3 factors: `b0 b1 b2 b3 b11 b12 b13 b22 b23 b33`.

With 10 or more factors an underscore separates indices (`b1_10`, `b2_10`) to avoid ambiguity.

## Coefficient CSV

Written by `save_results` / `fit`, read by `load_coefficients` / `--coefficients`:

```text
b0,77.047
b1,-47.026
b2,0.0
...
```

- Headerless `name,value` rows, full term list in model order.
- Terms removed with `drop_terms` are stored as `0.0`.
- The loader also accepts a file with a header row, and validates the names against the number of factors.

Back to [index](index.md).
