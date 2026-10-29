"""Reading experiments and reading/writing model coefficients.

Experiments come from an Excel sheet or a CSV file. Coefficients are stored as a
headerless name,value table, the format used by the original scripts.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from rsm_design.model import n_factors_from_terms, term_names

logger = logging.getLogger(__name__)


########################################################################
# EXPERIMENTS
########################################################################


def read_table(
    path: str | Path,
    sheet: str | int = 0,
    header_row: int = 0,
    sep: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """Read a table from an Excel workbook or a CSV file (by file extension).

    params:
        path (str | Path): .xlsx/.xlsm/.xls or .csv/.txt file
        sheet (str | int): sheet name or position (Excel only)
        header_row (int): row index with the column names
        sep (str): CSV separator
        decimal (str): CSV decimal mark
    return:
        data_df (pd.DataFrame): the table, with the rows below the header
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet, header=header_row)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path, header=header_row, sep=sep, decimal=decimal)
    raise ValueError(f"Unsupported file type {suffix!r}; use .xlsx or .csv")


def _check_columns(data_df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise KeyError listing the available columns and the close matches."""
    available = [str(c) for c in data_df.columns]
    for column in columns:
        if column in available:
            continue
        close = difflib.get_close_matches(column, available, n=3, cutoff=0.6)
        hint = f" Did you mean {close}?" if close else ""
        raise KeyError(
            f"Column {column!r} not found.{hint} Available columns: {available}"
        )


def load_experiments(
    path: str | Path,
    factors: Sequence[str],
    response: str | None = None,
    sheet: str | int = 0,
    header_row: int = 0,
    skip_rows: int = 0,
    scale: float = 1.0,
    sep: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """Load the factors and the response of an experimental design.

    Sheets often carry extra rows under the header (notation, units) that must
    not enter the regression; skip_rows drops them. The response can be
    rescaled, e.g. scale=100 turns a recovery fraction into percent.

    params:
        path (str | Path): Excel or CSV file
        factors (list): names of the factor columns
        response (str): name of the response column; None loads only factors
        sheet (str | int): sheet name or position (Excel only)
        header_row (int): row index with the column names
        skip_rows (int): rows right below the header to drop
        scale (float): the response is multiplied by this number
        sep (str): CSV separator
        decimal (str): CSV decimal mark
    return:
        data_df (pd.DataFrame): numeric factors (+ response) without NaN rows
    """
    factors = list(factors)
    columns = factors if response is None else factors + [response]

    data_df = read_table(
        path=path, sheet=sheet, header_row=header_row, sep=sep, decimal=decimal
    )
    _check_columns(data_df=data_df, columns=columns)

    data_df = data_df.iloc[skip_rows:]  # drop notation and units rows
    data_df = data_df[columns]
    data_df = data_df.apply(pd.to_numeric, errors="coerce").astype(float)
    if response is not None:
        data_df[response] = data_df[response] * scale

    n_before = len(data_df)
    data_df = data_df.dropna()
    n_dropped = n_before - len(data_df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with missing or non-numeric values.")

    logger.info(f"Loaded {len(data_df)} experiments from {Path(path).name}.")
    return data_df


def read_units(
    path: str | Path,
    columns: Sequence[str],
    units_row: int,
    sheet: str | int = 0,
    header_row: int = 0,
    sep: str = ",",
    decimal: str = ".",
) -> dict[str, str]:
    """Units of some columns, read from a row of the sheet (for axis labels).

    params:
        path (str | Path): Excel or CSV file
        columns (list): names of the columns whose units are wanted
        units_row (int): row index (counted from the first row below the
            header) holding the units; the DLLME sheets use 1
        sheet (str | int): sheet name or position (Excel only)
        header_row (int): row index with the column names
        sep (str): CSV separator
        decimal (str): CSV decimal mark
    return:
        units (dict): column name -> unit text; "" if empty or "-"
    """
    columns = list(columns)
    data_df = read_table(
        path=path, sheet=sheet, header_row=header_row, sep=sep, decimal=decimal
    )
    _check_columns(data_df=data_df, columns=columns)

    units = {}
    for column in columns:
        value = data_df[column].iloc[units_row]
        text = "" if pd.isna(value) else str(value).strip()
        units[column] = "" if text == "-" else text
    return units


########################################################################
# COEFFICIENTS
########################################################################


def save_coefficients(coefficients: pd.Series, path: str | Path) -> Path:
    """Save the coefficients as a headerless name,value CSV.

    params:
        coefficients (pd.Series): values indexed by term name (dropped = 0.0)
        path (str | Path): output file
    return:
        path (Path): the written file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(path, header=False)
    return path


def load_coefficients(path: str | Path, n_factors: int | None = None) -> pd.Series:
    """Load a coefficient CSV, with or without header, and check its names.

    The header is detected because the value of the first row is not a number.

    params:
        path (str | Path): CSV with name,value rows
        n_factors (int): expected number of factors; None infers it from the
            number of rows
    return:
        coefficients (pd.Series): values indexed by term name, in term order
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_df = pd.read_csv(path, header=None)
    if raw_df.shape[1] < 2:
        raise ValueError(f"{path.name} must have two columns: name,value")

    # DETECT HEADER
    first_value = pd.to_numeric(raw_df.iloc[0, 1], errors="coerce")
    if pd.isna(first_value):
        raw_df = raw_df.iloc[1:]

    names = [str(n).strip() for n in raw_df.iloc[:, 0]]
    values = pd.to_numeric(raw_df.iloc[:, 1], errors="raise").astype(float)
    coefficients = pd.Series(values.to_numpy(), index=names)

    # VALIDATE NAMES
    if n_factors is None:
        n_factors = n_factors_from_terms(len(coefficients))
    expected = term_names(n_factors)
    if names != expected:
        raise ValueError(
            f"Coefficient names in {path.name} are {names} but a model with "
            f"{n_factors} factors has {expected}"
        )
    return coefficients
